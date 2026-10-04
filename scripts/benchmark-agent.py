#!/usr/bin/env python3
"""Measure Oreo harness quality, latency, token use, and live Pollen cost."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

CATALOG_URL = "https://gen.pollinations.ai/text/models"
METRICS_PREFIX = "agent_metrics="
MAX_FIXTURES = 32
MAX_MODELS = 8
MAX_REPETITIONS = 10


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    prompt: str
    min_tool_calls: int
    max_tool_calls: int
    max_response_bytes: int
    required_any: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("self-test", help="run dependency-free parser tests")

    run = subcommands.add_parser("run", help="run live candidate comparisons")
    run.add_argument("--model", action="append", required=True)
    run.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "tests/agent/fixtures.json",
    )
    run.add_argument(
        "--binary",
        type=Path,
        default=repo_root / "target/debug/elixpo",
    )
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--timeout-seconds", type=int, default=60)
    run.add_argument(
        "--output",
        type=Path,
        default=repo_root / "target/agent-bench/report.json",
    )
    return parser.parse_args()


def load_fixtures(path: Path) -> list[Fixture]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("fixture manifest schema_version must be 1")
    entries = payload.get("fixtures")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_FIXTURES:
        raise ValueError(f"fixture count must be 1-{MAX_FIXTURES}")
    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for entry in entries:
        fixture_id = entry.get("id")
        prompt = entry.get("prompt")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError("fixture id must be a non-empty string")
        if fixture_id in seen:
            raise ValueError(f"duplicate fixture id: {fixture_id}")
        seen.add(fixture_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"fixture {fixture_id} prompt is empty")
        if len(prompt.encode("utf-8")) > 4_096:
            raise ValueError(f"fixture {fixture_id} prompt is too large")
        minimum = entry.get("min_tool_calls")
        maximum = entry.get("max_tool_calls")
        max_bytes = entry.get("max_response_bytes")
        required = entry.get("required_any", [])
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or not 0 <= minimum <= maximum <= 4
        ):
            raise ValueError(f"fixture {fixture_id} tool bounds are invalid")
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 8_192:
            raise ValueError(f"fixture {fixture_id} response bound is invalid")
        if not isinstance(required, list) or not all(
            isinstance(term, str) and term for term in required
        ):
            raise ValueError(f"fixture {fixture_id} required terms are invalid")
        fixtures.append(
            Fixture(
                fixture_id=fixture_id,
                prompt=prompt,
                min_tool_calls=minimum,
                max_tool_calls=maximum,
                max_response_bytes=max_bytes,
                required_any=tuple(term.casefold() for term in required),
            )
        )
    return fixtures


def fetch_catalog() -> list[dict[str, object]]:
    request = urllib.request.Request(
        CATALOG_URL,
        headers={"User-Agent": "oreo-agent-benchmark/0.1"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("Pollinations model catalog has an invalid shape")
    return payload


def pricing_for(
    catalog: list[dict[str, object]], model: str
) -> tuple[Decimal, Decimal]:
    for entry in catalog:
        names = [entry.get("name"), *(entry.get("aliases") or [])]
        if model not in names:
            continue
        capabilities = entry.get("capabilities") or []
        endpoints = entry.get("supported_endpoints") or []
        if "tool_calling" not in capabilities or "/v1/chat/completions" not in endpoints:
            raise ValueError(f"model {model} is incompatible with the Oreo harness")
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict) or pricing.get("currency") != "pollen":
            raise ValueError(f"model {model} has no Pollen token pricing")
        return (
            Decimal(str(pricing["promptTextTokens"])),
            Decimal(str(pricing["completionTextTokens"])),
        )
    raise ValueError(f"model {model} is absent from the live catalog")


def parse_metrics(stderr: str) -> dict[str, object]:
    matches = [
        line.removeprefix(METRICS_PREFIX)
        for line in stderr.splitlines()
        if line.startswith(METRICS_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError("agent run did not emit exactly one metrics record")
    payload = json.loads(matches[0])
    required = {
        "schema_version",
        "model",
        "surface",
        "input_tokens",
        "output_tokens",
        "model_rounds",
        "tool_calls",
        "first_text_ms",
        "total_ms",
        "response_bytes",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("agent metrics record has an invalid shape")
    if payload["schema_version"] != 1 or payload["surface"] != "text":
        raise ValueError("agent metrics record has an unsupported identity")
    return payload


def pollen_cost(
    metrics: dict[str, object], input_rate: Decimal, output_rate: Decimal
) -> Decimal:
    return Decimal(metrics["input_tokens"]) * input_rate + Decimal(
        metrics["output_tokens"]
    ) * output_rate


def quality_passed(fixture: Fixture, response: str, metrics: dict[str, object]) -> bool:
    response_folded = response.casefold()
    terms_pass = not fixture.required_any or any(
        term in response_folded for term in fixture.required_any
    )
    return (
        fixture.min_tool_calls
        <= int(metrics["tool_calls"])
        <= fixture.max_tool_calls
        and int(metrics["response_bytes"]) <= fixture.max_response_bytes
        and int(metrics["model_rounds"]) <= 4
        and terms_pass
    )


def percentile_95(values: list[int]) -> int:
    if not values:
        raise ValueError("cannot calculate a percentile without values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def run_once(
    binary: Path,
    model: str,
    fixture: Fixture,
    timeout_seconds: int,
    input_rate: Decimal,
    output_rate: Decimal,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["OREO_MODEL"] = model
    started = time.monotonic()
    completed = subprocess.run(
        [str(binary), "ask", "--metrics", fixture.prompt],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
    )
    wall_ms = round((time.monotonic() - started) * 1_000)
    if completed.returncode != 0:
        return {
            "fixture_id": fixture.fixture_id,
            "success": False,
            "exit_code": completed.returncode,
            "wall_ms": wall_ms,
        }
    metrics = parse_metrics(completed.stderr)
    cost = pollen_cost(metrics, input_rate, output_rate)
    return {
        "fixture_id": fixture.fixture_id,
        "success": quality_passed(fixture, completed.stdout, metrics),
        "input_tokens": metrics["input_tokens"],
        "output_tokens": metrics["output_tokens"],
        "model_rounds": metrics["model_rounds"],
        "tool_calls": metrics["tool_calls"],
        "first_text_ms": metrics["first_text_ms"],
        "total_ms": metrics["total_ms"],
        "wall_ms": wall_ms,
        "response_bytes": metrics["response_bytes"],
        "pollen": format(cost, "f"),
    }


def aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    successful = [run for run in runs if run["success"]]
    measured = [run for run in runs if "pollen" in run]
    latencies = [int(run["total_ms"]) for run in measured]
    costs = [Decimal(str(run["pollen"])) for run in measured]
    mean_cost = sum(costs, Decimal(0)) / Decimal(len(costs)) if costs else None
    return {
        "passed": len(successful),
        "total": len(runs),
        "pass_rate": len(successful) / len(runs),
        "mean_total_ms": statistics.fmean(latencies) if latencies else None,
        "p95_total_ms": percentile_95(latencies) if latencies else None,
        "mean_pollen": format(mean_cost, "f") if mean_cost is not None else None,
    }


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if "POLLINATIONS_API_KEY" not in os.environ:
        raise ValueError("POLLINATIONS_API_KEY is required")
    if not arguments.binary.is_file():
        raise ValueError("benchmark binary is missing; build elixpo-cli first")
    models = list(dict.fromkeys(arguments.model))
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"model count must be 1-{MAX_MODELS}")
    if not 1 <= arguments.repetitions <= MAX_REPETITIONS:
        raise ValueError(f"repetitions must be 1-{MAX_REPETITIONS}")
    if not 1 <= arguments.timeout_seconds <= 120:
        raise ValueError("timeout must be 1-120 seconds")
    fixtures = load_fixtures(arguments.manifest)
    catalog = fetch_catalog()
    candidates = []
    for model in models:
        input_rate, output_rate = pricing_for(catalog, model)
        runs = []
        for _ in range(arguments.repetitions):
            for fixture in fixtures:
                runs.append(
                    run_once(
                        arguments.binary,
                        model,
                        fixture,
                        arguments.timeout_seconds,
                        input_rate,
                        output_rate,
                    )
                )
        candidates.append(
            {
                "model": model,
                "pricing": {
                    "currency": "pollen",
                    "input_per_token": format(input_rate, "f"),
                    "output_per_token": format(output_rate, "f"),
                },
                "aggregate": aggregate(runs),
                "runs": runs,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "catalog_url": CATALOG_URL,
        "fixture_count": len(fixtures),
        "repetitions": arguments.repetitions,
        "candidates": candidates,
    }


def self_test() -> None:
    metrics = parse_metrics(
        'agent_metrics={"schema_version":1,"model":"fixture","surface":"text",'
        '"input_tokens":100,"output_tokens":20,"model_rounds":1,"tool_calls":0,'
        '"first_text_ms":10,"total_ms":20,"response_bytes":40}'
    )
    assert pollen_cost(metrics, Decimal("0.1"), Decimal("0.2")) == Decimal("14")
    assert percentile_95([1, 2, 3, 4, 5]) == 5
    fixture = Fixture("fixture", "hello", 0, 0, 64, ("hello",))
    assert quality_passed(fixture, "Hello there", metrics)
    print("Agent benchmark self-test passed")


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.command == "self-test":
            self_test()
            return 0
        report = run_benchmark(arguments)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for candidate in report["candidates"]:
            aggregate_report = candidate["aggregate"]
            print(
                f"{candidate['model']}: "
                f"{aggregate_report['passed']}/{aggregate_report['total']} passed, "
                f"p95={aggregate_report['p95_total_ms']} ms, "
                f"mean={aggregate_report['mean_pollen']} pollen"
            )
        print(f"Report: {arguments.output}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
