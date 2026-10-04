#!/usr/bin/env python3
"""Benchmark the pinned local PocketTTS candidate on bounded English fixtures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

PACKAGE_VERSION = "3.3.0"
MODEL_REVISION = "983151f13aaeab1b13c1e5e3c2c383d49a9edf3f"
VOICE_REVISION = "4e1e0a3e611c51c0b4ed8174fc10f32a54644303"
SAMPLE_RATE = 24_000
MAX_FIXTURES = 32
MAX_TEXT_BYTES = 2_048


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    text: str


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("self-test", help="run dependency-free harness tests")

    run = subcommands.add_parser("run", help="run the PocketTTS benchmark")
    run.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "tests/audio/tts-fixtures.json",
    )
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--voice", default="alba")
    run.add_argument("--output", type=Path)
    run.add_argument("--save-audio", type=Path)
    run.add_argument(
        "--offline",
        action="store_true",
        help="forbid network access after the pinned artefacts have been cached",
    )
    run.add_argument(
        "--unquantized",
        action="store_true",
        help="disable the default dynamic-int8 CPU quantization",
    )
    run.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def percentile(samples: list[float], percentile_value: float) -> float:
    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0 < percentile_value <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def load_fixtures(path: Path) -> list[Fixture]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("TTS fixture manifest schema_version must be 1")
    entries = payload.get("fixtures")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_FIXTURES:
        raise ValueError(f"TTS fixture count must be 1-{MAX_FIXTURES}")

    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each TTS fixture must be an object")
        fixture_id = entry.get("id")
        text = entry.get("text")
        if not isinstance(fixture_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{0,63}", fixture_id
        ):
            raise ValueError("TTS fixture id must be a lowercase slug")
        if fixture_id in seen:
            raise ValueError(f"duplicate TTS fixture id: {fixture_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"TTS fixture {fixture_id} has no text")
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError(f"TTS fixture {fixture_id} text is too large")
        seen.add(fixture_id)
        fixtures.append(Fixture(fixture_id=fixture_id, text=text.strip()))
    return fixtures


def resident_memory_kib() -> int:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    if match is None:
        raise RuntimeError("VmRSS is unavailable on this Linux host")
    return int(match.group(1))


def write_wav(path: Path, chunks: list[object], sample_rate: int) -> None:
    import numpy as np

    samples = np.concatenate([chunk.detach().cpu().numpy() for chunk in chunks])
    pcm = (np.clip(samples, -1.0, 1.0) * 32_767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def run_self_test() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    fixtures = load_fixtures(
        Path(__file__).resolve().parent.parent / "tests/audio/tts-fixtures.json"
    )
    assert len(fixtures) >= 3
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)
    print("TTS benchmark self-test passed")


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if not 1 <= arguments.repetitions <= 100:
        raise ValueError("repetitions must be 1-100")
    if not 1 <= arguments.threads <= 16:
        raise ValueError("threads must be 1-16")
    if arguments.voice != "alba":
        raise ValueError("only the reviewed alba voice is allowed in this pass")

    cache = arguments.repo_root / "models/cache/huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    if arguments.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        import torch
        from pocket_tts import TTSModel
    except ImportError as error:
        raise RuntimeError(
            f"benchmark requires pocket-tts=={PACKAGE_VERSION} and its CPU dependencies"
        ) from error

    installed = importlib.metadata.version("pocket-tts")
    if installed != PACKAGE_VERSION:
        raise RuntimeError(
            f"expected pocket-tts=={PACKAGE_VERSION}, found pocket-tts=={installed}"
        )
    torch.set_num_threads(arguments.threads)
    fixtures = load_fixtures(arguments.manifest.resolve())

    rss_before_load = resident_memory_kib()
    load_started = time.monotonic()
    model = TTSModel.load_model(language="english", quantize=not arguments.unquantized)
    model_load_ms = (time.monotonic() - load_started) * 1_000
    if model.sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"expected {SAMPLE_RATE} Hz output, model reports {model.sample_rate} Hz"
        )

    voice_started = time.monotonic()
    voice_state = model.get_state_for_audio_prompt(arguments.voice)
    voice_load_ms = (time.monotonic() - voice_started) * 1_000
    rss_after_load = resident_memory_kib()

    runs: list[dict[str, object]] = []
    for repetition in range(arguments.repetitions):
        for fixture in fixtures:
            started = time.monotonic()
            first_audio_ms: float | None = None
            chunks: list[object] = []
            sample_count = 0
            for chunk in model.generate_audio_stream(voice_state, fixture.text):
                if first_audio_ms is None:
                    first_audio_ms = (time.monotonic() - started) * 1_000
                if chunk.ndim != 1 or chunk.numel() == 0:
                    raise RuntimeError("PocketTTS emitted an invalid audio chunk")
                if not bool(torch.isfinite(chunk).all()):
                    raise RuntimeError("PocketTTS emitted non-finite samples")
                chunks.append(chunk)
                sample_count += chunk.numel()
            generation_ms = (time.monotonic() - started) * 1_000
            if first_audio_ms is None or sample_count == 0:
                raise RuntimeError("PocketTTS emitted no audio")
            audio_seconds = sample_count / model.sample_rate
            result = {
                "fixture_id": fixture.fixture_id,
                "repetition": repetition + 1,
                "first_audio_ms": round(first_audio_ms, 3),
                "generation_ms": round(generation_ms, 3),
                "audio_seconds": round(audio_seconds, 4),
                "realtime_factor": round(audio_seconds / (generation_ms / 1_000), 4),
                "chunks": len(chunks),
                "samples": sample_count,
                "rss_kib": resident_memory_kib(),
            }
            runs.append(result)
            print(
                f"{fixture.fixture_id} run {repetition + 1}: "
                f"first={first_audio_ms:.0f} ms rtf={result['realtime_factor']}x"
            )
            if arguments.save_audio is not None and repetition == 0:
                write_wav(
                    arguments.save_audio / f"{fixture.fixture_id}.wav",
                    chunks,
                    model.sample_rate,
                )

    cancellation_fixture = fixtures[-1]
    stop = threading.Event()
    cancellation_started = time.monotonic()
    stream = model.generate_audio_stream(
        voice_state, cancellation_fixture.text, stop=stop
    )
    first_chunk = next(stream)
    stop.set()
    remaining_chunks = sum(1 for _ in stream)
    cancellation_ms = (time.monotonic() - cancellation_started) * 1_000

    first_audio_samples = [float(run["first_audio_ms"]) for run in runs]
    generation_samples = [float(run["generation_ms"]) for run in runs]
    realtime_factors = [float(run["realtime_factor"]) for run in runs]
    report: dict[str, object] = {
        "schema_version": 1,
        "engine": "pocket-tts",
        "package_version": installed,
        "language": "english",
        "voice": arguments.voice,
        "quantized": not arguments.unquantized,
        "model_revision": MODEL_REVISION,
        "voice_embedding_revision": VOICE_REVISION,
        "sample_rate": model.sample_rate,
        "threads": arguments.threads,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cache_directory": str(cache.relative_to(arguments.repo_root)),
        "model_load_ms": round(model_load_ms, 3),
        "voice_load_ms": round(voice_load_ms, 3),
        "rss_before_load_kib": rss_before_load,
        "rss_after_load_kib": rss_after_load,
        "rss_model_growth_kib": max(0, rss_after_load - rss_before_load),
        "summary": {
            "runs": len(runs),
            "first_audio_p50_ms": round(statistics.median(first_audio_samples), 3),
            "first_audio_p95_ms": round(percentile(first_audio_samples, 0.95), 3),
            "generation_p95_ms": round(percentile(generation_samples, 0.95), 3),
            "realtime_factor_median": round(statistics.median(realtime_factors), 4),
            "peak_rss_kib": max(int(run["rss_kib"]) for run in runs),
        },
        "cancellation": {
            "first_chunk_samples": first_chunk.numel(),
            "remaining_chunks": remaining_chunks,
            "elapsed_ms": round(cancellation_ms, 3),
        },
        "runs": runs,
    }
    return report


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.command == "self-test":
            run_self_test()
            return 0
        report = run_benchmark(arguments)
        rendered = json.dumps(report, indent=2) + "\n"
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8")
            print(f"wrote {arguments.output}")
        else:
            print(rendered, end="")
        return 0
    except (OSError, ValueError, RuntimeError, StopIteration) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
