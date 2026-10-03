#!/usr/bin/env python3
"""Compare cached English STT candidates on identical bounded WAV fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_FIXTURES = 128
MAX_TRANSCRIPT_BYTES = 4096
MAX_AUDIO_SECONDS = 30
SAMPLE_RATE = 16_000
CHUNK_FRAMES = 320


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    path: Path
    transcript: str
    pcm: bytes
    duration_seconds: float


class Engine(Protocol):
    name: str
    package_version: str

    def transcribe(self, fixture: Fixture) -> str: ...


class SherpaEngine:
    name = "sherpa-onnx"

    def __init__(self, model: Path, threads: int) -> None:
        try:
            import numpy as np
            import sherpa_onnx
        except ImportError as error:
            raise RuntimeError(
                "sherpa benchmark requires sherpa-onnx and numpy"
            ) from error

        self._np = np
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model / "tokens.txt"),
            encoder=str(model / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(model / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(model / "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        self.package_version = importlib.metadata.version("sherpa-onnx")

    def transcribe(self, fixture: Fixture) -> str:
        samples = self._np.frombuffer(fixture.pcm, dtype="<i2").astype(
            self._np.float32
        )
        samples *= 1.0 / 32768.0
        stream = self._recognizer.create_stream()
        for offset in range(0, len(samples), CHUNK_FRAMES):
            stream.accept_waveform(
                SAMPLE_RATE, samples[offset : offset + CHUNK_FRAMES]
            )
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)

        stream.accept_waveform(
            SAMPLE_RATE, self._np.zeros(SAMPLE_RATE // 2, dtype=self._np.float32)
        )
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        if hasattr(self._recognizer, "get_result_all"):
            result = self._recognizer.get_result_all(stream)
        else:
            result = self._recognizer.get_result(stream)
        return result.text.strip()


class VoskEngine:
    name = "vosk"

    def __init__(self, model: Path, _threads: int) -> None:
        try:
            import vosk
        except ImportError as error:
            raise RuntimeError("Vosk benchmark requires vosk") from error

        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self._model = vosk.Model(str(model))
        self.package_version = importlib.metadata.version("vosk")

    def transcribe(self, fixture: Fixture) -> str:
        recognizer = self._vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        parts: list[str] = []
        chunk_bytes = CHUNK_FRAMES * 2
        for offset in range(0, len(fixture.pcm), chunk_bytes):
            chunk = fixture.pcm[offset : offset + chunk_bytes]
            if recognizer.AcceptWaveform(chunk):
                text = json.loads(recognizer.Result()).get("text", "").strip()
                if text:
                    parts.append(text)
        final = json.loads(recognizer.FinalResult()).get("text", "").strip()
        if final:
            parts.append(final)
        return " ".join(parts)


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("self-test", help="run dependency-free scoring tests")

    run = subcommands.add_parser("run", help="benchmark one cached engine")
    run.add_argument("--engine", choices=("sherpa", "vosk"), required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--model", type=Path)
    run.add_argument("--repetitions", type=int, default=5)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--output", type=Path)
    run.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            substitution = previous[hypothesis_index - 1] + (
                reference_word != hypothesis_word
            )
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> dict[str, int | float]:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        raise ValueError("fixture transcript must contain at least one word")
    edits = edit_distance(reference_words, hypothesis_words)
    return {
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "word_errors": edits,
        "wer": edits / len(reference_words),
    }


def load_fixtures(manifest_path: Path) -> list[Fixture]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("fixture manifest schema_version must be 1")
    entries = payload.get("fixtures")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_FIXTURES:
        raise ValueError(f"fixture count must be 1-{MAX_FIXTURES}")

    fixtures: list[Fixture] = []
    seen_ids: set[str] = set()
    for entry in entries:
        fixture_id = entry.get("id", "")
        transcript = entry.get("transcript", "")
        relative_wav = entry.get("wav", "")
        if not isinstance(fixture_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{0,63}", fixture_id
        ):
            raise ValueError("fixture id must be a lowercase slug")
        if fixture_id in seen_ids:
            raise ValueError(f"duplicate fixture id: {fixture_id}")
        seen_ids.add(fixture_id)
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"fixture {fixture_id} has no transcript")
        if len(transcript.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"fixture {fixture_id} transcript is too large")
        if not isinstance(relative_wav, str) or not relative_wav:
            raise ValueError(f"fixture {fixture_id} has no WAV path")

        wav_path = (manifest_path.parent / relative_wav).resolve()
        with wave.open(str(wav_path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != SAMPLE_RATE
                or audio.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"fixture {fixture_id} must be 16 kHz mono PCM16 WAV"
                )
            frame_count = audio.getnframes()
            if not 1 <= frame_count <= SAMPLE_RATE * MAX_AUDIO_SECONDS:
                raise ValueError(f"fixture {fixture_id} duration is out of bounds")
            pcm = audio.readframes(frame_count)
        fixtures.append(
            Fixture(
                fixture_id=fixture_id,
                path=wav_path,
                transcript=transcript,
                pcm=pcm,
                duration_seconds=frame_count / SAMPLE_RATE,
            )
        )
    return fixtures


def percentile(samples: list[float], percentile_value: float) -> float:
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def resident_memory_kib() -> int:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    if match is None:
        raise RuntimeError("VmRSS is unavailable on this Linux host")
    return int(match.group(1))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquired_digest(repo_root: Path, engine: str) -> str:
    name = {
        "sherpa": "sherpa-zipformer-en-20m-int8-2023-02-17.sha256.local",
        "vosk": "vosk-small-en-us-0.15.sha256.local",
    }[engine]
    parts = (repo_root / "models" / "cache" / name).read_text(
        encoding="utf-8"
    ).split()
    if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise ValueError(f"invalid cached digest file: {name}")
    return parts[0]


def create_engine(engine: str, model: Path, threads: int) -> Engine:
    if engine == "sherpa":
        return SherpaEngine(model, threads)
    return VoskEngine(model, threads)


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if not 1 <= arguments.repetitions <= 50:
        raise ValueError("repetitions must be 1-50")
    if not 1 <= arguments.threads <= 8:
        raise ValueError("threads must be 1-8")

    defaults = {
        "sherpa": arguments.repo_root
        / "models/cache/sherpa-zipformer-en-20m-int8-2023-02-17",
        "vosk": arguments.repo_root / "models/cache/vosk-model-small-en-us-0.15",
    }
    model = (arguments.model or defaults[arguments.engine]).resolve()
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    fixtures = load_fixtures(arguments.manifest)

    rss_before = resident_memory_kib()
    load_started = time.perf_counter()
    engine = create_engine(arguments.engine, model, arguments.threads)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    rss_after_load = resident_memory_kib()

    warmup_started = time.perf_counter()
    engine.transcribe(fixtures[0])
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    fixture_reports: list[dict[str, object]] = []
    total_errors = 0
    total_reference_words = 0
    all_latencies: list[float] = []
    for fixture in fixtures:
        latencies: list[float] = []
        hypotheses: list[str] = []
        for _ in range(arguments.repetitions):
            started = time.perf_counter()
            hypothesis = engine.transcribe(fixture)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if len(hypothesis.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
                raise RuntimeError("engine returned an oversized transcript")
            latencies.append(latency_ms)
            hypotheses.append(hypothesis)
        if len(set(hypotheses)) != 1:
            raise RuntimeError(f"fixture {fixture.fixture_id} was nondeterministic")
        transcript_score = score(fixture.transcript, hypotheses[0])
        total_errors += int(transcript_score["word_errors"])
        total_reference_words += int(transcript_score["reference_words"])
        all_latencies.extend(latencies)
        fixture_reports.append(
            {
                "id": fixture.fixture_id,
                "wav_sha256": file_sha256(fixture.path),
                "audio_seconds": fixture.duration_seconds,
                "reference": fixture.transcript,
                "hypothesis": hypotheses[0],
                **transcript_score,
                "latency_ms": {
                    "median": statistics.median(latencies),
                    "p95": percentile(latencies, 0.95),
                },
                "realtime_factor_p95": percentile(latencies, 0.95)
                / (fixture.duration_seconds * 1000.0),
            }
        )

    return {
        "schema_version": 1,
        "engine": engine.name,
        "engine_package_version": engine.package_version,
        "numpy_version": (
            importlib.metadata.version("numpy")
            if arguments.engine == "sherpa"
            else None
        ),
        "model_directory": model.name,
        "model_archive_sha256": acquired_digest(
            arguments.repo_root, arguments.engine
        ),
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "threads": arguments.threads,
        },
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "resident_memory_kib": {
            "before_load": rss_before,
            "after_load": rss_after_load,
            "after_benchmark": resident_memory_kib(),
        },
        "aggregate": {
            "fixtures": len(fixtures),
            "repetitions": arguments.repetitions,
            "word_errors": total_errors,
            "reference_words": total_reference_words,
            "wer": total_errors / total_reference_words,
            "latency_ms_p95": percentile(all_latencies, 0.95),
        },
        "fixture_results": fixture_reports,
    }


def self_test() -> None:
    assert normalize_words("Oreo, DON'T stop!") == ["oreo", "don't", "stop"]
    exact = score("set a timer", "SET A TIMER")
    assert exact["word_errors"] == 0 and exact["wer"] == 0.0
    changed = score("set a timer for tea", "set timer for two")
    assert changed["word_errors"] == 2
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wav_path = root / "fixture.wav"
        with wave.open(str(wav_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(SAMPLE_RATE)
            audio.writeframes(b"\0\0" * CHUNK_FRAMES)
        manifest_path = root / "fixtures.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fixtures": [
                        {
                            "id": "bounded-fixture",
                            "wav": "fixture.wav",
                            "transcript": "test the fixture",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        fixtures = load_fixtures(manifest_path)
        assert len(fixtures) == 1
        assert fixtures[0].duration_seconds == 0.02
    print("STT benchmark self-test passed")


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.command == "self-test":
            self_test()
            return 0
        report = run_benchmark(arguments)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8")
            print(arguments.output)
        else:
            print(rendered, end="")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
