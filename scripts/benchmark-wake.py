#!/usr/bin/env python3
"""Benchmark Oreo's local KWS -> Vosk -> intent wake cascade."""

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
import time
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
CHUNK_FRAMES = 320
OPENWAKEWORD_CHUNK_FRAMES = 1_280
OPENWAKEWORD_POST_ROLL_FRAMES = SAMPLE_RATE
TAIL_FRAMES = 10_560
RING_SECONDS = 3
MAX_FIXTURE_SECONDS = 15
MAX_FIXTURES = 64
MAX_TRANSCRIPT_BYTES = 512
INTENT_THRESHOLD = 1.0


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    path: Path
    reference: str
    expected_wake: bool
    pcm: bytes


class IntentClassifier:
    """Small Bernoulli Naive Bayes model matching the Rust runtime."""

    def __init__(self, corpus: Path, aliases: Path) -> None:
        self.aliases = {
            line.strip()
            for line in aliases.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        if not 1 <= len(self.aliases) <= 16:
            raise ValueError("wake identity alias list is incomplete or too large")
        self.weights: dict[str, list[int]] = {}
        self.document_totals = [0, 0]
        for raw_line in corpus.read_text(encoding="utf-8").splitlines():
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.split("\t", 2)
            if len(fields) != 3:
                raise ValueError("wake intent corpus is malformed")
            split, label, text = fields
            if split != "train":
                continue
            if label not in {"ignore", "wake"}:
                raise ValueError("wake intent corpus has an invalid label")
            category = int(label == "wake")
            self.document_totals[category] += 1
            for feature in set(features(text, self.aliases)):
                counts = self.weights.setdefault(feature, [0, 0])
                counts[category] += 1
        if min(self.document_totals) < 10 or not self.weights:
            raise ValueError("wake intent corpus is incomplete")

    def classify(self, transcript: str) -> tuple[bool, float]:
        if (
            not transcript.strip()
            or len(transcript.encode("utf-8")) > MAX_TRANSCRIPT_BYTES
        ):
            return False, float("-inf")
        documents = sum(self.document_totals)
        scores = [
            math.log((count + 1) / (documents + 2))
            for count in self.document_totals
        ]
        for feature in set(features(transcript, self.aliases)):
            counts = self.weights.get(feature)
            if counts is None:
                continue
            for category in range(2):
                scores[category] += math.log(
                    (counts[category] + 1)
                    / (self.document_totals[category] + 2)
                )
        score = scores[1] - scores[0]
        return score >= INTENT_THRESHOLD, score


def features(text: str, aliases: set[str]) -> list[str]:
    lexical = re.findall(r"[a-z0-9]+", text.casefold())
    words = [
        "assistantname" if word in aliases else word
        for word in lexical
    ]
    output = [f"w:{word}" for word in words]
    if words:
        output.extend((f"first:{words[0]}", f"last:{words[-1]}"))
    output.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    return output


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="test the dependency-free intent gate")
    run = commands.add_parser("run", help="run the complete local wake cascade")
    run.add_argument(
        "--manifest", type=Path, default=root / "tests/audio/wake-fixtures.local.json"
    )
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument(
        "--kws-model", choices=("english", "bilingual"), default="english"
    )
    run.add_argument(
        "--kws-engine", choices=("sherpa", "openwakeword"), default="sherpa"
    )
    run.add_argument("--keywords-score", type=float, default=1.5)
    run.add_argument("--keywords-threshold", type=float, default=0.20)
    run.add_argument(
        "--openwakeword-model",
        type=Path,
        default=root / "models/cache/openwakeword-oreo/oreo.onnx",
    )
    run.add_argument("--openwakeword-threshold", type=float, default=0.5)
    run.add_argument("--output", type=Path)
    run.set_defaults(repo_root=root)
    return parser.parse_args()


def load_fixtures(path: Path) -> list[Fixture]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("fixtures")
    if payload.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("wake fixture manifest schema_version must be 1")
    if not 1 <= len(entries) <= MAX_FIXTURES:
        raise ValueError(f"wake fixture count must be 1-{MAX_FIXTURES}")
    output: list[Fixture] = []
    seen: set[str] = set()
    for entry in entries:
        fixture_id = entry.get("id")
        reference = entry.get("transcript")
        expected_wake = entry.get("expected_wake")
        relative_wav = entry.get("wav")
        if not isinstance(fixture_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{0,63}", fixture_id
        ):
            raise ValueError("wake fixture id must be a lowercase slug")
        if fixture_id in seen:
            raise ValueError(f"duplicate wake fixture id: {fixture_id}")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"wake fixture {fixture_id} has no transcript")
        if not isinstance(expected_wake, bool):
            raise ValueError(f"wake fixture {fixture_id} has no expected_wake boolean")
        if not isinstance(relative_wav, str) or not relative_wav:
            raise ValueError(f"wake fixture {fixture_id} has no WAV path")
        wav_path = (path.parent / relative_wav).resolve()
        with wave.open(str(wav_path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != SAMPLE_RATE
                or audio.getcomptype() != "NONE"
            ):
                raise ValueError(f"wake fixture {fixture_id} must be 16 kHz mono PCM16")
            frames = audio.getnframes()
            if not 1 <= frames <= SAMPLE_RATE * MAX_FIXTURE_SECONDS:
                raise ValueError(
                    f"wake fixture {fixture_id} must be at most {MAX_FIXTURE_SECONDS} seconds"
                )
            pcm = audio.readframes(frames)
        seen.add(fixture_id)
        output.append(Fixture(fixture_id, wav_path, reference, expected_wake, pcm))
    return output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(len(ordered) * fraction)) - 1]


def resident_memory_kib() -> int:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    if match is None:
        raise RuntimeError("VmRSS is unavailable on this Linux host")
    return int(match.group(1))


def cached_digest(root: Path, model_id: str) -> str:
    path = root / "models/cache" / f"{model_id}.sha256.local"
    parts = path.read_text(encoding="utf-8").split()
    if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise ValueError(f"invalid cached digest: {path.name}")
    return parts[0]


def vosk_transcript(vosk: object, model: object, pcm: bytes) -> str:
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    parts: list[str] = []
    for offset in range(0, len(pcm), CHUNK_FRAMES * 2):
        chunk = pcm[offset : offset + CHUNK_FRAMES * 2]
        if recognizer.AcceptWaveform(chunk):
            text = json.loads(recognizer.Result()).get("text", "").strip()
            if text:
                parts.append(text)
    final = json.loads(recognizer.FinalResult()).get("text", "").strip()
    if final:
        parts.append(final)
    return " ".join(parts)


def append_ring(ring: bytearray, chunk: bytes) -> None:
    ring.extend(chunk)
    excess = len(ring) - RING_SECONDS * SAMPLE_RATE * 2
    if excess > 0:
        del ring[:excess]


def detect_keyword(kws: object, np: object, pcm: bytes) -> tuple[str, bytes]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    stream = kws.create_stream()
    ring = bytearray()
    for offset in range(0, len(samples), CHUNK_FRAMES):
        sample_chunk = samples[offset : offset + CHUNK_FRAMES]
        append_ring(ring, pcm[offset * 2 : (offset + len(sample_chunk)) * 2])
        stream.accept_waveform(SAMPLE_RATE, sample_chunk)
        while kws.is_ready(stream):
            kws.decode_stream(stream)
            result = kws.get_result(stream)
            if result:
                return result, bytes(ring)
    stream.accept_waveform(SAMPLE_RATE, np.zeros(TAIL_FRAMES, dtype=np.float32))
    stream.input_finished()
    while kws.is_ready(stream):
        kws.decode_stream(stream)
        result = kws.get_result(stream)
        if result:
            return result, bytes(ring)
    return "", b""


def detect_openwakeword(
    model: object, np: object, pcm: bytes, threshold: float
) -> tuple[str, bytes, float]:
    model.reset()
    ring = bytearray()
    best_score = 0.0
    detected_label = ""
    detected_score = 0.0
    post_roll_frames = 0
    for offset in range(0, len(pcm), OPENWAKEWORD_CHUNK_FRAMES * 2):
        audio_chunk = pcm[offset : offset + OPENWAKEWORD_CHUNK_FRAMES * 2]
        append_ring(ring, audio_chunk)
        if detected_label:
            post_roll_frames += len(audio_chunk) // 2
            if post_roll_frames >= OPENWAKEWORD_POST_ROLL_FRAMES:
                return detected_label, bytes(ring), detected_score
            continue
        model_chunk = audio_chunk
        if len(model_chunk) < OPENWAKEWORD_CHUNK_FRAMES * 2:
            model_chunk += bytes(OPENWAKEWORD_CHUNK_FRAMES * 2 - len(model_chunk))
        samples = np.frombuffer(model_chunk, dtype="<i2")
        predictions = model.predict(samples)
        if not isinstance(predictions, dict) or not predictions:
            raise RuntimeError("openWakeWord returned no model scores")
        label, raw_score = max(
            predictions.items(),
            key=lambda item: float(np.asarray(item[1]).reshape(-1)[-1]),
        )
        score = float(np.asarray(raw_score).reshape(-1)[-1])
        if score > best_score:
            best_score = score
        if score >= threshold:
            detected_label, detected_score = str(label), score
    if detected_label:
        return detected_label, bytes(ring), detected_score
    return "", b"", best_score


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if not 1 <= arguments.repetitions <= 20 or not 1 <= arguments.threads <= 4:
        raise ValueError("repetitions must be 1-20 and threads must be 1-4")
    if arguments.kws_engine == "sherpa" and (
        not 0 < arguments.keywords_threshold <= 1
        or not 0 < arguments.keywords_score <= 10
    ):
        raise ValueError("keyword score/threshold are outside safe benchmark bounds")
    if not 0 < arguments.openwakeword_threshold <= 1:
        raise ValueError("openWakeWord threshold must be greater than zero and at most one")
    try:
        import numpy as np
        import vosk
    except ImportError as error:
        raise RuntimeError("wake benchmark requires numpy and vosk") from error

    package_versions = {
        "vosk": importlib.metadata.version("vosk"),
        "numpy": importlib.metadata.version("numpy"),
    }
    if package_versions["vosk"] != "0.3.45":
        raise RuntimeError("wake benchmark requires vosk==0.3.45")

    root = arguments.repo_root
    kws_specs = {
        "english": {
            "id": "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
            "encoder": "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            "decoder": "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            "joiner": "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        },
        "bilingual": {
            "id": "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
            "encoder": "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
            "decoder": "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
            "joiner": "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        },
    }
    vosk_id = "vosk-model-small-en-us-0.15"
    vosk_dir = root / "models/cache" / vosk_id
    if not vosk_dir.is_dir():
        raise ValueError("Vosk model is not completely cached")
    fixtures = load_fixtures(arguments.manifest)
    classifier = IntentClassifier(
        root / "config/wake-intent-corpus.tsv",
        root / "config/wake-identity-aliases.txt",
    )
    rss_before = resident_memory_kib()
    load_started = time.perf_counter()
    keyword_lines: list[str] = []
    files: dict[str, Path] = {}
    kws_id: str
    if arguments.kws_engine == "sherpa":
        try:
            import sherpa_onnx
        except ImportError as error:
            raise RuntimeError("sherpa benchmark requires sherpa-onnx") from error
        package_versions["sherpa_onnx"] = importlib.metadata.version("sherpa-onnx")
        if package_versions["sherpa_onnx"] != "1.13.8":
            raise RuntimeError("sherpa benchmark requires sherpa-onnx==1.13.8")
        kws_spec = kws_specs[arguments.kws_model]
        kws_id = str(kws_spec["id"])
        kws_dir = root / "models/cache" / kws_id
        files = {
            "tokens": kws_dir / "tokens.txt",
            "encoder": kws_dir / str(kws_spec["encoder"]),
            "decoder": kws_dir / str(kws_spec["decoder"]),
            "joiner": kws_dir / str(kws_spec["joiner"]),
            "keywords_file": kws_dir / "oreo-keywords.txt",
        }
        if any(not path.is_file() for path in files.values()):
            raise ValueError("sherpa wake model is not completely cached")
        keyword_lines = [
            line.strip()
            for line in files["keywords_file"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(keyword_lines) != 1 or not keyword_lines[0].endswith(" @OREO"):
            raise ValueError(
                "wake keywords are stale; run scripts/fetch-wake-model.sh to refresh them"
            )
        kws = sherpa_onnx.KeywordSpotter(
            **{name: str(path) for name, path in files.items()},
            num_threads=arguments.threads,
            keywords_score=arguments.keywords_score,
            keywords_threshold=arguments.keywords_threshold,
            provider="cpu",
        )
    else:
        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise RuntimeError(
                "openWakeWord benchmark requires openwakeword==0.6.0"
            ) from error
        package_versions["openwakeword"] = importlib.metadata.version("openwakeword")
        if package_versions["openwakeword"] != "0.6.0":
            raise RuntimeError("openWakeWord benchmark requires openwakeword==0.6.0")
        model_path = arguments.openwakeword_model.resolve()
        if not model_path.is_file() or model_path.suffix != ".onnx":
            raise ValueError("custom openWakeWord ONNX model is missing")
        kws_id = model_path.stem
        kws = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    vosk.SetLogLevel(-1)
    vosk_model = vosk.Model(str(vosk_dir))
    load_ms = (time.perf_counter() - load_started) * 1000
    rss_after_load = resident_memory_kib()

    results: list[dict[str, object]] = []
    latencies: list[float] = []
    false_accepts = 0
    false_rejects = 0
    candidate_runs = 0
    for fixture in fixtures:
        runs: list[dict[str, object]] = []
        for _ in range(arguments.repetitions):
            started = time.perf_counter()
            if arguments.kws_engine == "sherpa":
                keyword, candidate_pcm = detect_keyword(kws, np, fixture.pcm)
                candidate_score = None
            else:
                keyword, candidate_pcm, candidate_score = detect_openwakeword(
                    kws, np, fixture.pcm, arguments.openwakeword_threshold
                )
            transcript = (
                vosk_transcript(vosk, vosk_model, candidate_pcm) if keyword else ""
            )
            addressed, intent_score = (
                classifier.classify(transcript)
                if keyword and transcript
                else (False, None)
            )
            accepted = bool(keyword) and addressed
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            candidate_runs += int(bool(keyword))
            false_accepts += int(accepted and not fixture.expected_wake)
            false_rejects += int(not accepted and fixture.expected_wake)
            runs.append(
                {
                    "keyword": keyword,
                    "candidate_score": candidate_score,
                    "transcript": transcript,
                    "intent_score": intent_score,
                    "window_ms": len(candidate_pcm) * 1000 // (SAMPLE_RATE * 2),
                    "accepted": accepted,
                    "latency_ms": latency_ms,
                }
            )
        representative = Counter(
            (run["keyword"], run["transcript"], run["accepted"]) for run in runs
        ).most_common(1)[0][0]
        scores = [
            float(run["intent_score"])
            for run in runs
            if run["intent_score"] is not None
        ]
        candidate_scores = [
            float(run["candidate_score"])
            for run in runs
            if run["candidate_score"] is not None
        ]
        results.append(
            {
                "id": fixture.fixture_id,
                "wav_sha256": file_sha256(fixture.path),
                "audio_seconds": len(fixture.pcm) / (SAMPLE_RATE * 2),
                "reference": fixture.reference,
                "expected_wake": fixture.expected_wake,
                "keyword": representative[0],
                "candidate_score_median": (
                    statistics.median(candidate_scores) if candidate_scores else None
                ),
                "transcript": representative[1],
                "accepted": representative[2],
                "stable": len(
                    {
                        (run["keyword"], run["transcript"], run["accepted"])
                        for run in runs
                    }
                )
                == 1,
                "intent_score_median": statistics.median(scores) if scores else None,
                "candidate_window_ms_median": statistics.median(
                    int(run["window_ms"]) for run in runs
                ),
                "latency_ms_p95": percentile(
                    [float(run["latency_ms"]) for run in runs], 0.95
                ),
            }
        )
    measured_runs = len(fixtures) * arguments.repetitions
    return {
        "schema_version": 1,
        "pipeline": f"{arguments.kws_engine}-kws-vosk-intent",
        "packages": package_versions,
        "models": {
            "kws": {
                "id": kws_id,
                **(
                    {
                        "archive_sha256": cached_digest(root, kws_id),
                        "keywords_sha256": file_sha256(files["keywords_file"]),
                        "keyword_entries": len(keyword_lines),
                    }
                    if arguments.kws_engine == "sherpa"
                    else {
                        "model_sha256": file_sha256(arguments.openwakeword_model),
                    }
                ),
            },
            "stt": {
                "id": vosk_id,
                "archive_sha256": cached_digest(root, "vosk-small-en-us-0.15"),
            },
        },
        "settings": {
            "window_seconds": RING_SECONDS,
            "kws_engine": arguments.kws_engine,
            "kws_model": (
                arguments.kws_model
                if arguments.kws_engine == "sherpa"
                else str(arguments.openwakeword_model)
            ),
            "threads": arguments.threads if arguments.kws_engine == "sherpa" else None,
            "keywords_score": (
                arguments.keywords_score if arguments.kws_engine == "sherpa" else None
            ),
            "keywords_threshold": (
                arguments.keywords_threshold
                if arguments.kws_engine == "sherpa"
                else arguments.openwakeword_threshold
            ),
            "intent_threshold": INTENT_THRESHOLD,
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "load_ms": load_ms,
        "resident_memory_kib": {
            "before_load": rss_before,
            "after_load": rss_after_load,
            "after_benchmark": resident_memory_kib(),
        },
        "aggregate": {
            "fixtures": len(fixtures),
            "repetitions": arguments.repetitions,
            "candidate_runs": candidate_runs,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "accuracy": (measured_runs - false_accepts - false_rejects) / measured_runs,
            "latency_ms_p95": percentile(latencies, 0.95),
        },
        "fixture_results": results,
    }


def self_test() -> None:
    root = Path(__file__).resolve().parent.parent
    classifier = IntentClassifier(
        root / "config/wake-intent-corpus.tsv",
        root / "config/wake-identity-aliases.txt",
    )
    corpus = (root / "config/wake-intent-corpus.tsv").read_text(encoding="utf-8")
    tested = 0
    for line in corpus.splitlines():
        fields = line.split("\t", 2)
        if fields[0] != "test":
            continue
        expected = fields[1] == "wake"
        actual, _ = classifier.classify(fields[2])
        assert actual == expected, fields[2]
        tested += 1
    assert tested >= 10
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    ring = bytearray()
    append_ring(ring, b"a" * (RING_SECONDS * SAMPLE_RATE * 2))
    append_ring(ring, b"b" * 640)
    assert len(ring) == RING_SECONDS * SAMPLE_RATE * 2
    assert ring[:1] == b"a" and ring[-1:] == b"b"

    class FakeArray:
        def __init__(self, value: float) -> None:
            self.value = value

        def reshape(self, *_shape: int) -> FakeArray:
            return self

        def __getitem__(self, _index: int) -> float:
            return self.value

    class FakeNumpy:
        @staticmethod
        def frombuffer(_chunk: bytes, dtype: str) -> object:
            assert dtype == "<i2"
            return object()

        @staticmethod
        def asarray(value: float) -> FakeArray:
            return FakeArray(value)

    class FakeOpenWakeWord:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls = 0

        def predict(self, _samples: object) -> dict[str, float]:
            self.calls += 1
            return {"oreo": 0.7 if self.calls == 2 else 0.1}

    fake_pcm = bytes(20 * OPENWAKEWORD_CHUNK_FRAMES * 2)
    keyword, candidate, score = detect_openwakeword(
        FakeOpenWakeWord(), FakeNumpy(), fake_pcm, 0.5
    )
    assert keyword == "oreo" and score == 0.7
    assert len(candidate) == 15 * OPENWAKEWORD_CHUNK_FRAMES * 2
    print("wake benchmark self-test passed")


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
            print(f"wrote {arguments.output}")
        else:
            print(rendered, end="")
        return 0
    except (AssertionError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
