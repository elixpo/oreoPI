#!/usr/bin/env python3
"""Measure openWakeWord cascade false activations on long ambient audio."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
CHUNK_FRAMES = 1_280
RING_SECONDS = 3
POST_ROLL_FRAMES = SAMPLE_RATE
COOLDOWN_FRAMES = SAMPLE_RATE
MAX_HOURS = 12


def load_benchmark_module(root: Path) -> object:
    path = root / "scripts/benchmark-wake.py"
    spec = importlib.util.spec_from_file_location("oreo_wake_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("wake benchmark module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models/cache/openwakeword-oreo/oreo.onnx",
    )
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(repo_root=root)
    return parser.parse_args()


def validate_wave(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != SAMPLE_RATE
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("soak WAV must be uncompressed 16 kHz mono PCM16")
        frames = audio.getnframes()
    if not SAMPLE_RATE <= frames <= SAMPLE_RATE * 60 * 60 * MAX_HOURS:
        raise ValueError(f"soak WAV must be between one second and {MAX_HOURS} hours")
    return frames, frames / SAMPLE_RATE


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(len(ordered) * fraction)) - 1]


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if not 0 < arguments.threshold <= 1:
        raise ValueError("threshold must be greater than zero and at most one")
    root = arguments.repo_root
    model_path = arguments.model.resolve()
    wav_path = arguments.wav.resolve()
    if not model_path.is_file() or model_path.suffix != ".onnx":
        raise ValueError("openWakeWord ONNX candidate is missing")
    frames, audio_seconds = validate_wave(wav_path)
    feature_dir = root / "models/cache/openwakeword-v0.5.1-features"
    feature_paths = {
        "melspectrogram": feature_dir / "melspectrogram.onnx",
        "embedding": feature_dir / "embedding_model.onnx",
    }
    if any(not path.is_file() for path in feature_paths.values()):
        raise ValueError("cached openWakeWord feature models are missing")
    vosk_path = root / "models/cache/vosk-model-small-en-us-0.15"
    if not vosk_path.is_dir():
        raise ValueError("cached Vosk model is missing")

    try:
        import numpy as np
        import vosk
        from openwakeword.model import Model
    except ImportError as error:
        raise RuntimeError("soak requires openwakeword, numpy, and vosk") from error
    versions = {
        "openwakeword": importlib.metadata.version("openwakeword"),
        "numpy": importlib.metadata.version("numpy"),
        "vosk": importlib.metadata.version("vosk"),
    }
    if versions["openwakeword"] != "0.6.0" or versions["vosk"] != "0.3.45":
        raise RuntimeError("soak dependency versions do not match the reviewed runtime")

    common = load_benchmark_module(root)
    classifier = common.IntentClassifier(
        root / "config/wake-intent-corpus.tsv",
        root / "config/wake-identity-aliases.txt",
    )
    load_started = time.perf_counter()
    detector = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(feature_paths["melspectrogram"]),
        embedding_model_path=str(feature_paths["embedding"]),
    )
    vosk.SetLogLevel(-1)
    vosk_model = vosk.Model(str(vosk_path))
    load_ms = (time.perf_counter() - load_started) * 1000

    ring = bytearray()
    candidate: dict[str, object] | None = None
    post_roll = 0
    cooldown = 0
    processed_frames = 0
    events: list[dict[str, object]] = []
    event_latencies: list[float] = []
    processing_started = time.perf_counter()

    def finish_candidate() -> None:
        nonlocal candidate, post_roll, cooldown
        if candidate is None:
            return
        started = time.perf_counter()
        transcript = common.vosk_transcript(vosk, vosk_model, bytes(ring))
        addressed, intent_score = (
            classifier.classify(transcript) if transcript else (False, None)
        )
        identity_present = classifier.has_identity(transcript) if transcript else False
        event_latencies.append((time.perf_counter() - started) * 1000)
        events.append(
            {
                "audio_second": candidate["audio_second"],
                "keyword": candidate["keyword"],
                "candidate_score": candidate["score"],
                "transcript": transcript,
                "identity_present": identity_present,
                "intent_score": intent_score,
                "accepted": addressed,
            }
        )
        candidate = None
        post_roll = 0
        cooldown = COOLDOWN_FRAMES
        detector.reset()

    with wave.open(str(wav_path), "rb") as audio:
        while True:
            chunk = audio.readframes(CHUNK_FRAMES)
            if not chunk:
                break
            actual_frames = len(chunk) // 2
            processed_frames += actual_frames
            common.append_ring(ring, chunk)
            if candidate is not None:
                post_roll += actual_frames
                if post_roll >= POST_ROLL_FRAMES:
                    finish_candidate()
                continue
            model_chunk = chunk
            if actual_frames < CHUNK_FRAMES:
                model_chunk += bytes((CHUNK_FRAMES - actual_frames) * 2)
            predictions = detector.predict(np.frombuffer(model_chunk, dtype="<i2"))
            if cooldown > 0:
                cooldown = max(0, cooldown - actual_frames)
                continue
            scored_predictions = {
                str(label): float(np.asarray(raw_score).reshape(-1)[-1])
                for label, raw_score in predictions.items()
            }
            label, score = max(scored_predictions.items(), key=lambda item: item[1])
            if score >= arguments.threshold:
                candidate = {
                    "audio_second": processed_frames / SAMPLE_RATE,
                    "keyword": str(label),
                    "score": score,
                }
        finish_candidate()

    elapsed = time.perf_counter() - processing_started
    accepted = sum(bool(event["accepted"]) for event in events)
    hours = audio_seconds / 3600
    return {
        "schema_version": 1,
        "pipeline": "openwakeword-vosk-intent-soak",
        "source": {
            "path": str(wav_path),
            "sha256": common.file_sha256(wav_path),
            "frames": frames,
            "audio_seconds": audio_seconds,
        },
        "model": {
            "path": str(model_path),
            "sha256": common.file_sha256(model_path),
        },
        "settings": {
            "threshold": arguments.threshold,
            "ring_seconds": RING_SECONDS,
            "post_roll_seconds": POST_ROLL_FRAMES / SAMPLE_RATE,
            "cooldown_seconds": COOLDOWN_FRAMES / SAMPLE_RATE,
            "intent_threshold_with_identity": common.IDENTITY_INTENT_THRESHOLD,
            "intent_threshold_without_identity": common.CONTEXT_ONLY_INTENT_THRESHOLD,
        },
        "packages": versions,
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "load_ms": load_ms,
        "processing": {
            "elapsed_seconds": elapsed,
            "realtime_factor": audio_seconds / elapsed,
            "event_latency_ms_p95": percentile(event_latencies, 0.95),
        },
        "aggregate": {
            "candidate_events": len(events),
            "accepted_false_activations": accepted,
            "candidate_events_per_hour": len(events) / hours,
            "accepted_false_activations_per_hour": accepted / hours,
        },
        "events": events,
    }


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run(arguments)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {arguments.output}")
        return 0
    except (KeyError, OSError, RuntimeError, ValueError, wave.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
