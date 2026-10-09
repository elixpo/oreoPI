#!/usr/bin/env python3
"""Pinned openWakeWord inference worker using a bounded binary protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import struct
import sys
from pathlib import Path
from typing import BinaryIO

REQUEST_AUDIO = 1
REQUEST_RESET = 2
FRAME_READY = 1
FRAME_SCORE = 2
FRAME_RESET = 3
FRAME_ERROR = 255
SAMPLE_RATE = 16_000
CHUNK_FRAMES = 1_280
CHUNK_BYTES = CHUNK_FRAMES * 2
MAX_REQUEST_BYTES = CHUNK_BYTES


def read_exact(source: BinaryIO, size: int) -> bytes | None:
    output = bytearray()
    while len(output) < size:
        chunk = source.read(size - len(output))
        if not chunk:
            return None
        output.extend(chunk)
    return bytes(output)


def read_request(source: BinaryIO) -> tuple[int, bytes] | None:
    header = read_exact(source, 5)
    if header is None:
        return None
    size = struct.unpack(">I", header[1:])[0]
    if size > MAX_REQUEST_BYTES:
        raise ValueError("request is too large")
    payload = read_exact(source, size)
    if payload is None:
        raise ValueError("request is truncated")
    if header[0] == REQUEST_AUDIO and size != CHUNK_BYTES:
        raise ValueError("audio request has an invalid size")
    if header[0] == REQUEST_RESET and size != 0:
        raise ValueError("reset request has an invalid size")
    if header[0] not in (REQUEST_AUDIO, REQUEST_RESET):
        raise ValueError("request kind is invalid")
    return header[0], payload


def write_frame(destination: BinaryIO, kind: int, payload: bytes = b"") -> None:
    destination.write(bytes((kind,)))
    destination.write(struct.pack(">I", len(payload)))
    destination.write(payload)
    destination.flush()


def validate_file(path: Path, suffix: str) -> None:
    if not path.is_file() or path.suffix != suffix:
        raise ValueError("model asset is invalid")


def serve(model_path: Path, melspec_path: Path, embedding_path: Path) -> None:
    validate_file(model_path, ".onnx")
    validate_file(melspec_path, ".onnx")
    validate_file(embedding_path, ".onnx")
    if importlib.metadata.version("openwakeword") != "0.6.0":
        raise RuntimeError("openWakeWord version is unsupported")

    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    logging.disable(logging.CRITICAL)

    import numpy as np
    from openwakeword.model import Model

    detector = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(melspec_path),
        embedding_model_path=str(embedding_path),
    )
    write_frame(
        protocol_output,
        FRAME_READY,
        struct.pack(">II", SAMPLE_RATE, CHUNK_FRAMES),
    )

    source = sys.stdin.buffer
    while True:
        request = read_request(source)
        if request is None:
            return
        kind, payload = request
        if kind == REQUEST_RESET:
            detector.reset()
            write_frame(protocol_output, FRAME_RESET)
            continue
        predictions = detector.predict(np.frombuffer(payload, dtype="<i2"))
        score = max(
            float(np.asarray(value).reshape(-1)[-1])
            for value in predictions.values()
        )
        write_frame(protocol_output, FRAME_SCORE, struct.pack(">f", score))


def run_self_test() -> None:
    request = bytes((REQUEST_AUDIO,)) + struct.pack(">I", CHUNK_BYTES) + bytes(
        CHUNK_BYTES
    )
    kind, payload = read_request(__import__("io").BytesIO(request)) or (0, b"")
    assert kind == REQUEST_AUDIO and len(payload) == CHUNK_BYTES
    framed = __import__("io").BytesIO()
    write_frame(framed, FRAME_SCORE, struct.pack(">f", 0.25))
    assert len(framed.getvalue()) == 9
    print("openWakeWord worker self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--melspec", type=Path)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        run_self_test()
        return 0
    if None in (arguments.model, arguments.melspec, arguments.embedding):
        parser.error("--model, --melspec, and --embedding are required")
    try:
        serve(arguments.model, arguments.melspec, arguments.embedding)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError):
        try:
            write_frame(sys.stdout.buffer, FRAME_ERROR)
        except OSError:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
