#!/usr/bin/env python3
"""Bounded local PocketTTS worker using a private framed PCM protocol."""

from __future__ import annotations

import argparse
import io
import logging
import select
import struct
import sys
from typing import BinaryIO

FRAME_READY = 1
FRAME_AUDIO = 2
FRAME_DONE = 3
FRAME_ERROR = 4
MAX_TEXT_BYTES = 4_096
SAMPLE_RATE = 24_000
MAX_CHUNK_SAMPLES = SAMPLE_RATE // 10


def read_exact(source: BinaryIO, size: int) -> bytes | None:
    output = bytearray()
    while len(output) < size:
        chunk = source.read(size - len(output))
        if not chunk:
            return None
        output.extend(chunk)
    return bytes(output)


def read_request(source: BinaryIO) -> str | None:
    header = read_exact(source, 4)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if not 1 <= size <= MAX_TEXT_BYTES:
        raise ValueError("request size is invalid")
    payload = read_exact(source, size)
    if payload is None:
        raise ValueError("request is truncated")
    text = payload.decode("utf-8")
    if not text.strip():
        raise ValueError("request text is empty")
    return text


def write_frame(destination: BinaryIO, kind: int, payload: bytes = b"") -> None:
    destination.write(bytes((kind,)))
    destination.write(struct.pack(">I", len(payload)))
    destination.write(payload)
    destination.flush()


def run_self_test() -> None:
    request = "Hello, Oreo.".encode()
    assert read_request(io.BytesIO(struct.pack(">I", len(request)) + request)) == (
        "Hello, Oreo."
    )
    framed = io.BytesIO()
    write_frame(framed, FRAME_AUDIO, b"\x01\x00\x02\x00")
    assert framed.getvalue() == b"\x02\x00\x00\x00\x04\x01\x00\x02\x00"
    print("PocketTTS worker self-test passed")


def serve(idle_seconds: int) -> None:
    protocol_output = sys.stdout.buffer
    # Third-party diagnostics must never corrupt the stdout framing protocol.
    sys.stdout = sys.stderr
    logging.disable(logging.CRITICAL)

    import numpy as np
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language="english", quantize=True)
    voice_state = model.get_state_for_audio_prompt("alba")
    if model.sample_rate != SAMPLE_RATE:
        write_frame(protocol_output, FRAME_ERROR, b"unsupported-format")
        return
    write_frame(protocol_output, FRAME_READY, struct.pack(">I", SAMPLE_RATE))

    source = sys.stdin.buffer
    while True:
        readable, _, _ = select.select([source], [], [], idle_seconds)
        if not readable:
            return
        try:
            text = read_request(source)
            if text is None:
                return
            for chunk in model.generate_audio_stream(voice_state, text):
                samples = chunk.detach().cpu().numpy()
                if samples.ndim != 1 or samples.size == 0:
                    raise RuntimeError("invalid audio")
                pcm = (np.clip(samples, -1.0, 1.0) * 32_767).astype("<i2")
                for offset in range(0, pcm.size, MAX_CHUNK_SAMPLES):
                    write_frame(
                        protocol_output,
                        FRAME_AUDIO,
                        pcm[offset : offset + MAX_CHUNK_SAMPLES].tobytes(),
                    )
            write_frame(protocol_output, FRAME_DONE)
        except (OSError, RuntimeError, UnicodeError, ValueError):
            write_frame(protocol_output, FRAME_ERROR, b"synthesis-failed")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        run_self_test()
        return 0
    if not 5 <= arguments.idle_seconds <= 3_600:
        parser.error("--idle-seconds must be 5-3600")
    try:
        serve(arguments.idle_seconds)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
