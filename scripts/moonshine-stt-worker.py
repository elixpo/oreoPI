#!/usr/bin/env python3
"""Persistent, bounded Moonshine streaming worker for Oreo's Rust runtime."""

from __future__ import annotations

import argparse
import array
import re
import struct
import sys
from pathlib import Path

COMMAND_BEGIN = 1
COMMAND_AUDIO = 2
COMMAND_FINISH = 3
COMMAND_ABORT = 4

FRAME_READY = 1
FRAME_ACK = 2
FRAME_PARTIAL = 3
FRAME_FINAL = 4
FRAME_ERROR = 5

MAX_AUDIO_BYTES = 32_000
MAX_TRANSCRIPT_BYTES = 4_096
SAMPLE_RATE = 16_000


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--attention-name", required=True)
    parser.add_argument("--update-interval", type=float, default=0.25)
    return parser.parse_args()


def write_frame(kind: int, payload: bytes = b"") -> None:
    if len(payload) > MAX_TRANSCRIPT_BYTES and kind in (FRAME_PARTIAL, FRAME_FINAL):
        raise ValueError("transcript exceeds worker bound")
    output = sys.stdout.buffer
    output.write(bytes((kind,)) + struct.pack(">I", len(payload)) + payload)
    output.flush()


def read_frame() -> tuple[int, bytes] | None:
    header = sys.stdin.buffer.read(5)
    if not header:
        return None
    if len(header) != 5:
        raise EOFError("truncated command header")
    size = struct.unpack(">I", header[1:])[0]
    if size > MAX_AUDIO_BYTES:
        raise ValueError("command payload exceeds worker bound")
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise EOFError("truncated command payload")
    return header[0], payload


def transcript_text(transcript: object) -> str:
    return " ".join(
        line.text.strip() for line in transcript.lines if line.text.strip()
    ).strip()


def pcm_floats(payload: bytes) -> list[float]:
    if not payload or len(payload) % 2:
        raise ValueError("audio payload must contain PCM16 samples")
    samples = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    return [sample / 32768.0 for sample in samples]


def run() -> int:
    config = arguments()
    attention_name = config.attention_name.strip().lower()
    if not re.fullmatch(r"[a-z]{3,24}", attention_name):
        raise ValueError("attention name must be one ASCII word")
    if not config.model.is_dir():
        raise ValueError("model directory does not exist")
    if not 0.1 <= config.update_interval <= 2.0:
        raise ValueError("update interval is out of bounds")

    from moonshine_voice import ModelArch, Transcriber

    transcriber = Transcriber(
        model_path=config.model,
        model_arch=ModelArch.SMALL_STREAMING,
        update_interval=config.update_interval,
        options={
            "keyterms": attention_name,
            "keyterm_boost": "3.0",
            "context": f"The local voice assistant is named {attention_name}.",
            "context_max_terms": "8",
            # Oreo owns utterance segmentation. Running Moonshine's VAD on the
            # already segmented stream changed boundaries and reduced accuracy
            # compared with the accepted whole-utterance benchmark.
            "vad_threshold": "0",
            "vad_max_segment_duration": "30",
            "return_audio_data": "false",
        },
    )
    active = False
    latest = ""

    def observe(event: object) -> None:
        nonlocal latest
        line = getattr(event, "line", None)
        if line is None:
            return
        candidate = line.text.strip()
        if candidate and candidate != latest:
            latest = candidate
            write_frame(FRAME_PARTIAL, candidate.encode("utf-8"))

    transcriber.add_listener(observe)
    write_frame(FRAME_READY)
    while True:
        incoming = read_frame()
        if incoming is None:
            break
        command, payload = incoming
        if command == COMMAND_BEGIN and not payload and not active:
            latest = ""
            transcriber.start()
            active = True
            write_frame(FRAME_ACK)
        elif command == COMMAND_AUDIO and active:
            transcriber.add_audio(pcm_floats(payload), SAMPLE_RATE)
        elif command == COMMAND_FINISH and not payload and active:
            result = transcriber.stop()
            active = False
            final = transcript_text(result) if result is not None else latest
            latest = final
            write_frame(FRAME_FINAL, final.encode("utf-8"))
        elif command == COMMAND_ABORT and not payload and active:
            transcriber.stop()
            active = False
            latest = ""
            write_frame(FRAME_ACK)
        else:
            raise ValueError("invalid worker protocol transition")
    transcriber.close()
    return 0


def main() -> int:
    try:
        return run()
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        try:
            write_frame(FRAME_ERROR)
        except (BrokenPipeError, OSError, ValueError):
            pass
        print(f"moonshine worker failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
