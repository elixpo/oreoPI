# Offline STT selection record

Date: 2026-10-03

## Decision

Use Vosk 0.3.45 with `vosk-model-small-en-us-0.15` as Oreo's selected offline
STT primary on the reference laptop. Keep sherpa-onnx 1.13.8 with the
English 20M streaming Zipformer INT8 files as the resource fallback and future
optimization candidate.

The choice favors correct device actions over the smallest possible model.
Vosk used 24 MB more extracted model storage and ended the benchmark at 49,488
KiB more resident memory, but produced 4.71% aggregate WER versus Sherpa's
41.18%. Its 779 ms warm p95 remained 421 ms inside Oreo's 1.2-second cached
short-command target.

## Reproduction identity

- Platform: Linux x86-64, Python 3.14.4, 16 logical CPUs, one inference thread.
- Fixtures: three operator-recorded English commands, five runs each, 16 kHz
  mono PCM16, with SHA-256 identities recorded in the local JSON reports.
- Vosk package/model: 0.3.45 / archive SHA-256
  `30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498`.
- sherpa-onnx package/model: 1.13.8 / archive SHA-256
  `9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9`.
- NumPy for the Sherpa adapter: 2.5.3.

## Observations

Vosk transcribed the timer and Kolkata commands exactly in all measured runs.
For device status it returned “their” instead of “the” four times and the exact
reference once. Sherpa was stable but omitted “set a timer,” changed “device
status” substantially, and rendered Kolkata as “Golcotta.”

The production Rust adapter, cancellation cleanup, fixed-WAV agent handoff, and
live laptop microphone-to-agent path pass. The fixed free-conversation WAV
completed cached transcription and agent handoff with 768 ms reported
utterance processing on the reference laptop. The release-mode 30-minute soak
completed 2,515 turns with zero hypothesis mismatches, 758 ms p95, and zero end
RSS growth after warm-up. Mean word confidence was 0.9838 and the minimum was
0.7079 across 18 words.

This finalizes the laptop STT selection. A larger quiet/noisy corpus and the
AArch64 benchmark remain production-image gates rather than blockers for the
next laptop pipeline layer.

Wake-word recognition is intentionally not part of the Vosk adapter. The first
implementation remains push-to-talk and exposes Vosk only through Oreo's
bounded `StreamingTranscriber` contract; a future wake-word layer can trigger
capture without changing STT engine ownership.
