# ADR 0007: Audio engines sit behind bounded streaming contracts

Status: accepted for WP-004

## Decision

Represent capture and playback as interleaved signed 16-bit PCM chunks of at
most 100 ms. The initial voice path accepts 8-48 kHz and one or two channels;
the default SBC profile uses 20 ms frames, a 30-second capture limit, and a
32-item queue budget.

Capture, streaming STT, streaming TTS, and playback are separate cancellable
interfaces. Push-to-talk lifecycle is an explicit state machine. WAV fixtures
use a strict in-tree parser for deterministic tests, while live devices and ML
engines remain replaceable adapters. A deterministic energy VAD provides a
fixture/fallback endpoint at 300 ms of silence with the default 20 ms frames.

## Consequences

- Audio callbacks never invoke a model, network provider, or SQLite directly.
- Raw PCM is transient and bounded; the audio crate cannot persist or log it.
- Laptop and SBC backends share the same PCM and cancellation contracts.
- CPAL, sherpa-onnx, Vosk, and PocketTTS can be benchmarked without changing
  the runtime-facing API.
- Resampling, channel conversion, learned VAD, pronunciation, and latency
  instrumentation remain explicit layers rather than hidden backend work.
