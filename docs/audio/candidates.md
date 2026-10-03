# Audio engine candidates

Status: benchmark plan for WP-004

The `oreo-audio` contracts intentionally contain no engine-specific types.
Candidates become build or image dependencies only after they pass the same
fixtures and resource harness on the reference laptop. Model files are pinned
and checksummed separately from application packages.

| Layer | Primary candidate | Fallback | Benchmark focus |
|---|---|---|---|
| Capture/playback | [CPAL 0.18](https://github.com/RustAudio/cpal/releases/tag/v0.18.0) | Direct ALSA adapter only if CPAL measurements require it | PipeWire/PulseAudio/ALSA behavior, callback overruns, fixed-buffer latency, AMD64/ARM64 packaging |
| Offline STT | [sherpa-onnx 1.13.8](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8) | [Vosk](https://alphacephei.com/vosk/) with a small language model | Accuracy on Oreo command fixtures, endpoint latency, cached latency, RSS, model size, cancellation, Linux AArch64 availability |
| Streaming TTS | [PocketTTS 3.3.0](https://github.com/kyutai-labs/pocket-tts/releases/tag/v3.3.0) | Selected only after the primary misses a hard gate | First-audio latency, real-time factor, RSS, cancellation, pronunciation, voice/model licensing, AMD64/ARM64 packaging |

## Selection rules

- Benchmark identical 16 kHz mono PCM fixtures and at least one real laptop
  microphone for each STT candidate.
- Measure warm and cold runs separately. The acceptance target applies to the
  cached/warm short-command path.
- Reject an engine that needs an unbounded queue, retains raw audio after the
  utterance, cannot cancel, or exceeds the SBC hard memory ceiling.
- Record package, native library, model/config commit, checksum, licence, and
  measured resource table before promoting a candidate into
  `docs/third-party.md`.
- Re-run sherpa-onnx and PocketTTS on Linux AArch64 before the hardware release;
  an upstream binary existing is not a substitute for a device benchmark.

Vosk remains valuable as a low-resource baseline: its official small-model
guidance targets mobile/Raspberry Pi-class systems, but its older native stack
and accuracy must be measured against current sherpa-onnx models rather than
assumed. PocketTTS source is MIT; the selected voice and model artefacts still
require a separate licence/provenance review.
