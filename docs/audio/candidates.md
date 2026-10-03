# Audio engine candidates

Status: laptop STT selection recorded for WP-004; soak and ARM confirmation pending

The `oreo-audio` contracts intentionally contain no engine-specific types.
Candidates become build or image dependencies only after they pass the same
fixtures and resource harness on the reference laptop. Model files are pinned
and checksummed separately from application packages.

| Layer | Primary candidate | Fallback | Benchmark focus |
|---|---|---|---|
| Capture/playback | [CPAL 0.18.2](https://github.com/RustAudio/cpal/releases/tag/v0.18.2), selected | Direct ALSA adapter only if CPAL measurements require it | PipeWire/PulseAudio/ALSA behavior, callback overruns, fixed-buffer latency, AMD64/ARM64 packaging |
| Offline STT | [Vosk 0.3.45](https://alphacephei.com/vosk/) with `vosk-model-small-en-us-0.15`, provisional laptop selection | [sherpa-onnx 1.13.8](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8) with English-only 20M streaming Zipformer INT8 as the resource fallback | Accuracy on Oreo command fixtures, endpoint latency, cached latency, RSS, model size, cancellation, Linux AArch64 availability |
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
- Re-run Vosk, sherpa-onnx, and PocketTTS on Linux AArch64 before the hardware
  release; an upstream binary existing is not a substitute for a device
  benchmark.

Vosk is the provisional primary and low-resource baseline: its official
small-model guidance targets mobile/Raspberry Pi-class systems, but its older
native stack and accuracy must be measured against current sherpa-onnx models
rather than assumed. PocketTTS source is MIT; the selected voice and model
artefacts still require a separate licence/provenance review.

The initial cache candidates are pinned in `models/manifest.toml`. The sherpa
candidate is English-only and documented upstream as suitable for Cortex-A7;
only its roughly 44 MB INT8 runtime subset is retained. The Vosk primary has a
40 MB US-English archive, occupies 68 MB extracted, and is officially listed
for Android and Raspberry Pi. The 36 MB Indian-English Vosk model is excluded
from the first comparison because its published NPTEL word error rate is
49.05%, an excessive accuracy tradeoff for four megabytes of storage.

## Laptop selection result

The first operator-recorded comparison used three English Oreo commands, five
measured runs per command, one inference thread, identical 16 kHz mono PCM16
fixtures, and the reference x86-64 laptop. Aggregate WER includes every run.

| Candidate | Aggregate WER | Warm p95 | After-run RSS | Extracted model | Stability |
|---|---:|---:|---:|---:|---|
| Vosk small en-US 0.15 | 4.71% | 779 ms | 193,476 KiB | 68 MB | 14/15 hypotheses matched the modal transcript |
| Sherpa Zipformer en-20M INT8 | 41.18% | 228 ms | 143,988 KiB | 44 MB | 15/15 hypotheses stable |

Vosk is the provisional primary because it stays below the 1.2-second cached
STT gate while reducing command error by almost an order of magnitude. Sherpa
is faster and lighter but its errors removed the action from “set a timer” and
badly distorted “device status”; it cannot be the default for this fixture set.
Sherpa remains the resource fallback for later hotword or model experiments.

This is not the hardware-release verdict. Vosk must still pass cancellation,
the 30-minute no-growth soak, a larger voice/noise fixture set, and the Linux
AArch64 benchmark before it is included in a production image.
