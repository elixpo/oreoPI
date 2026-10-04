# ADR 0009: PocketTTS is the laptop voice-quality primary

Status: accepted for the laptop profile

Date: 2026-10-04

## Decision

Use PocketTTS 3.3.0 with the six-layer English model, dynamic int8, and the
reviewed Alba voice as Oreo's laptop TTS primary. Run it outside the Rust daemon
behind the existing `StreamingSynthesizer` contract. The worker is started
before a response needs speech, reused for an active conversation, and evicted
after an idle timeout so PyTorch is not part of Oreo's idle memory footprint.

Do not promote sherpa-onnx PocketTTS or Kitten Nano solely for their lower
memory. Keep sherpa PocketTTS as an ARM/runtime experiment; reject Kitten Nano
as the product voice for this pass. Hardware release still requires an AArch64
measurement and may select a more efficient runtime only if it preserves the
accepted voice quality.

## Evidence

All candidates used the same four bounded English text fixtures, two CPU
threads, three repetitions, 24 kHz mono output, and an operator listening test.

| Candidate | First-audio p95 | Median speed | Loaded RSS | Peak RSS | Listening result |
|---|---:|---:|---:|---:|---|
| PocketTTS 3.3.0 Python int8 | 105 ms | 8.50x | 960,784 KiB | 1,055,640 KiB | Best; clear opening and strongest overall voice |
| sherpa PocketTTS int8, 2 steps | 1,624 ms | 3.24x | 353,452 KiB | 629,572 KiB | Second; first word blurry and overall worse than reference |
| Kitten Nano 0.8 int8 | 1,409 ms | 5.40x | 130,496 KiB | 340,756 KiB | Worse than both PocketTTS paths |

PocketTTS is the only candidate that passes the 500 ms first-audio gate and the
listening gate. Kitten's short phrases can begin below 500 ms, but its report
p95 and long-sentence cancellation are sentence-sized rather than genuinely
early streaming callbacks.

## Consequences

- TTS runs in a crash-isolated local process with a bounded framed protocol;
  raw PCM is never persisted or logged.
- The worker keeps the model and voice state warm during an active conversation
  and supports cancellation by terminating generation promptly.
- Idle eviction is required before the laptop adapter is considered complete.
- PocketTTS's active peak is close to the provisional 1,200 MiB whole-device
  ceiling. Full-pipeline soak and hardware measurements must include the daemon,
  Vosk, playback, and operating-system headroom rather than approving TTS alone.
- A lighter native implementation can replace the worker without changing the
  audio contract, but only after it matches the reference listening quality.
