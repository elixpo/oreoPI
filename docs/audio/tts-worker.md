# PocketTTS worker boundary

Status: adapter implemented; live playback integration pending

PocketTTS runs in a separate local Python process rather than inside the Rust
daemon. `PocketTtsSynthesizer` starts it lazily, can prewarm it while STT or the
agent is working, reuses the loaded model and Alba voice state across active
turns, and terminates it on cancellation, protocol failure, output failure, or
explicit shutdown. The worker exits itself after 60 seconds without a request,
so PyTorch does not remain in Oreo's idle footprint.

The child receives a cleared environment containing only a minimal executable
path, the repository-local Hugging Face cache, offline mode, and CPU thread
limits. Pollinations credentials, user environment variables, prompts, and
session state are not inherited. Worker stderr is discarded and stdout is
reserved exclusively for the framing protocol.

## Protocol

The protocol is private and versioned with the Oreo source rather than exposed
over the local device API.

- Rust sends a four-byte big-endian UTF-8 byte length followed by one bounded
  text chunk. Empty input and input over the configured response bound fail
  before the process is contacted.
- The worker emits a one-byte frame type and four-byte big-endian payload
  length. Frame types are ready, PCM audio, done, and redacted error.
- Ready carries the 24 kHz sample rate. Audio carries little-endian mono PCM16
  capped at 100 ms (4,800 bytes). The Rust reader rejects oversized, truncated,
  malformed, empty, or unexpected frames before they enter playback.
- PCM is passed immediately to the caller's `emit` callback. Neither side
  writes speech audio or text to disk or logs it.
- Cancellation kills the worker, which also stops its model generation. A
  later turn starts a clean worker rather than attempting to reuse a partial
  stream.

## Checks

The Python `--self-test` exercises request and response framing without loading
PyTorch. Rust tests use a deterministic fake worker to verify startup, bounded
PCM decoding, reuse across turns, shutdown, malformed-frame rejection, and
pre-cancellation without spawning. The real model remains an operator test
because loading it consumes roughly one GiB and is intentionally excluded from
routine CI.

```bash
rtk python3 scripts/pocket-tts-worker.py --self-test
rtk cargo test -p oreo-audio --features pocket-tts
rtk cargo clippy -p oreo-audio --all-targets --features pocket-tts -- -D warnings
```

The next checkpoint connects the adapter to `CpalOutput`, applies deterministic
pronunciation before synthesis, and adds a live CLI command that prewarms the
worker before measuring first audible playback.
