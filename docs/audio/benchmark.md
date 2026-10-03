# Offline STT benchmark

The benchmark compares cached engines with identical 16 kHz mono PCM16 WAV
fixtures. It measures model load time, warm-up time, per-fixture warm latency,
real-time factor, current resident memory, deterministic output, and normalized
word error rate. Reports include package, platform, model archive, and fixture
hashes so results are comparable across the laptop and later SBC runs.

Copy `tests/audio/fixtures.example.json` to
`tests/audio/fixtures.local.json`, record each listed phrase into its matching
ignored WAV path, then run both engines with the same manifest. Each recording
must contain one command, be no longer than 30 seconds, and contain no private
speech. Local fixture WAVs and benchmark reports must not be committed.

Python packages are benchmark-only adapters. They do not enter the Oreo daemon
or production container. The selected production engine will remain behind the
Rust `StreamingTranscriber` contract and must pass cancellation and soak tests
before issue WP-004 can close.

## Operator workflow

Create an isolated benchmark environment and install the pinned adapters:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "numpy==2.5.3" "sherpa-onnx==1.13.8" "vosk==0.3.45"
```

Create the local manifest and fixture directory:

```bash
cp tests/audio/fixtures.example.json tests/audio/fixtures.local.json
mkdir -p tests/audio/fixtures
```

Record each expected phrase from the manifest as 16 kHz mono PCM16. For
example, speak “set a timer for five minutes” and stop recording with Ctrl-C:

```bash
arecord -q -f S16_LE -r 16000 -c 1 tests/audio/fixtures/timer-five-minutes.wav
```

Run the dependency-free harness check, then both engine benchmarks:

```bash
.venv/bin/python scripts/benchmark-stt.py self-test
.venv/bin/python scripts/benchmark-stt.py run --engine sherpa --manifest tests/audio/fixtures.local.json --output target/audio-bench/sherpa.json
.venv/bin/python scripts/benchmark-stt.py run --engine vosk --manifest tests/audio/fixtures.local.json --output target/audio-bench/vosk.json
```

Use the same recordings, repetition count, and thread count for both reports.
The default is five measured runs and one inference thread, matching the first
SBC-oriented comparison. Output variation is recorded as hypothesis variants
and a mean WER across runs; it is a benchmark result rather than a fatal error.

## Rust adapter smoke test

The selected Vosk engine is optional so normal development builds do not need
the native library. The local Python package already contains the matching
`libvosk.so`; point the Rust linker and loader at it, then transcribe one of the
same ignored fixtures through Oreo's bounded source, converter, and adapter:

```bash
export OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk"
export LD_LIBRARY_PATH="$OREO_VOSK_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo run -p elixpo-cli --features vosk-stt -- audio transcribe-test tests/audio/fixtures/timer-five-minutes.wav
```

Set `OREO_VOSK_MODEL_DIR` only when the model is not at the default
`models/cache/vosk-model-small-en-us-0.15` path. The command prints the final
transcript and transcription time; it does not persist either the WAV or the
transcript. Wake-word detection is deliberately outside this adapter and will
be selected later.
