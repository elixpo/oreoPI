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

The Rust adapter also emits aggregate mean/minimum word confidence and word
count. Confidence is diagnostic evidence, not an automatic correction or
acceptance threshold: a recognizer can be confidently wrong, and rejecting a
single low-confidence name can discard an otherwise correct command.

## Production-adapter soak

Run a one-minute check first, then the 30-minute laptop acceptance soak. The
command loads Vosk once, replays the same bounded WAV in-process, and requires
identical hypotheses, cached p95 below 1.2 seconds, and no more than 8 MiB end
RSS growth after five warm-up turns. It reports peak RSS separately.

```bash
rtk env OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run -p elixpo-cli --features vosk-stt -- \
  audio stt-soak tests/audio/fixtures/base-voice.wav 1

rtk env OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run --release -p elixpo-cli --features vosk-stt -- \
  audio stt-soak tests/audio/fixtures/base-voice.wav 30
```

The release build is the acceptance measurement. A broader quiet/noisy fixture
manifest and an AArch64 run remain required before the production image is
approved; the fixed base recording closes only the repeatable laptop path.

## PocketTTS candidate benchmark

PocketTTS is evaluated separately because its Python/PyTorch reference runtime
is not part of the daemon. The harness pins PocketTTS 3.3.0, the six-layer
English model revisions, 24 kHz mono output, dynamic int8, and the reviewed
`alba` voice. It measures cold model/voice loading separately from warm first
audio, generation time, real-time factor, RSS, and cancellation. Generated WAV
files and the Hugging Face cache are ignored by Git.

First accept the gated model terms on the upstream Hugging Face page and make
`HF_TOKEN` available to the process (or use `hf auth login`). Install the
CPU-only build into the existing virtual environment; this is the large
download and must not be added to a production image yet:

```bash
rtk .venv/bin/python -m pip install "pocket-tts==3.3.0" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

The script automatically directs Hugging Face into
`models/cache/huggingface`. Run the dependency-free check first, then the
measured pass. The WAV directory is optional but recommended for the operator
listening check:

```bash
rtk .venv/bin/python scripts/benchmark-tts.py self-test
rtk .venv/bin/python scripts/benchmark-tts.py run \
  --output target/audio-bench/pocket-tts.json \
  --save-audio target/audio-bench/pocket-tts-wav
```

After the first authenticated download, repeat without network access to prove
the device path is local and the cache is complete:

```bash
rtk .venv/bin/python scripts/benchmark-tts.py run --offline \
  --output target/audio-bench/pocket-tts-offline.json
```

The laptop candidate gate is warm first-audio p95 below 500 ms, median real-
time factor above 1.0, successful cancellation, and no invalid samples. Listen
to every first-run WAV for intelligibility and pronunciation; numeric speed
results cannot approve voice quality. Peak RSS is recorded now and compared
against the final daemon/SBC memory budget before selection.

### Embedded sherpa-onnx comparison

The Python reference passes the timing gates but consumes about one GiB RSS.
Before integrating it, compare the official sherpa-onnx int8 conversion using
the same phrases and metrics. Downloading and extracting the checksum-pinned
98 MB archive is the long step:

```bash
rtk ./scripts/fetch-tts-models.sh
```

The existing `sherpa-onnx==1.13.8` benchmark package can then measure its
PocketTTS backend. This run uses the archive's bundled reference WAV only for
runtime comparison; it is not the final product voice.

```bash
rtk .venv/bin/python scripts/benchmark-tts.py run \
  --engine sherpa-onnx \
  --output target/audio-bench/sherpa-pocket-tts.json \
  --save-audio target/audio-bench/sherpa-pocket-tts-wav
```

Do not add the Rust dependency until this report demonstrates materially lower
RSS, acceptable first-audio latency, correct cancellation, and acceptable
spoken output. This avoids committing the daemon to a native runtime based on
archive size alone.
