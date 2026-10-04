# Live voice-input pipeline

The developer pipeline connects the default microphone to Oreo without a wake
word or speech output:

```text
Enter press -> CPAL capture -> bounded conversion -> Vosk -> Oreo agent -> text
```

Vosk is loaded before capture begins. Raw PCM stays in bounded transient
buffers and is never written to state, logs, or session journals. Capture is
limited to 30 seconds by the SBC profile. Any dropped chunk, device error,
transcription failure, or agent failure faults the current turn instead of
using a potentially incomplete command.

Use the native library installed in the local benchmark environment. The
offline command validates microphone capture, conversion, transcription, and
agent handoff without a network provider:

```bash
rtk env OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run -p elixpo-cli --features vosk-stt -- voice --offline
```

For the live agent path, add the Pollinations credential. GPT-5.4 Nano is the
selected default; set `OREO_MODEL` only to evaluate an explicit alternative:

```bash
rtk env OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  POLLINATIONS_API_KEY="..." \
  cargo run -p elixpo-cli --features vosk-stt -- voice
```

Press Enter once to begin capture, speak one command, then press Enter again to
release. The command prints the recognized text and response; it does not
persist either. Wake-word activation and TTS/emotional delivery are separate
pipeline stages.

## Repeatable base recording

Use one ignored 16 kHz mono PCM WAV while tuning recognition or comparing agent
models. This prevents a new microphone take from changing the input between
runs:

```bash
arecord -q -f S16_LE -r 16000 -c 1 tests/audio/fixtures/base-voice.wav
```

Stop `arecord` with Ctrl-C, then load `.env.local` and send the same recording
through the complete transcription and agent path:

```bash
set -a
source .env.local
set +a
rtk env OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run -p elixpo-cli --features vosk-stt -- \
  voice --metrics --wav tests/audio/fixtures/base-voice.wav
```

`--metrics` adds private-safe STT confidence and agent token/latency records to
standard error. It does not include the recording, transcript, prompt, or
response.

The repository ignores every WAV file. Keep the expected sentence separately
in the local benchmark manifest when measuring word error rate; do not tune a
free-conversation recognizer with a restrictive command grammar.
