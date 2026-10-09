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

## Always-on daemon wake runtime

The laptop-selected daemon path is:

```text
CPAL -> bounded 16 kHz conversion -> openWakeWord worker
     -> streaming Vosk/intent gate -> immediate turn or VAD follow-up
     -> 30-second contextual conversation window
```

The openWakeWord worker is pinned to `.venv-wake`, has networking and inherited
environment variables removed, accepts only fixed 1,280-sample frames, and
returns one bounded score. Rust owns the rolling window, thresholds, Vosk,
follow-up timeout, cancellation, and microphone lifecycle. Candidate audio is
streamed into Vosk until 300 ms of natural silence; there is no fixed one-second
post-roll.

There are three interaction modes:

- Saying only an accepted rendering of `Oreo` emits `wake_accepted` and opens a
  five-second follow-up window. The follow-up ends after 300 ms of silence.
- An accepted contextual utterance such as “Oreo, set a timer” or “what is the
  weather, Oreo?” is already the command. It emits `voice_command_ready`
  immediately after its endpoint and does not ask for the sentence again.
- Each command opens or refreshes a 30-second engaged window. During that
  window, VAD starts a natural follow-up without another wake phrase. An
  address-only nickname from `config/conversation-addresses.txt` opens the same
  five-second follow-up capture. Those nicknames never enter the cold acoustic
  detector. An exact phrase from `config/conversation-sleep-phrases.txt`, or
  30 seconds without another command, closes the window.

An engaged VAD event that produces no Vosk text is treated as ambient noise. It
returns to the remaining conversation window without emitting a command timeout
or resetting the session.

The maximum utterance remains 30 seconds. Rejected cold candidates use a 500 ms
reset cooldown to prevent one utterance from triggering twice. A bounded
supervisor restarts the local worker and microphone session with backoff after
recoverable failures instead of permanently stopping voice input.

Run the daemon from the repository root:

```bash
rtk env OREO_VOICE_ENABLED=1 OREO_REPO_ROOT="$PWD" \
  OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run -p oreo-daemon --features voice-runtime
```

Wait for `voice_listening`, then test all interaction modes above. A successful
transcription emits `voice_command_ready`; its text is deliberately absent from
logs. Session closure emits `voice_conversation_ended`. Stop the daemon from
another terminal with `cargo run -p elixpo-cli -- daemon stop`. This checkpoint
ends at a bounded command transcript. Routing that transcript into the existing
turn controller is the next layer.
