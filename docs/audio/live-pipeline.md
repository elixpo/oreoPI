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

Cold attention is a three-way local decision. Direct address proceeds, reported
or third-person mentions such as “I am speaking to Oreo” remain silent, and a
narrow ambiguous score emits `voice_clarification_needed` without entering the
agent queue. The reviewed mention examples live beside the wake corpus. Pocket
TTS turns that event into a fixed local “Were you talking to me?” prompt; the
ambiguity itself never causes a Pollinations request.

The maximum utterance remains 30 seconds. Rejected cold candidates use a 500 ms
reset cooldown to prevent one utterance from triggering twice. A bounded
supervisor restarts the local worker and microphone session with backoff after
recoverable failures instead of permanently stopping voice input.

For transcription-only validation, run the daemon from the repository root:

```bash
rtk env OREO_VOICE_ENABLED=1 OREO_REPO_ROOT="$PWD" \
  OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run -p oreo-daemon --features voice-runtime
```

For the persistent agent path, put `POLLINATIONS_API_KEY` in the ignored
`.env.local` file. `OREO_MODEL` is optional and defaults to
`openai/gpt-5.4-nano`. The daemon reads only those two assignments without
executing the file; process environment values take precedence. Then run:

```bash
rtk env OREO_VOICE_ENABLED=1 OREO_AGENT_ENABLED=1 OREO_REPO_ROOT="$PWD" \
  OREO_VOSK_LIB_DIR="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  LD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/vosk" \
  cargo run -p oreo-daemon --features voice-agent
```

Wait for `voice_listening`, then test all interaction modes above. A successful
transcription emits `voice_command_ready`; its text is deliberately absent from
logs. Session closure emits `voice_conversation_ended`. Stop the daemon from
another terminal with `cargo run -p elixpo-cli -- daemon stop`.

With `voice-agent`, commands enter a bounded eight-message controller on a
separate network thread. Response deltas are split at natural sentence
boundaries—or a bounded 80–160 byte soft boundary when punctuation is delayed—
and sent immediately to the prewarmed Pocket TTS worker, so playback starts
before the model completes its full answer. A 60 ms lead-in protects the first
spoken word from device startup clipping. A new accepted utterance
cancels queued and active speech before it steers or chains the agent. Leading
cues in `config/voice-steering-cues.tsv` can queue or replace work. The agent
retains at most four completed conversation turns and requests at most 256
output tokens. Speech, agent, and cancellation logs expose only fixed lifecycle
events; transcripts, response text, tool payloads, and credentials are excluded.

The microphone remains live during playback. The speaker adapter tracks queued
samples rather than treating synthesis completion as playback completion. While
audio is active, a bounded transient reference of Oreo's last eight spoken
fragments rejects high-overlap microphone transcripts as speaker echo, with a
750 ms tail for room reverberation. A distinct utterance remains a barge-in: it
stops the output stream immediately and steers or queues the next turn. The
reference is never persisted or logged. This is application-level echo
protection; SBC images may additionally enable their codec's hardware AEC when
the selected microphone and speaker expose it.

Speech lifecycle logs distinguish the stages precisely: `speech_generating`
means Pocket TTS is synthesizing, `speech_started` is emitted only after the
CPAL callback has consumed real queued samples, and `speech_finished` means the
physical playback queue drained. `speech_failed` therefore indicates a real
worker, conversion, or speaker-consumption failure rather than silently
claiming that audio played.
