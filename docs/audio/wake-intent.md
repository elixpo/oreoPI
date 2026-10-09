# Contextual Oreo wake pipeline

Date: 2026-10-09

## Candidate architecture

The laptop candidate uses a four-stage local cascade:

1. The current openWakeWord context-v2 candidate listens only for the assistant
   identity, `Oreo`. It is deliberately permissive and does not contain a list
   of greeting or command phrases. The rejected sherpa candidates remain only
   for reproducible comparison.
2. `WakeAudioWindow` retains only the most recent one to three seconds of
   16 kHz mono PCM. It drops old samples, rejects format changes, and clears on
   cancellation. It does not persist background audio.
3. A KWS candidate sends that bounded window to the already-selected local
   Vosk recognizer. No network request is made.
4. `WakeIntentClassifier` decides whether the arbitrary transcript addresses
   Oreo. It is a tiny in-memory Naive Bayes model trained from the reviewed TSV
   corpus, not a phrase matcher. Expected Vosk renderings of the identity are
   canonicalized from the reviewed `config/wake-identity-aliases.txt` data only
   after KWS fires. The surrounding learned context then distinguishes direct
   address from Oreo product mentions, ordinary uses of “audio,” and
   near-sounding words such as “stereo.” A transcript containing a reviewed
   identity rendering uses the normal intent threshold. When Vosk loses the
   name, the contextual score must pass a separate, much stronger threshold;
   generic dialogue is never accepted merely because it sounds like direct
   address.

Both sherpa candidates missed natural Oreo recordings after threshold and
boost tuning. The corpus-trained openWakeWord model passed the recorded suite
and an untouched one-hour LibriSpeech validation soak, so it is selected for
the laptop daemon; see `docs/audio/openwakeword.md`. AArch64 and final enclosure
testing remain required before hardware selection.

## Privacy and resource bounds

- Audio is mono PCM16 at 16 kHz and the rolling window is capped at three
  seconds (96,000 bytes).
- A KWS hit is only a candidate. No agent request is created unless both Vosk
  and the intent classifier accept it.
- Empty and oversized transcripts fail closed. Cancellation immediately clears
  retained samples.
- Models and local recordings live under ignored cache/fixture paths. Neither
  audio nor transcripts are logged by the daemon path.
- One CPU thread is the benchmark default. KWS score and threshold are recorded
  in each report so laptop and SBC results remain comparable.

## Fetch and record

The long model download is intentionally left to the operator. It verifies the
published size and SHA-256, retains only the chunk-8 int8 files, and generates
the Oreo keyword tokens with the pinned sherpa CLI:

```bash
rtk ./scripts/fetch-wake-model.sh english
```

Create the ignored local manifest and record all eight phrases. Each source WAV
must be no more than 15 seconds and must be 16 kHz mono PCM16. The harness
streams the whole fixture through KWS but exposes only its rolling three-second
window to Vosk, matching the runtime memory bound:

```bash
rtk cp tests/audio/wake-fixtures.example.json tests/audio/wake-fixtures.local.json
rtk mkdir -p tests/audio/fixtures
rtk arecord -q -f S16_LE -r 16000 -c 1 tests/audio/fixtures/wake-oreo.wav
```

Repeat `arecord` for every path in the manifest. Record the negative examples
with normal conversational emphasis; they are essential to measuring false
activation rather than just proving the keyword can trigger.

## Benchmark and selection gate

Run the dependency-free classifier check, then the complete local cascade:

```bash
rtk .venv/bin/python scripts/benchmark-wake.py self-test
rtk .venv/bin/python scripts/benchmark-wake.py run \
  --kws-model english \
  --manifest tests/audio/wake-fixtures.local.json \
  --output target/audio-bench/wake.json
```

The initial laptop gate is zero false accepts and zero false rejects across the
reviewed manifest, stable outcomes across three runs, and p95 processing below
the three-second window duration. Do not loosen the intent threshold to repair
an acoustic miss. Tune one KWS setting at a time using separate reports, then
listen to the relevant WAV and inspect the actual Vosk transcript.

The untouched third LibriSpeech hour produced 209 permissive acoustic
candidates, zero accepted false activations, 594 ms p95 candidate latency, and
20.4x real-time throughput. The native daemon now owns the bounded worker,
three-second wake window, and VAD-ended follow-up command capture. Agent
handoff, barge-in, speech output, and AArch64 measurements are the next layers.

The bilingual phone candidate can be reproduced with
`./scripts/fetch-wake-model.sh bilingual` and `--kws-model bilingual`. Do not
compare reports unless their fixture hashes, KWS settings, and keyword-file
hashes match.
