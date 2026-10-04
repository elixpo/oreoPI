# Contextual Oreo wake pipeline

Date: 2026-10-04

## Candidate architecture

The laptop candidate uses a four-stage local cascade:

1. sherpa-onnx's English-only 3.3M open-vocabulary KWS candidate listens only
   for the configurable assistant identity, `Oreo`. It uses BPE subwords and
   does not contain a list of greeting or command phrases. Its token file can
   be regenerated without retraining the acoustic model. The earlier bilingual
   phone candidate remains available only for reproducible comparison.
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
   near-sounding words such as “stereo.”

The model is still a candidate until the recorded benchmark and background
soak pass. The benchmark adapter is Python-only; the production daemon will
receive a feature-gated native sherpa adapter after thresholds are selected.

## Privacy and resource bounds

- Audio is mono PCM16 at 16 kHz and the rolling window is capped at three
  seconds (96,000 bytes).
- A KWS hit is only a candidate. No agent request is created unless both Vosk
  and the intent classifier accept it.
- Empty and oversized transcripts fail closed. Cancellation immediately clears
  retained samples.
- Models and local recordings live under ignored cache/fixture paths. Neither
  audio nor transcripts are logged by the future daemon path.
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

After the quiet fixture gate passes, add locally ignored far-field, fan-noise,
music, and television fixtures and run an eight-hour idle soak. The native
daemon adapter, barge-in/follow-up state machine, and AArch64 measurements are
the next implementation steps; this pass selects the trigger behavior and its
resource envelope first.

The bilingual phone candidate can be reproduced with
`./scripts/fetch-wake-model.sh bilingual` and `--kws-model bilingual`. Do not
compare reports unless their fixture hashes, KWS settings, and keyword-file
hashes match.
