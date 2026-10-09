# Generic openWakeWord candidate

Date: 2026-10-05

## Boundary

openWakeWord is Oreo's acoustic candidate detector, not its final intent
authority. Its positive label means that the spoken identity is present
anywhere in the window. Training therefore includes the identity alone and a
large, reviewed mix of direct-address and product carrier sentences. Product
sentences are intentionally positive here: this prevents the acoustic model
from learning semantics and leaves Vosk plus the intent classifier responsible
for rejecting product discussion. The carriers are training variation, not a
runtime phrase lookup table. Each carrier is capped at two neighboring words
on either side of the identity so the classifier retains a short input window
and low SBC cost. A positive score only releases the bounded
three-second window to Vosk and the local contextual classifier. The capture
keeps one bounded second after an early acoustic hit so the classifier sees
enough context to distinguish direct address from product discussion.

Energy VAD is not used to identify Oreo. openWakeWord's optional Silero VAD may
later gate non-speech noise, but an openWakeWord score, a Vosk transcript, and
the contextual intent decision remain separate evidence.

The corpus-driven generator excludes Oreo product sentences from the negative
class and includes them in identity-present positives. Its negative set
contains speech without the identity plus acoustic confusables such as
“audio,” “ordeal,” “all you,” “oriole,” and “stereo.” Changing the reviewed
corpus regenerates the training configuration; application code contains no
wake-phrase list.

## Isolated environment

openWakeWord 0.6.0 is pinned at upstream tag commit
`c8ef6912c5feccf1037b852d9bc6c7ed644135ba`. Its Linux dependency on
`tflite-runtime` cannot currently resolve under Oreo's Python 3.14 virtual
environment. Use the installed Python 3.11 interpreter and keep this large
training/runtime stack isolated:

```bash
rtk uv venv --python /home/elixpo/.local/bin/python3.11 .venv-wake
rtk uv pip install --python .venv-wake/bin/python \
  "openwakeword==0.6.0" "vosk==0.3.45"
```

Do not install openWakeWord into `.venv`; that environment remains the pinned
STT/TTS benchmark environment.

The wheel does not bundle the shared ONNX feature extractors. Cache the two
small, checksum-pinned upstream files inside the repository:

```bash
rtk ./scripts/fetch-openwakeword-assets.sh
```

The runtime and benchmark receive these paths explicitly and never write model
files into `.venv-wake`.

## Generate the training configuration

The checked-in base config follows openWakeWord's official custom-model
schema. Generate the ignored, absolute-path version from Oreo's reviewed
corpus:

```bash
rtk .venv/bin/python scripts/prepare-openwakeword-training.py self-test
rtk .venv/bin/python scripts/prepare-openwakeword-training.py build
```

The resulting file is
`models/training/openwakeword/oreo-training.json`. JSON is valid YAML and can
be read by the upstream `train.py`. The preparation step is small; it does not
download datasets or start training.

## Training gate

The official high-quality recipe includes a 17.28 GB negative feature file.
That is excessive for the first laptop candidate. Oreo instead starts with
10,000 synthetic identity samples, 10,000 generated confusable negatives, a
2,000/2,000 validation split, and 20,000 training steps. Its negative
validation features are derived from the held-out generated negatives. This is
a cheap development model, not evidence of a production false-accept rate.

Install the CPU-only training stack. PyTorch is installed from its CPU wheel
index first so the environment does not pull unused CUDA libraries:

```bash
rtk uv pip install --python .venv-wake/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.2.2" "torchaudio==2.2.2"

rtk uv pip install --python .venv-wake/bin/python \
  -r requirements/openwakeword-train.txt
```

Then fetch the two small shared feature models and bootstrap the pinned Piper
sample generator. The bootstrap downloads one 204,089,915-byte English
multi-speaker checkpoint and records its acquired SHA-256 locally:

```bash
rtk ./scripts/fetch-openwakeword-assets.sh
rtk ./scripts/bootstrap-openwakeword-training.sh
rtk .venv-wake/bin/python scripts/run-openwakeword-training.py preflight
```

The following are the long-running stages. Run them separately so a completed
stage remains reusable after an interruption:

```bash
rtk .venv-wake/bin/python scripts/run-openwakeword-training.py phase generate
rtk .venv-wake/bin/python scripts/run-openwakeword-training.py phase augment
rtk .venv-wake/bin/python scripts/run-openwakeword-training.py phase train
rtk .venv-wake/bin/python scripts/run-openwakeword-training.py install-candidate
```

The runner verifies the installed package versions and reviewed upstream
`train.py`, supplies the repository-cached feature backbones, flattens held-out
negative features for the validation interface, exports ONNX, and omits the
unneeded TensorFlow/TFLite conversion. The spellings `orio` and `oreos` are
absent from CMUdict, so the runner supplies their reviewed `AO R IY OW` and
`AO R IY OW Z` pronunciations locally. Preflight rejects any remaining unknown
target word instead of using openWakeWord's obsolete DeepPhonemizer URL.

The first exported `oreo.onnx` is only a candidate. It must pass the existing
eight-fixture cascade, new pronunciation/accent fixtures, television/music
negatives, an overnight false-activation soak, and an AArch64 resource run.
The target is not approved merely because it recognizes the training phrases.

### Rejected isolated-identity candidate

The first 205,430-byte ONNX candidate (`a3e5cd99be4f3c6c960b1c55a0f17355f43a1d7c9fddc522dc22447e667f5b76`)
trained only on isolated `oreo`/`orio`. It reached 78.05% synthetic validation
accuracy, 56.2% recall, and 2.57 synthetic false positives/hour. At threshold
0.5 it detected only the standalone-name fixture; lowering the threshold made
valid phrases overlap with `stereo` and product negatives. No threshold passed
the cascade, so this artifact is rejected. The context-v2 experiment replaces
it with the identity-present carrier strategy above.

### Context-v2 candidate

The context-v2 artifact is also 205,430 bytes
(`3f15cc0aa8e23a897e089475b5447f16f049a4b70cb65da8131db5780a35674e`).
Its synthetic-only aggregate is poor (70.53% accuracy, 42.45% recall, and
78.05 false positives/hour), so it cannot be promoted from training metrics.
On the recorded cascade at threshold 0.5, however, it accepts all three
contextual wakes and rejects all four negatives; only standalone Oreo is
missed. Vosk renders that standalone recording as `ordeal`, so `ordeal` is a
reviewed STT identity alias like `audio`. Ordinary sentences containing the
word still receive negative contextual evidence. A permissive threshold is
experimental until hard-confusable recordings and a background soak pass. At
threshold 0.005, after adding the observed `ordeal` STT alias, the existing
eight-fixture cascade completed 24/24 stable runs with zero false accepts and
zero false rejects; p95 end-to-end latency was about 815 ms. This is sufficient
to promote context-v2 to the current candidate, but not to select it for the
product.

Place a candidate at `models/cache/openwakeword-oreo/oreo.onnx`, then run the
same end-to-end fixtures through the openWakeWord candidate, Vosk, and intent
gate:

```bash
rtk .venv-wake/bin/python scripts/benchmark-wake.py run \
  --kws-engine openwakeword \
  --manifest tests/audio/wake-fixtures.local.json \
  --openwakeword-threshold 0.005 \
  --output target/audio-bench/wake-openwakeword.json
```

The report records the exact ONNX digest and raw candidate score so thresholds
can be compared without replacing the model artifact.

## Background soak

The reproducible first gate uses one hour from the official LibriSpeech
`test-other` corpus. It is challenging English audiobook speech at 16 kHz and
is licensed CC BY 4.0. The builder downloads the 328 MB archive, checks the MD5
digest published by OpenSLR, caches the source outside Git, and creates a
deterministic PCM16 mono WAV:

```bash
rtk ./scripts/fetch-wake-soak-corpus.sh
```

Process it with the memory-bounded cascade runner:

```bash
rtk .venv-wake/bin/python scripts/soak-wake.py \
  --wav target/audio-bench/librispeech-test-other-1h.wav \
  --threshold 0.005 \
  --output target/audio-bench/wake-librispeech-1h.json
```

The runner retains only event timestamps, scores, transcripts, and decisions in
the report. Candidate rate measures Vosk/CPU pressure; accepted false
activations determine the user-visible gate. The candidate must record zero
accepted activations in this speech-heavy hour before a longer corpus run.

The first run at the original single intent threshold produced 229 acoustic
candidates and 40 accepted false activations. None of the 229 Vosk transcripts
contained a reviewed Oreo identity alias: ordinary audiobook dialogue was being
accepted solely because it resembled direct address. An initial `5.0` nameless
threshold rejected all of those candidates, but an independent second hour
found one false activation at score `5.60` among 176 candidates. The only
recorded positive where Vosk loses the identity scores `9.18`.

The intent gate therefore uses two confidence levels. A transcript with an
identity alias retains the reviewed `1.0` threshold; a transcript where Vosk
lost the identity must reach the high-confidence `7.0` log-odds boundary. Both
observed hours are now calibration data rather than independent selection
gates. Replaying all 405 candidates through this rule yields zero accepted
activations, while the 12-fixture cascade remains 36/36 correct.

Build a disjoint third hour from the already-cached archive and use that as the
untouched validation gate. The second argument is the source offset in seconds,
so this does not download anything again:

```bash
rtk ./scripts/fetch-wake-soak-corpus.sh \
  target/audio-bench/librispeech-test-other-validation-1h.wav 7200
rtk .venv-wake/bin/python scripts/soak-wake.py \
  --wav target/audio-bench/librispeech-test-other-validation-1h.wav \
  --threshold 0.005 \
  --output target/audio-bench/wake-librispeech-validation-1h.json
```

This reproducible test replaces the laptop-room recording, not the final device
acceptance test. Once Oreo moves to its SBC and microphone enclosure, it still
needs an on-device ambient soak because microphone gain, echo, room acoustics,
television, and music are hardware-specific.

## Licence boundary

openWakeWord code is Apache-2.0. Upstream states that its included pretrained
models are CC BY-NC-SA 4.0. Oreo records that restriction explicitly and keeps
the runtime/model isolated so a future commercially cleared replacement can be
substituted without changing the Vosk or intent contracts.
