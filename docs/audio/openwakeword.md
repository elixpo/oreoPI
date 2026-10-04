# Generic openWakeWord candidate

Date: 2026-10-05

## Boundary

openWakeWord is Oreo's acoustic candidate detector, not its final intent
authority. The binary model learns only the spoken identity (`oreo` plus the
pronunciation alias `orio`) across synthetic voices and acoustic augmentation.
It does not learn command sentences or a four-phrase lookup table. Because it
runs over a sliding audio window, that identity can occur anywhere in an
otherwise unseen sentence. A positive score only releases the bounded
three-second window to Vosk and the local contextual classifier. The capture
keeps one bounded second after an early acoustic hit so the classifier sees
enough context to distinguish direct address from product discussion.

Energy VAD is not used to identify Oreo. openWakeWord's optional Silero VAD may
later gate non-speech noise, but an openWakeWord score, a Vosk transcript, and
the contextual intent decision remain separate evidence.

The corpus-driven generator intentionally excludes Oreo product sentences from
acoustic training: openWakeWord should hear the name permissively, then let the
intent classifier reject product discussion. Its negative set contains speech
without the identity plus acoustic confusables such as “audio,” “ordeal,” “all
you,” “oriole,” and “stereo.” Changing the reviewed corpus regenerates the
training configuration; application code contains no wake-phrase list.

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
unneeded TensorFlow/TFLite conversion. The spelling alias `orio` is absent from
CMUdict, so the runner supplies its reviewed `AO R IY OW` pronunciation locally;
it does not use openWakeWord's obsolete DeepPhonemizer download URL.

The first exported `oreo.onnx` is only a candidate. It must pass the existing
eight-fixture cascade, new pronunciation/accent fixtures, television/music
negatives, an overnight false-activation soak, and an AArch64 resource run.
The target is not approved merely because it recognizes the training phrases.

Place a candidate at `models/cache/openwakeword-oreo/oreo.onnx`, then run the
same end-to-end fixtures through the openWakeWord candidate, Vosk, and intent
gate:

```bash
rtk .venv-wake/bin/python scripts/benchmark-wake.py run \
  --kws-engine openwakeword \
  --manifest tests/audio/wake-fixtures.local.json \
  --openwakeword-threshold 0.5 \
  --output target/audio-bench/wake-openwakeword.json
```

The report records the exact ONNX digest and raw candidate score so thresholds
can be compared without replacing the model artifact.

## Licence boundary

openWakeWord code is Apache-2.0. Upstream states that its included pretrained
models are CC BY-NC-SA 4.0. Oreo records that restriction explicitly and keeps
the runtime/model isolated so a future commercially cleared replacement can be
substituted without changing the Vosk or intent contracts.
