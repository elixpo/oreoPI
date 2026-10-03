# Local model cache

All downloaded speech models live under `models/cache/`. The cache is local to
this checkout and ignored by Git; only this policy, the candidate manifest, and
the empty cache directory are tracked.

The initial English-only STT candidates deliberately optimize for an SBC:

- `sherpa-zipformer-en-20m-int8-2023-02-17` is the resource fallback. The
  fetcher downloads the official archive temporarily but retains only the INT8
  encoder/joiner, small decoder, and token table needed at runtime.
- `vosk-model-small-en-us-0.15` is the provisional primary after the laptop
  command-fixture comparison. Its roughly 40 MB archive expands to 68 MB
  because its complete directory is required by Vosk.

Run `scripts/fetch-stt-models.sh all` from anywhere inside the checkout. The
script checks the exact upstream archive byte count, records the acquired
SHA-256 digest inside the ignored cache, extracts through a temporary
directory, and installs a model only after all required files exist. Neither
upstream publishes an archive digest, so a release must promote the locally
recorded digests into reviewed provenance before shipping an image.

Model archives and temporary extraction trees are removed after a successful
install. Do not put credentials, recordings, transcripts, or benchmark output
in the model cache.
