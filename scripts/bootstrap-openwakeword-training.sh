#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
training_root="$repo_root/models/training/openwakeword"
piper_root="$training_root/piper-sample-generator"
piper_revision="213d4d561ab8a84f71de7dddac827cb07e92c031"
model_name="en_US-libritts_r-medium.pt"
model_url="https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/$model_name"
model_bytes="204089915"

mkdir -p -- \
  "$training_root/data/mit_rirs" \
  "$training_root/data/background_clips" \
  "$training_root/output"

"$repo_root/scripts/fetch-openwakeword-assets.sh"

if [[ -d "$piper_root/.git" ]]; then
  actual_revision="$(git -C "$piper_root" rev-parse HEAD)"
  if [[ "$actual_revision" != "$piper_revision" ]]; then
    echo "error: piper sample generator is at unexpected revision $actual_revision" >&2
    exit 1
  fi
else
  git clone --filter=blob:none --no-checkout \
    https://github.com/rhasspy/piper-sample-generator.git "$piper_root"
  git -C "$piper_root" checkout --detach "$piper_revision"
fi

model_path="$piper_root/models/$model_name"
if [[ ! -f "$model_path" ]]; then
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$model_path" "$model_url"
fi
actual_bytes="$(stat --format='%s' "$model_path")"
if [[ "$actual_bytes" != "$model_bytes" ]]; then
  echo "error: Piper generator checkpoint size mismatch" >&2
  exit 1
fi
actual_sha256="$(sha256sum "$model_path")"
actual_sha256="${actual_sha256%% *}"
printf '%s  %s\n' "$actual_sha256" "$model_name" \
  > "$training_root/piper-generator.sha256.local"

"$repo_root/.venv/bin/python" \
  "$repo_root/scripts/prepare-openwakeword-training.py" build

echo "openWakeWord training sources and Piper checkpoint are ready"
echo "Piper checkpoint SHA-256: $actual_sha256"
