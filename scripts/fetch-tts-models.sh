#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
cache_dir="$repo_root/models/cache"
model_id="sherpa-onnx-pocket-tts-int8-2026-01-26"
archive_name="$model_id.tar.bz2"
archive_bytes="98336520"
archive_sha256="2f3b88823cbbb9bf0b2477ec8ae7b3fec417b3a87b6bb5f256dba66f2ad967cb"
destination="$cache_dir/$model_id"

mkdir -p -- "$cache_dir"
work_dir="$(mktemp -d "$cache_dir/.fetch-tts.XXXXXX")"

cleanup() {
  rm -rf -- "$work_dir"
}

trap cleanup EXIT
trap 'exit 130' HUP INT TERM

validate_model() {
  local root="$1"
  local required
  for required in \
    lm_flow.int8.onnx \
    lm_main.int8.onnx \
    encoder.onnx \
    decoder.int8.onnx \
    text_conditioner.onnx \
    vocab.json \
    token_scores.json \
    test_wavs/bria.wav
  do
    if [[ ! -f "$root/$required" ]]; then
      echo "error: TTS model is missing $required" >&2
      return 1
    fi
  done
}

if [[ -d "$destination" ]]; then
  validate_model "$destination"
  echo "$model_id is already cached"
  exit 0
fi

archive="$work_dir/$archive_name"
curl --fail --location --retry 3 --retry-all-errors \
  --output "$archive" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$archive_name"

actual_bytes="$(stat --format='%s' "$archive")"
if [[ "$actual_bytes" != "$archive_bytes" ]]; then
  echo "error: archive size mismatch: expected $archive_bytes, got $actual_bytes" >&2
  exit 1
fi

actual_sha256="$(sha256sum "$archive")"
actual_sha256="${actual_sha256%% *}"
if [[ "$actual_sha256" != "$archive_sha256" ]]; then
  echo "error: archive SHA-256 does not match the published digest" >&2
  exit 1
fi
printf '%s  %s.archive\n' "$actual_sha256" "$model_id" \
  > "$cache_dir/$model_id.sha256.local"

while IFS= read -r member; do
  case "$member" in
    /* | ../* | */../* | */..)
      echo "error: archive contains an unsafe path: $member" >&2
      exit 1
      ;;
  esac
done < <(tar -tjf "$archive")

tar -xjf "$archive" -C "$work_dir"
unpacked="$work_dir/$model_id"
validate_model "$unpacked"

mv -- "$unpacked" "$destination"
echo "$model_id cached at $destination"
