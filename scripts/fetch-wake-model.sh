#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
cache_dir="$repo_root/models/cache"
model_id="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
archive_name="$model_id.tar.bz2"
archive_bytes="32885699"
archive_sha256="68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
destination="$cache_dir/$model_id"
keywords_raw="$repo_root/config/wake-keywords.raw.txt"
keywords="$destination/oreo-keywords.txt"

required_files=(
  encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx
  decoder-epoch-13-avg-2-chunk-8-left-64.onnx
  joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx
  tokens.txt
  en.phone
)

validate_model() {
  local root="$1"
  local required
  for required in "${required_files[@]}"; do
    if [[ ! -f "$root/$required" ]]; then
      echo "error: wake model is missing $required" >&2
      return 1
    fi
  done
}

generate_keywords() {
  local cli="$repo_root/.venv/bin/sherpa-onnx-cli"
  if [[ ! -x "$cli" ]]; then
    echo "error: sherpa-onnx-cli 1.13.8 is required to tokenize wake phrases" >&2
    return 1
  fi
  "$cli" text2token \
    --tokens "$destination/tokens.txt" \
    --tokens-type phone+ppinyin \
    --lexicon "$destination/en.phone" \
    "$keywords_raw" "$keywords"
  if [[ ! -s "$keywords" ]]; then
    echo "error: wake keyword tokenization produced no output" >&2
    return 1
  fi
}

mkdir -p -- "$cache_dir"
if [[ -d "$destination" ]]; then
  validate_model "$destination"
  generate_keywords
  echo "$model_id is already cached; Oreo keywords refreshed"
  exit 0
fi

work_dir="$(mktemp -d "$cache_dir/.fetch-wake.XXXXXX")"
cleanup() {
  rm -rf -- "$work_dir"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

archive="$work_dir/$archive_name"
curl --fail --location --retry 3 --retry-all-errors \
  --output "$archive" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$archive_name"

actual_bytes="$(stat --format='%s' "$archive")"
if [[ "$actual_bytes" != "$archive_bytes" ]]; then
  echo "error: archive size mismatch: expected $archive_bytes, got $actual_bytes" >&2
  exit 1
fi
actual_sha256="$(sha256sum "$archive")"
actual_sha256="${actual_sha256%% *}"
if [[ "$actual_sha256" != "$archive_sha256" ]]; then
  echo "error: wake archive SHA-256 does not match the published digest" >&2
  exit 1
fi

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
mkdir -- "$destination"
for required in "${required_files[@]}"; do
  install -m 0644 "$unpacked/$required" "$destination/"
done
printf '%s  %s.archive\n' "$actual_sha256" "$model_id" \
  > "$cache_dir/$model_id.sha256.local"
generate_keywords
echo "$model_id cached at $destination"
