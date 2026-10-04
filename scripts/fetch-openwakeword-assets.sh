#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
cache_dir="$repo_root/models/cache"
asset_id="openwakeword-v0.5.1-features"
destination="$cache_dir/$asset_id"
release_url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

files=(melspectrogram.onnx embedding_model.onnx)
sizes=(1087958 1326578)
digests=(
  ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f
  70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f
)

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_digest="$3"
  local actual_size actual_digest
  actual_size="$(stat --format='%s' "$path")"
  actual_digest="$(sha256sum "$path")"
  actual_digest="${actual_digest%% *}"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "error: $(basename "$path") size mismatch" >&2
    return 1
  fi
  if [[ "$actual_digest" != "$expected_digest" ]]; then
    echo "error: $(basename "$path") SHA-256 mismatch" >&2
    return 1
  fi
}

if [[ -d "$destination" ]]; then
  for index in "${!files[@]}"; do
    verify_file \
      "$destination/${files[$index]}" \
      "${sizes[$index]}" \
      "${digests[$index]}"
  done
  echo "$asset_id is already cached and verified"
  exit 0
fi

mkdir -p -- "$cache_dir"
work_dir="$(mktemp -d "$cache_dir/.fetch-openwakeword.XXXXXX")"
cleanup() {
  rm -rf -- "$work_dir"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

for index in "${!files[@]}"; do
  name="${files[$index]}"
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$work_dir/$name" "$release_url/$name"
  verify_file "$work_dir/$name" "${sizes[$index]}" "${digests[$index]}"
done

mkdir -- "$destination"
for name in "${files[@]}"; do
  install -m 0644 "$work_dir/$name" "$destination/$name"
done
{
  for index in "${!files[@]}"; do
    printf '%s  %s/%s\n' "${digests[$index]}" "$asset_id" "${files[$index]}"
  done
} > "$cache_dir/$asset_id.sha256.local"

echo "$asset_id cached at $destination"
