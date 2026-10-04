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
  a29f182c6cb55ac1f1369e82dc801376e4207c580409a6d11208bbcf32f78820
  ad8b2142cca2c9a0dce8349138fb2afc4d558a884ea5ff1c3f9439a87fff7cdb
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
