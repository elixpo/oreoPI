#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
cache_dir="$repo_root/models/cache"
mkdir -p -- "$cache_dir"
work_dir="$(mktemp -d "$cache_dir/.fetch.XXXXXX")"

cleanup() {
  rm -rf -- "$work_dir"
}

trap cleanup EXIT
trap 'exit 130' HUP INT TERM

download() {
  local url="$1"
  local destination="$2"
  local expected_bytes="$3"

  curl --fail --location --retry 3 --retry-all-errors \
    --output "$destination" "$url"

  local actual_bytes
  actual_bytes="$(stat --format='%s' "$destination")"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "error: archive size mismatch: expected $expected_bytes, got $actual_bytes" >&2
    return 1
  fi
}

record_digest() {
  local archive="$1"
  local model_id="$2"
  local checksum
  checksum="$(sha256sum "$archive")"
  printf '%s  %s.archive\n' "${checksum%% *}" "$model_id" \
    > "$cache_dir/$model_id.sha256.local"
}

validate_member_paths() {
  local member
  while IFS= read -r member; do
    case "$member" in
      /* | ../* | */../* | */..)
        echo "error: archive contains an unsafe path: $member" >&2
        return 1
        ;;
    esac
  done
}

fetch_sherpa() {
  local model_id="sherpa-zipformer-en-20m-int8-2023-02-17"
  local upstream_name="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
  local destination="$cache_dir/$model_id"
  local archive="$work_dir/$upstream_name.tar.bz2"
  local unpacked="$work_dir/$upstream_name"
  local staged="$work_dir/$model_id"

  if [[ -d "$destination" ]]; then
    echo "$model_id is already cached"
    return
  fi

  download \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$upstream_name.tar.bz2" \
    "$archive" \
    127887156
  record_digest "$archive" "$model_id"
  tar -tjf "$archive" | validate_member_paths
  tar -xjf "$archive" -C "$work_dir"
  mkdir -- "$staged"

  install -m 0644 "$unpacked/encoder-epoch-99-avg-1.int8.onnx" "$staged/"
  install -m 0644 "$unpacked/decoder-epoch-99-avg-1.onnx" "$staged/"
  install -m 0644 "$unpacked/joiner-epoch-99-avg-1.int8.onnx" "$staged/"
  install -m 0644 "$unpacked/tokens.txt" "$staged/"
  if [[ -f "$unpacked/LICENSE" ]]; then
    install -m 0644 "$unpacked/LICENSE" "$staged/"
  fi

  mv -- "$staged" "$destination"
  echo "$model_id cached at $destination"
}

fetch_vosk() {
  local model_id="vosk-small-en-us-0.15"
  local upstream_name="vosk-model-small-en-us-0.15"
  local destination="$cache_dir/$upstream_name"
  local archive="$work_dir/$upstream_name.zip"
  local unpacked="$work_dir/$upstream_name"

  if [[ -d "$destination" ]]; then
    echo "$model_id is already cached"
    return
  fi

  download \
    "https://alphacephei.com/vosk/models/$upstream_name.zip" \
    "$archive" \
    41205931
  record_digest "$archive" "$model_id"
  unzip -Z1 "$archive" | validate_member_paths
  unzip -q "$archive" -d "$work_dir"

  for required in am conf graph ivector; do
    if [[ ! -e "$unpacked/$required" ]]; then
      echo "error: Vosk archive is missing $required" >&2
      return 1
    fi
  done

  mv -- "$unpacked" "$destination"
  echo "$model_id cached at $destination"
}

fetch_moonshine() {
  local model_id="moonshine-small-streaming-en-quantized-2026-08-21"
  local moonshine_cache="$cache_dir/moonshine-voice"
  local destination="$moonshine_cache/download.moonshine.ai/model/small-streaming-en/quantized_26_08_21"
  local cli="${MOONSHINE_VOICE_CLI:-$repo_root/.venv-stt/bin/moonshine-voice}"
  local digest_file="$cache_dir/$model_id.files.sha256.local"
  local required=(
    adapter.ort
    cross_kv.ort
    decoder_kv.ort
    encoder.ort
    frontend.model.ort
    frontend.weights.ort
    streaming_config.json
    tokenizer.bin
  )

  if [[ ! -x "$cli" ]]; then
    echo "error: moonshine-voice 0.1.5 is required at $cli" >&2
    echo "install requirements/moonshine-stt.txt into .venv-stt first" >&2
    return 1
  fi

  if [[ ! -d "$destination" ]]; then
    MOONSHINE_VOICE_CACHE="$moonshine_cache" \
      "$cli" download --stt --language en --model-arch 4
  fi

  local artifact
  for artifact in "${required[@]}"; do
    if [[ ! -f "$destination/$artifact" ]]; then
      echo "error: Moonshine cache is missing $artifact" >&2
      return 1
    fi
  done

  (
    cd -- "$destination"
    sha256sum "${required[@]}"
  ) > "$digest_file"
  local tree_digest
  tree_digest="$(sha256sum "$digest_file")"
  printf '%s  %s.files\n' "${tree_digest%% *}" "$model_id" \
    > "$cache_dir/$model_id.sha256.local"
  echo "$model_id cached at $destination"
}

case "${1:-}" in
  sherpa)
    fetch_sherpa
    ;;
  vosk)
    fetch_vosk
    ;;
  moonshine)
    fetch_moonshine
    ;;
  all)
    fetch_sherpa
    fetch_vosk
    fetch_moonshine
    ;;
  *)
    echo "usage: $0 <sherpa|vosk|moonshine|all>" >&2
    exit 2
    ;;
esac
