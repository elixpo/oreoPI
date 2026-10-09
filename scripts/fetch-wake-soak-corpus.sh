#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
cache_dir="$repo_root/tests/audio/corpora/cache"
archive="$cache_dir/test-other.tar.gz"
archive_part="$archive.part"
dataset_dir="$cache_dir/LibriSpeech/test-other"
output="${1:-$repo_root/target/audio-bench/librispeech-test-other-1h.wav}"
source_url="https://www.openslr.org/resources/12/test-other.tar.gz"
expected_md5="fb5a50374b501bb3bac4815ee91d3135"
target_seconds=3600

for command in curl md5sum tar ffmpeg python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: required command is unavailable: $command" >&2
    exit 1
  fi
done

mkdir -p -- "$cache_dir" "$(dirname -- "$output")"

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --retry-all-errors --continue-at - \
    --output "$archive_part" "$source_url"
  mv -- "$archive_part" "$archive"
fi

actual_md5="$(md5sum "$archive")"
actual_md5="${actual_md5%% *}"
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "error: LibriSpeech archive MD5 does not match the upstream digest" >&2
  echo "remove $archive and retry" >&2
  exit 1
fi

if [[ ! -d "$dataset_dir" ]]; then
  while IFS= read -r member; do
    case "$member" in
      /* | ../* | */../* | */..)
        echo "error: archive contains an unsafe path: $member" >&2
        exit 1
        ;;
    esac
  done < <(tar -tzf "$archive")

  extract_dir="$(mktemp -d "$cache_dir/.extract-librispeech.XXXXXX")"
  cleanup_extract() {
    rm -rf -- "$extract_dir"
  }
  trap cleanup_extract EXIT
  trap 'exit 130' HUP INT TERM
  tar -xzf "$archive" -C "$extract_dir"
  mkdir -p -- "$cache_dir/LibriSpeech"
  mv -- "$extract_dir/LibriSpeech/test-other" "$dataset_dir"
  rm -rf -- "$extract_dir"
  trap - EXIT HUP INT TERM
fi

list_file="$(mktemp "$cache_dir/.librispeech-concat.XXXXXX")"
output_part="$output.part.wav"
cleanup_build() {
  rm -f -- "$list_file" "$output_part"
}
trap cleanup_build EXIT
trap 'exit 130' HUP INT TERM

while IFS= read -r path; do
  printf "file '%s'\n" "$path" >> "$list_file"
done < <(find "$dataset_dir" -type f -name '*.flac' | LC_ALL=C sort)

if [[ ! -s "$list_file" ]]; then
  echo "error: the extracted LibriSpeech corpus contains no FLAC audio" >&2
  exit 1
fi

ffmpeg -hide_banner -loglevel error -nostdin -y \
  -f concat -safe 0 -i "$list_file" -t "$target_seconds" \
  -ac 1 -ar 16000 -c:a pcm_s16le "$output_part"

python3 - "$output_part" "$target_seconds" <<'PY'
import sys
import wave

path = sys.argv[1]
expected_frames = int(sys.argv[2]) * 16_000
with wave.open(path, "rb") as audio:
    valid_format = (
        audio.getnchannels() == 1
        and audio.getsampwidth() == 2
        and audio.getframerate() == 16_000
        and audio.getcomptype() == "NONE"
    )
    frames = audio.getnframes()
if not valid_format or frames != expected_frames:
    raise SystemExit("error: generated soak corpus has an invalid format or duration")
PY

mv -- "$output_part" "$output"
sha256sum "$output" > "$output.sha256.local"
echo "LibriSpeech one-hour wake soak corpus written to $output"
