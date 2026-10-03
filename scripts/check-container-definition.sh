#!/bin/sh
set -eu

fail() {
    echo "container definition check failed: $1" >&2
    exit 1
}

require_fixed() {
    grep -Fq "$2" "$1" || fail "$1 is missing: $2"
}

require_fixed Dockerfile 'FROM rust:1.95-slim-bookworm AS builder'
require_fixed Dockerfile 'cargo build --locked --release -p oreo-daemon -p elixpo-cli'
require_fixed Dockerfile 'libasound2-dev pkg-config'
require_fixed Dockerfile 'ca-certificates libasound2'
require_fixed Dockerfile 'USER 10001:10001'
require_fixed Dockerfile 'CMD ["elixpo", "diagnostics"]'
require_fixed Dockerfile 'ENTRYPOINT ["/usr/local/bin/oreo-entrypoint"]'
require_fixed compose.yaml 'read_only: true'
require_fixed compose.yaml 'no-new-privileges:true'
require_fixed compose.yaml 'oreo-state:/var/lib/oreo'
require_fixed .dockerignore '**'
require_fixed .dockerignore '!apps/**'
require_fixed .dockerignore '!crates/**'
require_fixed deploy/container-entrypoint.sh 'elixpo daemon stop'

if grep -Eq '^!(\.env|target/|\.git/)' .dockerignore; then
    fail '.dockerignore must not re-include secrets, build output, or Git metadata'
fi

echo 'container definition check passed'
