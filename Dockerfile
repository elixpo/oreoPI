# syntax=docker/dockerfile:1.7

FROM rust:1.95-slim-bookworm AS builder

WORKDIR /src
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libasound2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY apps ./apps
COPY crates ./crates
COPY config ./config

RUN cargo build --locked --release -p oreo-daemon -p elixpo-cli

FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="Oreo local assistant"
LABEL org.opencontainers.image.source="https://github.com/elixpo/oreoPI"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libasound2 \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -m 0700 -o 10001 -g 10001 /var/lib/oreo

COPY --from=builder /src/target/release/oreo-daemon /usr/local/bin/oreo-daemon
COPY --from=builder /src/target/release/elixpo /usr/local/bin/elixpo
COPY --chmod=0555 deploy/container-entrypoint.sh /usr/local/bin/oreo-entrypoint
COPY LICENSE LICENSES/NOTICE /usr/share/licenses/oreopi/

ENV HOME=/var/lib/oreo
ENV OREO_STATE_DIR=/var/lib/oreo

USER 10001:10001
VOLUME ["/var/lib/oreo"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["elixpo", "diagnostics"]

ENTRYPOINT ["/usr/local/bin/oreo-entrypoint"]
