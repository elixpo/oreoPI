# Elixpo Voice / Oreo

Oreo is a local-first, open-source ambient voice assistant for the Elixpo
platform. Development happens on a laptop first and is continuously checked
against SBC-class resource budgets before the unchanged runtime moves to Radxa
or Raspberry Pi hardware.

The GitHub issue tracker is the source of truth:

- [delivery roadmap](https://github.com/Circuit-Overtime/oreoPI/issues/1)
- [current work package](https://github.com/Circuit-Overtime/oreoPI/issues/2)

## Current state

The workspace contains the deterministic reference path and the vendored,
Oreo-owned agent boundary required by WP-001 and WP-002:

```text
CLI input -> local router or Oreo agent -> response plan -> output sink
```

The microphone and hardware paths are not connected yet. The CLI keeps an
explicit offline path and enables the live Pollinations provider only when its
credential and exact model are supplied through the process environment.

## Build and test

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p oreo-daemon
cargo run -p elixpo-cli -- status
cargo run -p elixpo-cli -- tools
cargo run -p elixpo-cli -- timer set tea 300
cargo run -p elixpo-cli -- timer list
cargo run -p elixpo-cli -- timer cancel tea
cargo run -p elixpo-cli -- daemon stop
cargo run -p elixpo-cli -- ask --offline "What is running?"
POLLINATIONS_API_KEY=... OREO_MODEL=... cargo run -p elixpo-cli -- ask "Hello Oreo"
```

`OREO_STATE_DIR` can override the local state location. Session journals store
only bounded metadata and digests; prompts, response text, credentials, and
tool payloads are not persisted.

## Invariants

- Normal deterministic commands never require AI.
- The audio loop will never wait on the model, website, or a network tool.
- Models cannot grant permissions or directly control hardware.
- Raw audio and credentials are never persisted in memory or logs.
- The local runtime remains useful when every network feature fails.
- Affect changes presentation, never truth, safety, or permissions.
- Queues, model output, sessions, and tool output are bounded.

## Licence

Source code is provided under the Elixpo licensing standard: MIT with the
Oreo/Elixpo trademark exception. Brand and visual assets use CC-BY-4.0 with the
same exception. See [`LICENSE`](LICENSE) and [`LICENSES/NOTICE`](LICENSES/NOTICE).
