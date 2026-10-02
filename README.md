# Elixpo Voice / Oreo

Oreo is a local-first, open-source ambient voice assistant for the Elixpo
platform. Development happens on a laptop first and is continuously checked
against SBC-class resource budgets before the unchanged runtime moves to Radxa
or Raspberry Pi hardware.

The GitHub issue tracker is the source of truth:

- [delivery roadmap](https://github.com/Circuit-Overtime/oreoPI/issues/1)
- [current work package](https://github.com/Circuit-Overtime/oreoPI/issues/2)

## Current state

The workspace contains the deterministic reference path required by WP-001:

```text
CLI input -> local router or fake harness -> response plan -> output sink
```

It intentionally has no microphone, model, network, account, or hardware
dependency yet.

## Build and test

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p elixpo-cli -- status
cargo run -p elixpo-cli -- ask "What is running?"
```

## Invariants

- Normal deterministic commands never require AI.
- The audio loop will never wait on the model, website, or a network tool.
- Models cannot grant permissions or directly control hardware.
- Raw audio and credentials are never persisted in memory or logs.
- The local runtime remains useful when every network feature fails.
- Affect changes presentation, never truth, safety, or permissions.
- Queues, model output, sessions, and tool output are bounded.

