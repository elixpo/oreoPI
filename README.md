# Elixpo Voice / Oreo

Oreo is a local-first, open-source ambient voice assistant for the Elixpo
platform. Development happens on a laptop first and is continuously checked
against SBC-class resource budgets before the unchanged runtime moves to Radxa
or Raspberry Pi hardware.

The GitHub issue tracker is the source of truth:

- [delivery roadmap](https://github.com/Circuit-Overtime/oreoPI/issues/1)
- [current work package](https://github.com/Circuit-Overtime/oreoPI/issues/2)

## Current state

The workspace contains the deterministic reference path, Oreo-owned agent
boundary, bounded SQLite state, single-owner local daemon, and an optional
offline speech-input path:

```text
CLI input -> local Unix socket -> daemon -> bounded SQLite state
         \-> local router or Oreo agent -> response plan -> output sink
microphone -> bounded PCM conversion -> Vosk -> Oreo agent -> text response
```

The developer voice command uses explicit Enter-to-start/Enter-to-stop capture.
Wake-word detection, TTS, and hardware GPIO are later layers. The CLI keeps an
explicit offline path and enables the live Pollinations provider only when its
credential and exact model are supplied through the process environment.

## Build and test

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p oreo-daemon
cargo run -p elixpo-cli -- status
cargo run -p elixpo-cli -- diagnostics
cargo run -p elixpo-cli -- audio devices
cargo run -p elixpo-cli -- audio capture-test 3
cargo run -p elixpo-cli -- audio playback-test 2
cargo run -p elixpo-cli -- tools
cargo run -p elixpo-cli -- timer set tea 300
cargo run -p elixpo-cli -- timer list
cargo run -p elixpo-cli -- timer cancel tea
cargo run -p elixpo-cli -- memory list
cargo run -p elixpo-cli -- memory inspect <session-id>
cargo run -p elixpo-cli -- daemon stop
cargo run -p elixpo-cli -- ask --offline "What is running?"
POLLINATIONS_API_KEY=... OREO_MODEL=... cargo run -p elixpo-cli -- ask "Hello Oreo"
POLLINATIONS_API_KEY=... OREO_MODEL=... cargo run -p elixpo-cli -- ask --metrics "Hello Oreo"
```

The Vosk-enabled microphone-to-agent acceptance commands and native-library
setup are documented in [the live voice pipeline guide](docs/audio/live-pipeline.md).
The private-safe multi-model cost workflow is documented in
[the agent evaluation guide](docs/agent-cost.md).

The first container checkpoint packages the same daemon and CLI for AMD64 and
ARM64 without exposing a network port. See [container deployment](docs/container.md)
for the non-root Compose workflow and multi-architecture build commands.

`OREO_STATE_DIR` can override the local state location. Session journals store
only bounded metadata and digests; prompts, response text, credentials, and
tool payloads are not persisted.

The daemon emits newline-delimited JSON operational logs to standard error.
Their schema accepts only a timestamp, component, fixed event name, and fixed
outcome; timer names and user or model content are excluded. `elixpo
diagnostics` reports lifecycle state and current resource/bound utilization
without exposing private content.

## Invariants

- Normal deterministic commands never require AI.
- The audio loop will never wait on the model, website, or a network tool.
- Models cannot grant permissions or directly control hardware.
- Raw audio is transient and bounded; it never enters persistent memory,
  session journals, or logs.
- Credentials never enter local state, session journals, or logs.
- The local runtime remains useful when every network feature fails.
- Affect changes presentation, never truth, safety, or permissions.
- Queues, model output, sessions, and tool output are bounded.

## Licence

Source code is provided under the Elixpo licensing standard: MIT with the
Oreo/Elixpo trademark exception. Brand and visual assets use CC-BY-4.0 with the
same exception. See [`LICENSE`](LICENSE) and [`LICENSES/NOTICE`](LICENSES/NOTICE).
