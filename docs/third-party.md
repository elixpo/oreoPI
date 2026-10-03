# Third-party component register

This register is updated before a third-party component enters the build or a
vendored source tree.

| Component | Upstream | Pin | Licence | Purpose | Local changes | Update path |
|---|---|---|---|---|---|---|
| Crumb native harness fork | [elixpo/crumb.elixpo](https://github.com/elixpo/crumb.elixpo) | Fork point `5e5b517ac5fb5ce9360b164db2fc9b62247f2fda` | MIT / Elixpo notices | Starting point for the provider-neutral agent loop, tool policy, model contracts, and Pollinations adapter | Maintained as Oreo-specific first-party crates under `crates/`; see `docs/upstream/crumb-harness.md` | Review and merge useful upstream patches selectively; never overwrite the Oreo fork wholesale |
| rusqlite / SQLite | [rusqlite](https://github.com/rusqlite/rusqlite) / [SQLite](https://sqlite.org) | `rusqlite 0.40.x`, locked in `Cargo.lock`; bundled SQLite | MIT (`rusqlite`); public domain (`SQLite`) | Local migrations, bounded runtime metadata, and restart-safe timers | None | Update through Cargo, review upstream changes, then run migration and persistence tests |
| CPAL | [RustAudio/cpal](https://github.com/RustAudio/cpal) | `0.18.2`, locked in `Cargo.lock` | Apache-2.0 / MIT | Cross-platform laptop and SBC microphone/speaker access; default Linux build uses ALSA | Oreo adapter adds bounded preallocated queues, cancellation, redacted errors, and overflow/underrun counters | Update through Cargo; run callback tests, live capture/playback diagnostics, and the audio soak before release |
