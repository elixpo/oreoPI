# ADR 0001: Runtime and language boundaries

Status: accepted for WP-001

## Decision

Use Rust for the long-running runtime, state machine, routing, permissions,
device security, event contracts, and CLI. Audio and ML engines communicate
through bounded adapters. Python is permitted only in an isolated process when
a selected engine lacks a suitable native interface.

The first implementation is synchronous and deterministic. Concurrency enters
behind bounded channels only when a work package demonstrates that it is
required. Network/provider code can never occupy the real-time audio path.

## Consequences

- Core behaviour is testable without microphones, models, credentials, or a
  network connection.
- SBC resource ownership stays visible.
- A crashing optional ML process cannot corrupt policy or device state.
- Cross-process audio adds complexity and must be justified by benchmarks.

