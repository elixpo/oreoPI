# ADR 0003: Oreo owns the agent boundary

Status: accepted for WP-002

## Decision

Keep the selected Crumb crates as an unchanged, checksummed source snapshot in
`vendor/crumb-harness/upstream`. Applications and device services depend on the
`oreo-agent` crate, never directly on that directory.

`oreo-agent` owns the Oreo profile, session lifecycle, transient event
vocabulary, redacted errors, and the public provider/tool extension points.
Crumb owns the bounded model/tool loop, typed calls, approvals, cancellation,
deadlines, and secret-safe session metadata.

## Consequences

- Oreo-specific behavior can evolve without patching vendored source.
- Upstream refreshes remain byte-comparable and auditable.
- Audio, emotion, CLI, and device layers share one narrow agent API.
- Credentials remain construction-time inputs to providers and are not part of
  serializable or debuggable Oreo profile state.
