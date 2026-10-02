# ADR 0003: Oreo owns the agent boundary

Status: accepted for WP-002

## Decision

Use the selected Crumb crates as the documented starting point for first-party
OreoPI engine crates under `crates/`. Applications and device services depend
on `oreo-agent`, not directly on the Crumb-derived implementation crates.

`oreo-agent` owns the Oreo profile, session lifecycle, transient event
vocabulary, redacted errors, and the public provider/tool extension points.
The promoted engine crates own the bounded model/tool loop, typed calls,
approvals, cancellation, deadlines, and secret-safe session metadata.

## Consequences

- Oreo-specific behavior evolves in one repository and one workspace.
- The exact Crumb fork point remains recorded for attribution and auditing.
- Later upstream changes are reviewed and merged selectively rather than
  replacing Oreo's implementation wholesale.
- Audio, emotion, CLI, and device layers share one narrow agent API.
- Credentials remain construction-time inputs to providers and are not part of
  serializable or debuggable Oreo profile state.
