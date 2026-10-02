# ADR 0002: Versioned contracts before integrations

Status: accepted for WP-001

## Decision

All harnesses, audio engines, tools, outputs, and hardware backends implement
small runtime-owned contracts. Integrations do not exchange untyped maps inside
the core. Inputs, responses, events, limits, cancellation, and failures are
explicit.

The fake harness and output sink are reference implementations and remain in
the test surface as real integrations are added.

## Consequences

- Laptop and SBC implementations share behaviour.
- The Crumb-derived engine remains behind the narrow `oreo-agent` adapter.
- Deterministic tests remain available when live services are unavailable.
