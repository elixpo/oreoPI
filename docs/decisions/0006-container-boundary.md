# ADR 0006: The container preserves the local device boundary

Status: accepted for deployment checkpoint #20

## Decision

Package `oreo-daemon` and `elixpo` together in one Linux image built for AMD64
and ARM64. Run as an unprivileged numeric user with a read-only root filesystem,
all capabilities dropped, and no published network port. Persist only
`OREO_STATE_DIR` on a named volume.

Container health uses the same private Unix-socket diagnostics request as local
clients. Graceful container termination asks the daemon to shut down through
that API before falling back to a process signal. Credentials are supplied at
invocation time and are never copied into an image layer or Compose file.

## Consequences

- Laptop and SBC containers exercise the same binaries, schema, and local API.
- SQLite, timers, and redacted session metadata survive image replacement.
- Containerization does not turn the local API into a remote control endpoint.
- Website pairing still requires a separately designed authenticated outbound
  control plane.
- Audio-device mapping is intentionally deferred until the audio work package
  defines its Linux device and permission requirements.
