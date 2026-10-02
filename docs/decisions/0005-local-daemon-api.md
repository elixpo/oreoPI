# ADR 0005: A local daemon owns device runtime state

Status: accepted for WP-003

## Decision

Run one Oreo daemon per user and make it the sole long-lived owner of the local
SQLite store and timer scheduler. Local clients communicate with it through a
versioned JSON protocol over a Unix-domain socket inside the private device
state directory.

The protocol accepts one bounded request per connection, applies read and write
timeouts, and limits timer duration. The state directory is mode `0700`; the
database and socket are mode `0600`. The daemon does not expose a TCP or LAN
listener. A condition-variable scheduler sleeps until the next deadline or a
state change instead of polling.

## Consequences

- The CLI, future audio pipeline, and future hardware adapters share one state
  authority instead of opening SQLite independently.
- Persistent timers survive client exits and daemon restarts.
- Laptop and SBC deployments use the same local API and scheduler behavior.
- Remote website control remains a separate authenticated control-plane design;
  it cannot connect directly to the local socket or database.
- Timer expiry is recorded as a bounded runtime event. Audible delivery and
  subscriber notifications are added as a later output layer.
