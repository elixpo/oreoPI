# ADR 0004: Local state is authoritative on the device

Status: accepted for WP-003

## Decision

Use embedded SQLite as the authoritative store for each Oreo device. Timers,
runtime metadata, local configuration, and future offline queues must continue
to work without a network connection.

Cloudflare D1 may later store website accounts, device-registration metadata,
pairing records, and bounded synchronization state. Device code will reach D1
through an authenticated Worker API; it will not treat D1 as a remote drop-in
replacement for the local database.

Credentials remain in an operating-system credential store. Raw audio,
prompts, model responses, and private tool payloads are excluded from both the
local operational database and default cloud synchronization.

## Consequences

- Alarms and local commands remain available during network or cloud failure.
- The device database has explicit migrations and bounded tables.
- Cloud synchronization is a separate protocol with an outbox and conflict
  policy, rather than shared SQL access.
- Laptop, native SBC, and container deployments mount the same persistent
  state directory and exercise the same storage implementation.
