# Local memory

Oreo uses two separate memory layers:

- The harness keeps at most four completed conversation turns in RAM for a
  fluid active conversation. Raw turns are not written to SQLite.
- SQLite stores only explicit, user-managed memory records. `session` records
  have a required expiry of at most 24 hours; `long_term` records remain until
  the user replaces or deletes them.

Both scopes share a 256-record device limit and a 512-byte limit per record.
Identifiers are bounded ASCII names. Multiline values and credential-like
content are rejected. Expired short-term records are deleted during normal
reads and writes.

Manage memory through the local Unix socket while the daemon is running:

```bash
rtk cargo run -p elixpo-cli -- memory remember locale The user prefers British English
rtk cargo run -p elixpo-cli -- memory remember-session current-task 3600 We are choosing flowers
rtk cargo run -p elixpo-cli -- memory durable
rtk cargo run -p elixpo-cli -- memory short
rtk cargo run -p elixpo-cli -- memory forget locale
```

The voice agent receives up to 16 recent records from each scope as clearly
marked, untrusted context when it starts. It also has a read-only
`memory_recall` capability for records changed after startup. The model has no
memory-write or memory-delete capability; only the trusted local API—and later
the authenticated device website—may mutate these records.

The older `memory list` and `memory inspect` commands inspect redacted harness
session journals. Those journals contain lifecycle metadata and digests, not
memory content, prompts, or responses.
