# Vendored Crumb harness

This directory contains an unchanged source snapshot of the native Crumb
agent harness used by OreoPI.

- Upstream: <https://github.com/elixpo/crumb.elixpo>
- Commit: `19dc32313e57b3cfed82f430f81343cf519ac78f`
- Review: <https://github.com/elixpo/crumb.elixpo/pull/71>
- Imported: 2026-10-02
- Included crates: `crumb-agent`, `crumb-harness`, `crumb-llm`, and
  `crumb-pollinations`
- Licence: MIT; the copied upstream `LICENSE` and `LICENSES` directory are
  retained beside the source

Files below `upstream/` must remain byte-for-byte identical to the pinned
commit. Oreo-specific workspace configuration and integration code live
outside that directory. `MANIFEST.sha256` records every file in the snapshot
and can be checked from this directory with:

```sh
sha256sum --check MANIFEST.sha256
```

To update the snapshot, first land and validate changes upstream, replace only
the selected files from one commit, update this record and the third-party
register, regenerate the manifest, and run the complete OreoPI workspace gates.
