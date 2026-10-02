# Third-party component register

This register is updated before a third-party component enters the build or a
vendored source tree.

| Component | Upstream | Pin | Licence | Purpose | Local changes | Update path |
|---|---|---|---|---|---|---|
| Crumb native harness fork | [elixpo/crumb.elixpo](https://github.com/elixpo/crumb.elixpo) | Fork point `5e5b517ac5fb5ce9360b164db2fc9b62247f2fda` | MIT / Elixpo notices | Starting point for the provider-neutral agent loop, tool policy, model contracts, and Pollinations adapter | Maintained as Oreo-specific first-party crates under `crates/`; see `docs/upstream/crumb-harness.md` | Review and merge useful upstream patches selectively; never overwrite the Oreo fork wholesale |
