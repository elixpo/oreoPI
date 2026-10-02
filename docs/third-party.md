# Third-party component register

This register is updated before a third-party component enters the build or a
vendored source tree.

| Component | Upstream | Pin | Licence | Purpose | Local changes | Update path |
|---|---|---|---|---|---|---|
| Crumb native harness | [elixpo/crumb.elixpo](https://github.com/elixpo/crumb.elixpo) | `5e5b517ac5fb5ce9360b164db2fc9b62247f2fda` | MIT; upstream notices retained | Provider-neutral bounded agent loop, tool policy, model contracts, and Pollinations adapter | None inside the snapshot; Oreo adapts it in `oreo-agent` | Replace from a reviewed upstream commit, regenerate `MANIFEST.sha256`, then run the workspace gates |
