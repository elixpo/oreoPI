# Third-party component register

This register is updated before a third-party component enters the build or a
vendored source tree.

| Component | Upstream | Pin | Licence | Purpose | Local changes | Update path |
|---|---|---|---|---|---|---|
| Crumb native harness | [elixpo/crumb.elixpo](https://github.com/elixpo/crumb.elixpo) | `19dc32313e57b3cfed82f430f81343cf519ac78f` | MIT; upstream notices retained | Provider-neutral bounded agent loop, tool policy, model contracts, and Pollinations adapter | None inside the snapshot; Oreo exposes a thin re-export in `oreo-core` | Replace from a reviewed upstream commit, regenerate `MANIFEST.sha256`, then run the workspace gates |
