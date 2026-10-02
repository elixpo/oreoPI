# Crumb harness fork point

OreoPI's initial native agent engine was imported from Crumb and then promoted
into first-party workspace crates so it can evolve for an ambient device rather
than a terminal coding agent.

- Upstream: <https://github.com/elixpo/crumb.elixpo>
- Commit: `5e5b517ac5fb5ce9360b164db2fc9b62247f2fda`
- Work item: <https://github.com/elixpo/crumb.elixpo/issues/70>
- Imported: 2026-10-02
- Included crates: `crumb-agent`, `crumb-harness`, `crumb-llm`, and
  `crumb-pollinations`
- Local crates: `crates/crumb-agent`, `crates/crumb-harness`,
  `crates/crumb-llm`, and `crates/crumb-pollinations`
- Licence: MIT under the Elixpo licensing standard; upstream attribution is
  retained in `docs/third-party.md`

Commit `5e5b517ac5fb5ce9360b164db2fc9b62247f2fda` is the immutable fork point.
Changes after that commit are ordinary OreoPI development and do not need to
remain byte-identical to Crumb.

Useful general-purpose fixes may still be contributed upstream first. Bringing
later Crumb work into OreoPI requires a reviewed patch or merge, an update to
the third-party register, and the normal OreoPI test gates; source directories
must not be overwritten wholesale.
