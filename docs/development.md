# Development workflow

GitHub issues are the source of truth for scope and acceptance criteria. Work
on one ordered work package at a time.

## Package cycle

1. Mark the issue as started and restate any narrowed scope.
2. Inspect existing repository and upstream components before designing new
   code.
3. Implement the smallest contract-complete change.
4. Run formatting, Clippy, workspace tests, and package-specific acceptance
   checks.
5. Update the issue with evidence and remaining limitations.
6. Commit the completed package independently with its issue number.
7. Close the issue only when every acceptance criterion is satisfied.

Do not combine unrelated cleanup or a later work package into the current
commit.

## Structure

- `apps/` contains deployable entry points.
- `crates/` contains runtime-owned contracts and implementations.
- `vendor/` contains pinned, reproducible upstream source imports.
- `tests/` contains cross-crate fixtures and behavioural scenarios.
- `docs/decisions/` records decisions that constrain later work.
- `config/` contains non-secret checked-in profiles.

Integrations depend inward on small contracts. Core policy cannot depend on a
specific provider, model, website, operating system audio server, or SBC.

## Open-source reuse

Before implementing a subsystem, evaluate maintained open-source candidates
against functionality, measured resource use, ARM64 support, cancellation,
offline operation, security history, release cadence, licence compatibility,
and reproducible packaging.

Every reused component must be recorded in `docs/third-party.md` with its
upstream URL, pinned version or commit, licence, purpose, local modifications,
and update procedure. Copy only the required surface. Never copy credentials,
runtime state, generated output, or unrelated application code.

Prefer adapting a sound existing component over rewriting it. Keep it behind a
runtime-owned interface so it remains replaceable and testable with a fake.

