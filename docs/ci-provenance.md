# Elixpo CI provenance

The shared repository automation was imported from `elixpo/crumb.elixpo`,
`elixpo/blogs.elixpo`, and `elixpo/oreoOS` during issue #14.

Imported and adapted:

- cross-platform Rust CI;
- reusable Elixpo quality, PR management, stale, labeling, issue/PR triage,
  repository-agent, acknowledgement, retry, context-artifact, and merge flows;
- issue templates, PR template, Dependabot, label rules, shared helper scripts,
  and repository-agent documentation;
- the OreoOS README-update helper needed by the shared merge workflow;
- Rust-specific release and vulnerability gates.

Deliberately excluded because they are product-specific or need functionality
that OreoPI does not yet ship:

- Blogs deployment, contest, moderation, and Cloudflare workflows;
- Crumb website deployment and payment-catalog synchronization;
- OreoOS firmware release, icon synchronization, and JavaScript CodeQL jobs.

Repository metadata, changed-path rules, prompts, issue templates, and quality
commands were adapted for OreoPI. Organization automation assumes the canonical
repository owner is `elixpo` and uses organization-managed secrets where a
networked agent workflow is explicitly authorized.
