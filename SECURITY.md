# Security policy

## Reporting

Do not open a public issue for a suspected vulnerability involving credentials,
device identity, remote control, pairing, permissions, memory disclosure, audio
privacy, or update verification. Report it privately through GitHub Security
Advisories for `elixpo/oreoPI`, or contact `hello@elixpo.com` when advisories
are unavailable.

Include the affected commit, reproduction steps, impact, and any suggested
mitigation. Do not include real credentials, private recordings, or personal
data.

## Supported versions

The project is pre-release. Only the latest commit on `main` receives security
fixes until versioned releases begin.

## Security boundaries

- Models cannot grant permissions or directly control hardware.
- Raw microphone audio and credentials are not persisted by default.
- Local commands remain available when network services fail.
- Remote device control requires authenticated, scoped, replay-protected
  messages and explicit confirmation for sensitive actions.
