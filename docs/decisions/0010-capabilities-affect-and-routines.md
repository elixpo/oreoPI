# ADR 0010: Extensions remain behind typed local policy

Status: accepted

## Decision

Oreo uses one capability boundary for native functions, cloud connectors, and
device hardware. Every function has a JSON schema, risk class, location,
confirmation policy, timeout, cancellation contract, and disclosure. The model
sees descriptors and arguments; it never receives connector credentials,
driver handles, GPIO access, or the approval interface.

Cloud connector registration requires a cloud location, a network or
credential-sensitive risk class, and `works_offline = false`. Hardware
registration requires an explicit device-hardware location. Existing policy
then applies confirmation and connectivity rules before the native handler can
run.

Routines are bounded data: one identifier and at most 16 capability calls with
object arguments and a 16 KiB aggregate limit. Each step is executed through
the same tool host and approval broker. A routine cannot contain a shell
command or bypass a denied, failed, or cancelled step.

Affect is a small local state with bounded valence, arousal, and confidence.
It produces conservative rate, pitch, and warmth hints for speech backends.
Affect may shape presentation only; it cannot alter facts, capability choice,
permission, confirmation, or safety behavior.

## Consequences

- Adding GitHub, calendar, home automation, GPIO, or sensor adapters does not
  expand model authority.
- The website can become the trusted approval and connector-account surface
  without moving device state or credentials into the model prompt.
- Routines compose existing reviewed capabilities instead of adding a second
  automation execution path.
- Pocket TTS may ignore unsupported affect hints; future backends can consume
  them without changing the agent or safety contracts.
