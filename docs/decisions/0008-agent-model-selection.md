# ADR 0008: GPT-5.4 Nano is the default voice-agent model

Status: accepted for the laptop profile

Date: 2026-10-04

## Decision

Use the canonical `openai/gpt-5.4-nano` model ID as Oreo's default remote
agent model. `OREO_MODEL` remains an explicit override for evaluation and
future migrations. Deterministic local commands continue to bypass the model.

## Evidence

The real Rust harness ran five fixed behavioral fixtures three times per model.
The fixtures cover concise conversation, correct device-status tool use,
avoiding an unnecessary tool, asking for clarification, and refusing credential
disclosure. Reports contain only fixture identifiers and numeric metrics.

| Model | Passed | Mean Pollen | p95 total latency |
|---|---:|---:|---:|
| `openai/gpt-5.4-nano` | 15/15 | 0.0000686025 | 5,709 ms |
| `qwen/qwen3.7-flash` | 15/15 | 0.000186458 | 46,971 ms |
| `nvidia/nemotron-3.5-lightning` | 9/15 | 0.00005191 | 7,036 ms |

Qwen's lower catalog rate did not translate to lower turn cost: some simple
requests reported thousands of billed output/reasoning tokens. It cost about
2.7 times as much per evaluated turn and had about 8.2 times Nano's p95 latency.
Nemotron was cheaper but failed the ambiguity and credential-refusal gates.

## Consequences

- A developer needs only `POLLINATIONS_API_KEY` for the selected default.
- Benchmarks set `OREO_MODEL` to compare exact alternatives.
- The 256-token voice cap, four model rounds, and four tool calls remain hard
  client-side limits.
- Cross-model fallback is not enabled after side-effecting tool execution.
- Pricing and model behavior are re-evaluated before changing the default.
