# Agent cost and quality evaluation

Oreo sends deterministic device commands to local code whenever possible. A
cloud model is used only when a request needs conversational interpretation or
reasoning. The spoken-agent profile limits one turn to 256 output tokens, four
model rounds, four tool calls, four retained history turns, 8 KiB of total
harness output, and 45 seconds.

`elixpo ask --metrics` and `elixpo voice --metrics` emit a single
`agent_metrics` JSON record to standard error. It contains model identity,
token counts, rounds, tool calls, first-text and total latency, and response
byte length. It never contains the prompt, response, transcript, tool payload,
session identifier, or credential.

## Candidate benchmark

The benchmark invokes the real Rust harness and native `device_status` tool.
It fetches current per-token Pollen pricing from the public catalog, checks
bounded response and tool behavior, and writes only fixture IDs and numeric
results under the ignored `target/` directory.

Build once, load the local secret environment, run the dependency-free
self-test, then execute the paid comparison:

```bash
rtk cargo build -p elixpo-cli
set -a
source .env.local
set +a
rtk .venv/bin/python scripts/benchmark-agent.py self-test
rtk .venv/bin/python scripts/benchmark-agent.py run \
  --model openai/gpt-5.4-nano \
  --model qwen/qwen3.7-flash \
  --model nvidia/nemotron-3.5-lightning \
  --repetitions 3 \
  --output target/agent-bench/report.json
```

A candidate is eligible only when every repeated fixture passes. Among
eligible candidates, prefer the lowest mean Pollen cost, then lower p95 total
latency. Catalog health is useful availability evidence but does not replace
Oreo's tool-selection, refusal, ambiguity, and concision fixtures.

Model fallback is allowed only before a side-effecting tool executes. Replaying
an entire failed turn after a device action could duplicate the action, so
cross-model fallback is not enabled by this benchmark.

## Laptop selection

GPT-5.4 Nano passed 15/15 corrected fixtures at 0.0000686025 mean Pollen and
5,709 ms p95. Qwen 3.7 Flash also passed 15/15 but cost 0.000186458 mean Pollen
and reached 46,971 ms p95 because some simple turns billed thousands of output
or reasoning tokens. Nemotron passed only 9/15. GPT-5.4 Nano is therefore the
pinned default; see ADR 0008.
