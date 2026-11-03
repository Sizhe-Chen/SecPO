# AgentDojo evaluation

AgentDojo is an external, pinned benchmark dependency rather than vendored
SecPO code. The release performs three safeguards before importing it:

1. `HEAD` must equal the configured immutable commit.
2. the upstream MIT license must match its audited checksum;
3. the checkout must be pristine or contain exactly the optional one-file
   resumption patch, pinned by SHA-256. A separately materialized expected
   tree is compared file-for-file, so untracked or additional edits inside an
   already-patched file are rejected.

The local victim adapter uses an OpenAI-compatible endpoint, native tool
schemas, structured tool calls, and separate Qwen reasoning content. API
credentials are referenced only by the `AGENTDOJO_API_KEY` environment
variable. The endpoint is resolved only at runtime from
`AGENTDOJO_BASE_URL`; there is deliberately no default. The config stores no
key, hostname, or institution-specific endpoint.

## Setup

Clone and pin AgentDojo as described in
[`third_party/README.md`](../third_party/README.md), install that checkout in a
separate environment, and serve the pinned Qwen revision with a tool-call
parser and chat template appropriate for Qwen3.6. Model weights and vLLM are
not bundled here.

Verify the checkout before spending GPU time:

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py verify \
  --checkout /path/to/agentdojo
```

Set the endpoint only in the process environment. For example, point
`AGENTDOJO_BASE_URL` at the OpenAI-compatible `/v1` root exposed by your local
server. Do not commit the resolved value, endpoint credentials, or private
hostnames.

## Utility and static attack

Utility runs all four suites without injection. Static evaluation uses
AgentDojo's `important_instructions` attack. Both modes use
`repeat_user_prompt`, benchmark version `v1.2.1`, and the exact upstream task
checkers.

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py utility \
  --checkout /path/to/agentdojo \
  --output results/agentdojo_utility.json

python scripts/evaluate_agentdojo_qwen36_27b.py static \
  --checkout /path/to/agentdojo \
  --output results/agentdojo_static.json
```

For a smoke run, select one suite and task:

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py static \
  --checkout /path/to/agentdojo \
  --suite slack \
  --user-task user_task_0 \
  --injection-task injection_task_1 \
  --output results/agentdojo_smoke.json
```

`security_results` in the upstream API means that the injection task was
executed successfully. The released summary therefore calls it
`exact_attack_success` and reports ASR directly, avoiding the ambiguous name.

## Adaptive Genetic search

The release expands the exact 80-pair Slack/Travel/Workspace subset stored in
the JSON config: 20 Slack, 12 Travel, and 48 Workspace pairs. Banking is not
part of that adaptive subset. Search uses the shared SecPO island
Genetic/MAP-Elites implementation with 100 generations, eight candidates per
generation, and an 800-victim-call Pass@800 cap for each pair.

For the full benchmark, one local attacker callable is required:

- `proposer(request) -> Sequence[str]` follows the interface documented by
  `secpo.attacks.AttackRequest`.
- an optional `dense_scorer(task, injection, transcript) -> float` may rank
  failed candidates from 0 through 9. It cannot declare success.

Run or resume all 80 pairs with:

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py adaptive \
  --checkout /path/to/agentdojo \
  --proposer local_attack:propose \
  --dense-scorer local_attack:score_failed_trajectory \
  --output-dir results/agentdojo_adaptive \
  --resume
```

Omit `--dense-scorer` to use zero dense fitness, and omit `--resume` for a new
run. Repeat `--task-index` only for an intentional subset smoke test; the
summary marks partial coverage and cannot be reported as paper-80 ASR.

The concrete episode runner derives every injection-task goal from the verified
checkout, creates a fresh AgentDojo environment for every candidate, inserts
the candidate into every task-selected vector, executes the native tool loop,
and calls AgentDojo's post-state/trajectory checker. That exact checker is the
sole success oracle. A trajectory model may supply dense fitness only for a
failure; failures are capped at 9 and exact success is scored as 10.

Each pair has atomic per-step state and an atomic completed result. Resume
rejects incompatible configurations, and aggregate output fails closed on a
duplicate or missing member of an intended full-80 run. The summary reports
exact-checker ASR, victim calls, and best-candidate utility without treating a
dense score as success.

The legacy `genetic` subcommand remains as a one-pair callback harness. It is
useful for integration tests but is not the recommended paper-80 entry point.

## Low-cost AttackerMovesSecond (Section 4.6)

The released [AttackerMovesSecond search](../src/secpo/attacks/agentdojo_exact_first.py)
uses **MiniMax-M3.1-Flash with high reasoning** as the attacker and
**DeepSeek-V4.1-Flash-1017 with reasoning disabled** as the critic. The
80-case ASR@800 evaluation costs **less than $200** in the reported setup,
including the worst case in which every case exhausts its budget without a
successful attack. Actual charges depend on provider pricing and usage;
the runner does not enforce a dollar cap.

The main AgentDojo results in [Performance](PERFORMANCE.md) use the original **Gemini-2.5-Pro** attack
configuration. The cheaper profile is a separate configuration, so its cost
should not be combined with the main result as if both came from one run.

Implementation entry points:

- [Exact-first search and candidate checkpoints](../src/secpo/attacks/agentdojo_exact_first.py)
- [Portable 80-pair orchestration and resume](../src/secpo/evaluation/agentdojo_search.py)
- [Attacker, critic, and victim configuration](../configs/evaluate_agentdojo_search_qwen36_secpo.json)
- [Evaluation CLI](../scripts/evaluate_agentdojo_qwen36_27b.py)

After installing the pinned AgentDojo checkout, serving the victim model, and
setting `AGENTDOJO_BASE_URL`, optional `AGENTDOJO_API_KEY`,
`OPENROUTER_BASE_URL`, and `OPENROUTER_API_KEY`, run:

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py \
  --config configs/evaluate_agentdojo_search_qwen36_secpo.json \
  adaptive \
  --checkout /path/to/agentdojo \
  --backend-config configs/evaluate_agentdojo_search_qwen36_secpo.json \
  --search-only \
  --output-dir results/agentdojo_search \
  --resume
```

The default budget is 100 generations of eight candidates per task pair.
AgentDojo's exact checker determines success, and search stops at the first
successful candidate. Failed generations receive critic scores after their
exact checks. Completed candidates and scores are checkpointed for resume.
`--search-only` reports search success without an independent final replay;
ordinary `adaptive` evaluation retains the final replay.
