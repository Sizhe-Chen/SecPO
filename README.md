# SecPO: Principled Adversarial Training for Prompt Injection Security

[Paper](https://arxiv.org/abs/2610.41998) | [Models and datasets](#released-artifacts) | [Low-cost AttackerMovesSecond](#low-cost-attackermovessecond-section-46)

SecPO combines a clean-input reference model with adversarial training to
improve robustness to prompt injection while preserving task utility.
The repository is maintained by [Sizhe Chen](https://github.com/Sizhe-Chen)
and contains the benign-reference objective, both main training settings,
static and adaptive SEP evaluation, reference-based utility evaluation, and
the reviewed agent-evaluation integrations for:

- Llama-3.1-8B-Instruct with online adaptive training; and
- Qwen3.6-27B with offline non-thinking training on saved strong attacks.

## Main results

All scores are percentages. Higher utility and lower attack success rate
(ASR) are better. The main Qwen3.6-27B results use thinking disabled.
Meta-SecAlign uses DPO without adversarial training. SecPO uses online
adversarial training for Llama and reuses the optimized injections offline
for Qwen.

### Llama-3.1-8B-Instruct

| Metric | Undefended | Meta-SecAlign | SecPO |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 46.5% | 46.7% | 46.3% |
| GPQA Diamond | 23.7% | 24.2% | 24.2% |
| GSM8K | 82.9% | 84.4% | 81.3% |
| Minerva Math | 49.2% | 48.8% | 50.1% |
| AlpacaEval2 raw win rate | 23.7% | 24.0% | 23.7% |
| SEP utility raw win rate | 55.4% | 55.9% | 59.4% |
| SEP static ASR | 98.1% | 1.8% | 1.4% |
| SEP Genetic adaptive ASR@800 | 100.0% | 89.1% | 12.3% |
| SEP TAP adaptive ASR@800 | 100.0% | 84.6% | 12.2% |
| SEP PISmith adaptive ASR@10 | 98.7% | 95.1% | 7.0% |

### Qwen3.6-27B

| Metric | Undefended | Meta-SecAlign | SecPO |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 84.8% | 83.9% | 84.6% |
| GPQA Diamond | 81.3% | 79.3% | 80.8% |
| GSM8K | 97.3% | 97.0% | 96.4% |
| Minerva Math | 96.1% | 94.8% | 95.8% |
| AlpacaEval2 raw win rate | 80.6% | 81.6% | 82.9% |
| SEP utility raw win rate | 93.0% | 93.9% | 94.3% |
| AgentDojo utility | 89.7% | 87.6% | 90.7% |
| AgentDyn utility | 71.7% | 76.7% | 71.7% |
| DTaP utility under attack | 65.8% | 70.0% | 74.5% |
| SEP static ASR | 98.9% | 2.9% | 1.8% |
| SEP Genetic adaptive ASR@800 | 100.0% | 93.7% | 13.3% |
| SEP TAP adaptive ASR@800 | 100.0% | 90.1% | 12.5% |
| SEP PISmith adaptive ASR@10 | 98.1% | 94.0% | 1.8% |
| AgentDojo static ASR | 26.9% | 2.2% | 0.0% |
| AgentDojo Genetic adaptive ASR@800 | 81.3% | 78.8% | 8.8% |
| AgentDyn static ASR | 22.0% | 4.8% | 0.0% |
| DTaP ASR | 50.8% | 45.9% | 7.6% |

Across the three adaptive SEP attacks, mean ASR falls from **89.6% to
10.5%** for Llama (**8.5x lower**) and from **92.6% to 9.2%** for Qwen
(**10.1x lower**). Mean utility across the six instruction-following,
science, and math benchmarks is **47.5%** for Llama SecPO versus **46.9%**
undefended, and **89.1%** for Qwen SecPO versus **88.9%** undefended.

SEP static ASR covers 9,160 examples; adaptive SEP uses 1,024 examples, with
800 attempts for Genetic/TAP and 10 for PISmith. AgentDojo adaptive ASR uses
80 task pairs and up to 800 attempts each. DTaP covers 1,838 indirect-injection
cases, and its utility is measured **under attack**. AlpacaEval2 and SEP
utility are raw win rates. A measured 0.0% ASR means no successful attacks
were observed in that evaluation.

AgentDojo and AgentDyn use the `repeat_user_prompt` defense and the
`important_instructions` static attack.

The Qwen model also records **0.0% ASR@1600** for each of the three evaluated
Repeat-After-Me visual injection goals, with **95.2% DocVQA utility**.

## Released artifacts

| Artifact | Hugging Face repository |
| --- | --- |
| Llama training dataset | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset) |
| Qwen training dataset | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset) |
| Llama SecPO model | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) |
| Qwen SecPO model | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| Qwen SecOPD model trained on optimized data | [Sizhe-Chen/Qwen3.6-27B-Thinking-SecOPD](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Thinking-SecOPD) |

The license and use terms on each Hugging Face model or dataset card govern
that artifact. The Apache-2.0 license in this repository applies to
project-owned code and documentation; it does not replace model, dataset, or
third-party terms.

## Authors

[Sizhe Chen](https://openreview.net/profile?id=~Sizhe_Chen1),
[Yibo Peng](https://openreview.net/profile?id=~Yibo_Peng3), Jaewon Chang,
[Chawin Sitawarin](https://openreview.net/profile?id=~Chawin_Sitawarin1), and
[David Wagner](https://openreview.net/profile?id=~David_Wagner3).

Baselines, ablations, GCG, instruction-hierarchy experiments, unrelated model
families, cluster administration, one-time recovery scripts, credentials,
caches, model weights, checkpoints, and raw evaluation outputs are outside
this repository's implementation scope.

## What is executable in this snapshot

| Component | Status | Boundary |
| --- | --- | --- |
| Benign-reference SecPO loss and trainer | Included and unit tested | Requires public base-model weights at runtime |
| Qwen offline preferences | Exact split archive included | Redistribution license and content review remain publication gates |
| Llama online training | Included | Atomic exchange plus source-backed current-policy PAIR backend; asynchronous scheduling, model serving, and weights remain runtime boundaries |
| Six static SEP attacks | Included | Victim generator is an injected local callable |
| Two Basic Adaptive delimiter attacks | Included for both model families | Frozen delimiters; regeneration script is not included |
| Genetic/MAP-Elites, PAIR, and TAP search | Clean provider-neutral implementations included | Attacker, victim, and judge models are injected callables |
| Utility evaluation | Included for four lm-eval suites plus AlpacaEval2 and SEP Utility | Exact external GPT-4o/Llama-3-8B references are checksum-locked |
| AgentDojo | Pinned external integration included | Exact checker, utility/static runners, resumable 80-pair Genetic evaluation, low-cost exact-first search, verified MIT pin and optional patch |
| AgentDyn | Pinned external integration included | Qwen native-tool adapter preserves trusted `user` and untrusted `input` roles |
| DTap | Clean runnable Qwen-Agent integration included | Final trusted-`user`/untrusted-`input` adapter, exact 1,838-task identity, external DTap lifecycle/judge backend, portable 33-by-6 execution, resume, and summaries; environments, service, and weights stay external |
| Qwen LoRA merge | Included and unit tested | Produces a standalone checkpoint and path-sanitized manifest; model weights are never bundled |

These boundaries are intentional and machine-readable in
`release_manifest.json`. The active release/backup profile can be narrowed or
expanded later without copying the research workspace wholesale; see
`docs/RELEASE_SCOPE.md`. The original-to-clean file lineage and public naming
changes are recorded in `docs/SOURCE_TO_RELEASE_MAP.md`.

## SecPO objective

For preferred response `chosen`, dispreferred response `rejected`, attacked
prompt `x_adv`, and matching injection-free prompt `x_clean`, SecPO minimizes:

```text
-log sigmoid(beta * (
    [log pi(chosen | x_adv)   - log pi(rejected | x_adv)]
  - [log ref(chosen | x_clean) - log ref(rejected | x_clean)]
))
```

`pi` is the trainable LoRA policy. `ref` is the same base model with its LoRA
adapter disabled. Unlike ordinary DPO, policy probabilities are evaluated on
the attacked prompt while reference probabilities are evaluated on the clean
prompt.

## Repository layout

```text
configs/                 Main training and evaluation settings
data/                    Audited dataset archive, card, and small fixture
docs/                    Scope, evaluation protocol, and provenance notes
examples/                Non-scientific local orchestration example
scripts/                 Narrow training, evaluation, data, and audit CLIs
src/secpo/attacks/       Provider-neutral Genetic, PAIR, and TAP search
src/secpo/evaluation/    SEP, utility, AgentDojo, AgentDyn, and DTap adapters
src/secpo/               Data contract, objective, trainer, online, and merge code
templates/               Reviewed Qwen trust-role chat templates
tests/                   Model-free unit and integration tests
release_manifest.json    Editable deny-first release/backup source of truth
```

## Installation

Python 3.11 or newer is required. The pinned training environment is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For only dependency-light attack construction and release validation, an
editable installation can be used without loading model weights. Optional
extras are declared in `pyproject.toml`:

```bash
pip install -e '.[distributed,test,evaluation]'
```

Keep `model_name_or_path` as a public model ID. A local cache path can be
serialized into PEFT metadata and must not enter a released checkpoint.

## Preference data

The exact Qwen offline dataset is stored as five deterministic gzip parts. It
contains 19,157 records and reconstructs to SHA-256
`cd349baa406eae0bf7e4ee2d938d087d6656469d88076d68ba76baeb7e5ec669`.

```bash
python scripts/reconstruct_dataset.py \
  --manifest data/qwen36_offline_dataset/manifest.json \
  --output /tmp/qwen36_offline_preferences.raw.json
python scripts/audit_release_dataset.py \
  /tmp/qwen36_offline_preferences.raw.json \
  --schema raw --expected-records 19157
python scripts/prepare_preferences.py \
  --input /tmp/qwen36_offline_preferences.raw.json \
  --output /tmp/qwen36_offline_preferences.jsonl.gz \
  --expected-records 19157
```

The public trainer schema is:

```json
{
  "id": "stable-example-id",
  "prompt": "rendered attacked chat prompt",
  "clean_prompt": "the same prompt with one contiguous injection removed",
  "chosen": "response to the trusted instruction",
  "rejected": "response following the injection",
  "attack_goal": "optional online-worker goal",
  "witness": "optional exact attack witness"
}
```

`clean_prompt` is preferred. Legacy records may instead provide
`rejected_input_whole`; preprocessing removes exactly one occurrence and
fails if the match is missing or ambiguous. Read `data/DATASET_CARD.md` before
redistributing the archive.

## Training

### Qwen3.6-27B offline

```bash
DATA_FILE=/tmp/qwen36_offline_preferences.jsonl.gz \
  bash scripts/train_qwen36_27b_offline.sh
```

The main configuration uses one epoch, sigmoid beta 0.1, learning rate
1.6e-4, LoRA rank 64/alpha 8/dropout 0.1, bf16, eight processes, and ZeRO-3.

### Llama-3.1-8B online

Prepare the raw 19,157-row Llama preference file with the explicit online
contract. This maps legacy `rejected_input` to a normalized `attack_goal` and
derives deterministic content-hash IDs when the raw rows have no `id`:

```bash
python scripts/prepare_preferences.py \
  --input /path/to/llama31_raw_preferences.json \
  --output /tmp/llama31_online_preferences.jsonl \
  --expected-records 19157 \
  --online-pair
```

The attacker and defender use a private shared runtime directory. Start eight
current-policy PAIR workers on the attack GPU group, one process per visible
GPU and rank. Each worker receives caller-owned provider routing callables:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_llama31_current_policy_pair.py \
  --config configs/llama31_current_policy_pair.json \
  --exchange-dir runtime/online_exchange \
  --policy-root outputs/llama31_8b_online \
  --rank 0 \
  --proposer your_provider.pair:propose \
  --iteration-judge your_provider.pair:score_iteration \
  --final-judge your_provider.pair:judge_final
```

Repeat for ranks 1 through 7, then start the defender on its eight training
GPUs:

```bash
DATA_FILE=/tmp/llama31_online_preferences.jsonl \
ONLINE_EXCHANGE_DIR=runtime/online_exchange \
  bash scripts/train_llama31_8b_online.sh
```

Every online record must have a stable `id` and attack goal. The released
backend preserves the main run's 8 streams, 30 rounds, keep-last-3 history,
1--10 progress scoring, deterministic current-policy victim generation, and
unanimous 3-by-3 final check. Before every victim round, it re-detects the
newest recognizable LoRA checkpoint and waits for that checkpoint to finish
publishing; it never serves an older complete checkpoint while a newer named
one is incomplete. The trainer publishes a standard PEFT checkpoint after
every optimizer/gradient-accumulation step (`save_steps: 1`), so the next
attack group observes the updated policy.

Search and final-verification candidates are source-faithfully appended after
the parsed clean input, regardless of whether the dataset seed injection was
at the front or end. Only the optimized prompt returned to DPO restores the
original seed slot. The Llama config uses sampler `data_seed: 0`, drops an
incomplete microbatch, and caps the full 19,157-row/eight-rank run at 74
complete optimizer updates; release RNG seed 42 is a deterministic replacement
because the original training seed was not retained.

The default connected worker accepts and blocks on the trainer's two-example
microbatches. The defender therefore cannot update during one request's
30-round search, unlike the archived asynchronous producer, so this changes
attack trajectories rather than only call batching. The separate
`run_llama31_current_policy_pair_exact_batch.py` entry point enforces 32
examples per rank but still requires an external accumulation-group producer
that pipelines the next group concurrently; that producer is not included.
These scheduling boundaries and the provider callable schemas are documented in
`docs/LLAMA31_CURRENT_POLICY_PAIR.md`.

The exchange uses 0700 directories, 0600 atomic JSON files, schema validation,
one-contiguous-injection validation, bounded polling, and redacted failure
records. Runtime exchange files must never be committed.

For an orchestration-only smoke test, the generic worker can still load
`examples/online_backend.py:reuse_seed_attacks`. That example reuses existing
seed attacks and is not a scientific substitute for the PAIR worker above.

## Static and Basic Adaptive SEP

Prepare portable requests without loading a model:

```bash
python scripts/evaluate_sep_static.py prepare \
  --input /path/to/SEP_dataset_test.json \
  --output /tmp/static_requests.jsonl \
  --model-family llama31_8b
```

Run a local callable victim and optional semantic judge:

```bash
python scripts/evaluate_sep_static.py run \
  --input /path/to/SEP_dataset_test.json \
  --output /tmp/static_predictions.jsonl \
  --model-family qwen36_27b \
  --generator your_package.victim:generate \
  --judge your_package.judge:judge
```

The default registry contains exactly six static attacks plus the two
model-specific completion attacks. There is no dynamic `eval()` dispatch.
Witness misses are always failures; witness hits remain pending unless a judge
callable decides them.

To apply the paper's strict three-judges-by-three-trials protocol to generated
JSONL, supply a provider-neutral routing callable:

```bash
python scripts/judge_sep_outputs.py \
  --input /tmp/static_predictions.jsonl \
  --output /tmp/static_judged.jsonl \
  --config configs/evaluate_sep_judges.json \
  --backend your_package.judges:judge
```

The backend receives the three manuscript routing labels from
`configs/evaluate_sep_judges.json`. All nine valid votes must be `YES`.
Exhausted backend failures produce an explicit unknown result, never a
synthetic negative vote; successful votes resume from a fingerprinted cache.

## Genetic, PAIR, and TAP adaptive SEP

A deterministic dry run verifies installation and checkpoint behavior:

```bash
python scripts/evaluate_sep_adaptive.py \
  --algorithm genetic \
  --output /tmp/adaptive_dry_run.json
```

For a scientific run, supply an importable backend implementing `propose`,
`victim`, and dense `score`, plus a provider-neutral binary judge callable:

```bash
python scripts/evaluate_sep_adaptive.py \
  --config configs/evaluate_sep_adaptive_qwen36_27b.json \
  --input /path/to/SEP_dataset_test.json \
  --output /tmp/adaptive_results.json \
  --backend your_package.backend:Backend \
  --judge-backend your_package.judges:judge \
  --algorithm tap \
  --checkpoint-dir /tmp/adaptive_state
```

Use `configs/evaluate_sep_adaptive_llama31_8b.json` for Llama. Genetic retains
the working island/MAP-Elites implementation. Qwen PAIR and TAP share the
current Llama-derived orchestration while applying Qwen-specific stop tokens,
trusted-`user`/untrusted-`input` rendering, and reasoning removal before every
score or witness check. TAP defaults to depth 10, width 10, branch factor 8;
Genetic and TAP cap all victim calls, including seed evaluation, at 800.

Dense fitness guides search but cannot declare success. Every witness-positive
candidate receives the released three-model-by-three-trial panel, and only
unanimous `9/9 YES` can early-stop or count toward ASR. Malformed/exhausted
votes remain unknown. Candidate votes and search state are checkpointed with a
configuration/example/model-policy fingerprint and reject incompatible resume.

## Utility evaluation

For MMLU-Pro, GPQA Diamond, GSM8K, and MATH, print the exact commands before
executing them:

```bash
python scripts/evaluate_lm_eval_utility.py \
  --config configs/evaluate_lm_eval_qwen36_27b.json \
  --adapter-path /path/to/secpo_adapter \
  --output-dir /tmp/qwen36_utility
```

Add `--execute` after reviewing the printed commands. The child receives no
credential variables by default. For a gated model, pass only an already-set
variable name explicitly; the option is repeatable:

```bash
python scripts/evaluate_lm_eval_utility.py \
  --config configs/evaluate_lm_eval_qwen36_27b.json \
  --adapter-path /path/to/secpo_adapter \
  --output-dir /tmp/qwen36_utility \
  --execute \
  --pass-env HF_TOKEN
```

AlpacaEval2 and SEP Utility use the dedicated reference workflow:

```bash
export SECPO_UTILITY_ANNOTATORS_CONFIG=<reviewed-alpaca-eval-config>
python scripts/evaluate_reference_utility.py \
  --config configs/evaluate_alpacaeval2_utility.json \
  --candidate /path/to/alpacaeval2_candidate_outputs.json \
  --reference /path/to/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json \
  --output-dir /tmp/alpacaeval2_utility

python scripts/evaluate_reference_utility.py \
  --config configs/evaluate_sep_utility.json \
  --candidate /path/to/sep_candidate_outputs.json \
  --reference "/path/to/SEP_dataset_test_Meta-Llama-3-8B-Instruct copy.json" \
  --output-dir /tmp/sep_utility
```

The default mode performs full preflight and prints the external
`alpaca_eval` command; add `--execute` to run it. The exact 805-row GPT-4o and
9,160-row Llama-3-8B comparator files remain external, but their SHA-256,
generator identity, ordering, and metrics are frozen and validated. See
`docs/REFERENCE_UTILITY.md`. Reference evaluation likewise passes no
credential by default; when needed, append a reviewed name such as
`--pass-env EVALUATOR_API_TOKEN` after setting that variable outside the
repository.

## AgentDojo

AgentDojo is not vendored. The integration verifies its immutable commit, MIT
license checksum, and optional one-file patch before importing the benchmark:

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py verify \
  --checkout /path/to/agentdojo
python scripts/evaluate_agentdojo_qwen36_27b.py utility \
  --checkout /path/to/agentdojo \
  --output /tmp/agentdojo_utility.json
python scripts/evaluate_agentdojo_qwen36_27b.py static \
  --checkout /path/to/agentdojo \
  --output /tmp/agentdojo_static.json
python scripts/evaluate_agentdojo_qwen36_27b.py adaptive \
  --checkout /path/to/agentdojo \
  --proposer your_package.agentdojo_attack:propose \
  --output-dir /tmp/agentdojo_adaptive \
  --resume
```

The endpoint and optional credential are runtime environment variables only.
The adaptive runner expands exactly 80 pairs (Slack 20, Travel 12, Workspace
48), starts a fresh environment for every candidate, and uses AgentDojo's task
checker—not a model judge—as the sole success oracle. Atomic per-step/per-pair
state supports whole-80 resume and rejects duplicate or missing coverage.
Setup, native-tool adapter, and Genetic callback contracts are documented in
`docs/AGENTDOJO.md` and `third_party/README.md`.

### Low-cost AttackerMovesSecond (Section 4.6)

The released [AttackerMovesSecond search](src/secpo/attacks/agentdojo_exact_first.py)
uses **MiniMax-M3.1-Flash with high reasoning** as the attacker and
**DeepSeek-V4.1-Flash-1017 with reasoning disabled** as the critic. The
80-case ASR@800 evaluation costs **less than $200** in the reported setup,
including the worst case in which every case exhausts its budget without a
successful attack. Actual charges depend on provider pricing and usage;
the runner does not enforce a dollar cap.

The main AgentDojo results above use the original **Gemini-2.5-Pro** attack
configuration. The cheaper profile is a separate configuration, so its cost
should not be combined with the main result as if both came from one run.

Implementation entry points:

- [Exact-first search and candidate checkpoints](src/secpo/attacks/agentdojo_exact_first.py)
- [Portable 80-pair orchestration and resume](src/secpo/evaluation/agentdojo_search.py)
- [Attacker, critic, and victim configuration](configs/evaluate_agentdojo_search_qwen36_secpo.json)
- [Evaluation CLI](scripts/evaluate_agentdojo_qwen36_27b.py)

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

## Qwen model merge

Merge a trained Qwen LoRA adapter into a standalone checkpoint before an
evaluation that cannot load PEFT adapters directly:

```bash
python scripts/merge_qwen36_27b_lora.py \
  --base /path/to/Qwen3.6-27B \
  --lora /path/to/secpo_adapter \
  --output /path/to/merged_secpo_model
```

The command refuses to overwrite an existing directory, checks that the LoRA
was loaded, preserves multimodal weights when present, verifies every indexed
safetensors shard, requires a usable tokenizer (falling back to the base), and
writes a path-sanitized `merge_manifest.json`. It builds in a temporary sibling
and publishes the directory only after all checks pass. The merged weights
remain a local artifact and must not be committed.

## AgentDyn and DTap

The Qwen AgentDyn adapter and DTap implementation are separate pinned
external integrations. AgentDyn preserves native tools and the trusted-`user` /
untrusted-`input` boundary. DTap exposes checksum-locked preparation, portable
33-process sharded execution, strict resume, and complete-result summarization
for the reviewed 1,838-task non-VM selection. Its concrete backend invokes the
patched external DTap evaluation, task setup, Compose/MCP lifecycle,
Qwen-Agent loop, judges, and verified shutdown. The exact task-selection
manifest is bundled; the selected payloads remain in the pinned DTap checkout.
Neither integration stores a runtime
connection value, external checkout, task environment, model weight, or result.

See `docs/AGENTDYN.md` and `docs/DTAP.md` for the exact upstream pins,
verification commands, runtime contracts, and clean evaluation workflows.

## Completeness and publication gates

- AgentDojo, AgentDyn, DTap, and Qwen-Agent remain external installations;
  their accepted revisions, licenses, reviewed patches, and integration
  contracts are pinned.
- DTap task payloads, containers/MCP environments, external research judge,
  the evaluated SecPO checkpoint, and its model service remain prerequisites;
  the exact selection manifest, task callable, and judge mapping are included.
- Exact scientific runs still require local victim/attacker/judge serving and
  separately distributed model weights. The release stores neither endpoints
  nor credentials.
- Project-owned code and documentation are licensed under Apache-2.0.
  Benchmark, dataset, model, and third-party terms remain separate; the
  bundled Qwen dataset redistribution and privacy decisions are still
  publication gates.
- No model weights, attacker/judge credentials, private endpoints, or raw evaluation outputs
  are included.

See `REPRODUCIBILITY.md`, `docs/EVALUATION.md`, `docs/THIRD_PARTY.md`, and
`RELEASE_CHECKLIST.md` for exact parameters and gates.

## Tests and release audit

```bash
python -m pytest -q
python -m compileall -q src scripts tests examples
bash -n scripts/*.sh
python scripts/validate_release_manifest.py
bash scripts/audit_public_release.sh
```

The manifest validator is read-only and deny-first. To change what is backed
up or released later, edit the component classification and narrow path rules
in `release_manifest.json`, add tests, and rerun every audit. Do not weaken a
global deny rule or import files directly from the research workspace.

## Release scope

Project-owned code and documentation are released under Apache-2.0. Model
weights, datasets, benchmarks, and third-party components remain subject to
their own terms. Release and provenance records are maintained in
`PUBLIC_RELEASE.md`, `RELEASE_CHECKLIST.md`, and `release_manifest.json`.

## Citation

```bibtex
@misc{chen2026secpo,
  title         = {{SecPO}: Principled Adversarial Training for Prompt Injection Security},
  author        = {Chen, Sizhe and Peng, Yibo and Chang, Jaewon and Sitawarin, Chawin and Wagner, David},
  year          = {2026},
  eprint        = {2610.41998},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2610.41998}
}
```
