# Reproducibility notes

## Released artifacts

| Artifact | Hugging Face repository |
| --- | --- |
| Llama training dataset | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset) |
| Qwen training dataset | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset) |
| Llama SecPO model | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) |
| Qwen SecPO model | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |

Use the immutable revisions and license terms recorded on the individual
Hugging Face cards. These repositories provide the released artifacts; they
do not change the external-runtime and exact-trajectory boundaries below.

## Main settings and implementation notes

| Setting | Llama-3.1-8B-Instruct | Qwen3.6-27B |
| --- | --- | --- |
| Adversarial data | Online, current-policy attacks | Offline, saved strong attacks |
| Inference mode | Direct response | Non-thinking |
| Preference pairs | 19,157 | 19,157 archived; 19,145 after length filter |
| Epochs | 1 | 1 |
| DPO activation | Sigmoid | Sigmoid |
| Beta | 0.1 | 0.1 |
| Learning rate | 1.6e-4 | 1.6e-4 |
| Precision | bf16 | bf16 |
| LoRA rank / alpha / dropout | 64 / 8 / 0.1 | 64 / 8 / 0.1 |
| Per-device batch | 2 | 1 |
| Gradient accumulation | 16 | 8 |
| Training processes | 8 | 8 |
| Maximum sequence length | 2,048 | 4,608 |
| Warmup ratio | 0 | 0.03 |
| LR schedule | Cosine | Linear |
| Gradient clipping | Disabled | Norm 1.0 |
| Release seed | 42 (see note) | 42 |
| Data/sampler seed | 0 | 42 (release default) |
| Drop incomplete microbatch | Yes | No |
| Optimizer updates | 74 for the 19,157-row/eight-rank run | Dataset-derived |

The shared DeepSpeed configuration intentionally leaves scheduler construction
to Transformers so that each model JSON selects the schedule shown above.

The original Llama Torchtune configuration used a synchronized
runtime-generated seed and did not retain a fixed value. This release pins 42
to make reruns of the clean wrapper deterministic; it cannot recover the
original run's seed. The Qwen checkpoint metadata records seed 42.

The Llama run used eight additional GPUs for current-policy inference during
attack generation, for 16 H200 GPUs in total. The paper reports approximately
18 hours for this setting. The Qwen offline run used eight H200 GPUs and took
approximately seven hours.

The paper's shared training paragraph lists query/value attention projections
plus the MLP projections. The executable Qwen implementation and the retained
Qwen checkpoint metadata also include key/output attention projections. The
Qwen configuration in this release follows that executable implementation:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. The Llama configuration uses `q_proj`, `v_proj`, and the three
MLP projections.

Qwen non-thinking training uses prompts rendered with an empty reasoning
block before the visible answer. Reasoning-on training variants are not part
of this release.

## Data construction

Start with Cleaned-Alpaca examples that have both a trusted instruction and an
untrusted data field. For each example:

1. Place another example's instruction before or after the untrusted data.
2. Render the trusted instruction with role `user` and the data with role
   `input`.
3. Use the undefended model's response to the clean trusted task as `chosen`.
4. Use a response to the injected instruction as `rejected`.
5. For the Llama main run, replace the weak injection with the best
   current-policy adaptive injection produced by the online worker.
6. For the Qwen main run, reuse the saved strong Llama-targeted injection and
   rerender the prompt with the Qwen chat template.

In the exact released Qwen archive, the optimized prompt and
`rejected_input_whole` come from that saved-attack replacement.
`chosen` and `rejected` were carried over by record index from the earlier
Qwen generation stage rather than regenerated after the replacement.
`rejected_input` retains the earlier seed and is not consumed by training.
These details are documented to prevent the archive from being
mischaracterized as freshly generated responses to every optimized prompt.

The trainer requires both the attacked and clean rendered prompt. Supplying
them explicitly is preferred because it avoids ambiguous text replacement.
For fail-closed preprocessing, this release requires the clean prompt to be
obtainable by one contiguous deletion and rejects multiple exact injection
matches. The experimental scripts used unrestricted string replacement, so
this validation rule is intentionally stricter.

The exact Qwen raw dataset is included as a deterministic split gzip archive.
It contains 19,157 records; the configured 512-token prompt filter removes 12
and trains on 19,145. See `data/DATASET_CARD.md` for checksums, known
auxiliary-field inconsistencies, and the required preparation commands.

## Qwen merge and agent evaluation

`scripts/merge_qwen36_27b_lora.py` converts a completed Qwen LoRA adapter into
a standalone checkpoint for external harnesses that cannot load PEFT adapters
directly. The clean merge validates the adapter delta and saved weight set,
preserves multimodal tensors when present, and writes a path-sanitized
`merge_manifest.json`. Model weights and merge outputs are never release
artifacts.

The Qwen AgentDojo, AgentDyn, and DTap integrations are pinned external
workflows. AgentDyn records native-tool, thinking, and trusted-`user` /
untrusted-`input` behavior in
`configs/evaluate_agentdyn_qwen36_27b.json`. DTap freezes a thinking-off,
1,838-task non-VM selection with 33 deterministic shards, a 262,144-token
context, and a 32,768-token output cap. The historical DTap profile used a
benign-reference checkpoint trained at learning rate `1.2e-4`, whereas the
main Qwen configuration released here uses `1.6e-4`; the protocol code does
not imply that those checkpoints are identical. The clean Qwen-Agent adapter
and concrete external-DTap lifecycle/judge backend are included. Actual runs
still require the pinned upstream benchmark data, task environments, evaluated
checkpoint, and a separately operated model service.
See `docs/AGENTDOJO.md`,
`docs/AGENTDYN.md`, and `docs/DTAP.md` for the exact external boundaries.

## Main results

See [Performance](docs/PERFORMANCE.md) for the complete utility and security
comparisons, model settings, metrics, and evaluation sizes.

The release includes static SEP, Basic Adaptive, Genetic/MAP-Elites, PAIR,
TAP, witness-first unanimous-nine judging, reference-utility evaluation,
Qwen merge, AgentDojo, AgentDyn, and DTaP interfaces. Scientific runs require
the external models, benchmark environments, and evaluator dependencies
described below. AlpacaEval2 and SEP report raw win rates. Fixed comparator
outputs are validated by generator, row count, alignment, and SHA-256.

## Reproduction boundary

The supplied code, data validator, and configurations implement the
benign-reference objective and main hyperparameters in a clean, shared Trainer
wrapper. The original Llama executable used Torchtune, so the framework runtime
is not bit-for-bit identical. The main current-policy PAIR behavior is now
included: eight streams, 30 rounds, current LoRA refresh, canonical
32-examples-per-rank mode, final victim generation, and unanimous nine-vote
confirmation. The Hugging Face trainer publishes a PEFT checkpoint after every
optimizer step. It uses data seed 0 and a 74-update cap for the full archived
dataset/topology, preventing a framework-generated partial 75th update. This
does not prove an identical Torchtune/Hugging Face index schedule.

The connected blocking collator waits for a two-example PAIR request before
the defender can continue, so one 30-round search normally sees a fixed policy.
The archived asynchronous 32-example/rank producer could observe an optimizer
update mid-search. The separate exact-batch worker enforces group size but needs
an external accumulation-group producer/pipeline to recover that concurrency;
the producer is not included. This is a trajectory difference, not merely a
batching difference. See
`docs/LLAMA31_CURRENT_POLICY_PAIR.md` for this explicit boundary. Model weights,
historical hosted-provider revisions, and the original cluster environment
remain external.

The adaptive search modules are clean source-backed behavioral refactors with
an explicit 800-victim-query cap for Genetic and TAP. The Qwen path reuses the
working Llama TAP orchestration rather than the stale Qwen fork; Genetic is
retained. Dense scores rank candidates, while witness-first unanimous 3-by-3
judging alone defines success. They do not reproduce proprietary provider
clients, asynchronous scheduling, or unrecorded random streams byte-for-byte.
Every final result must record the backend, prompts, immutable model/software
revisions, decoding parameters, nine-judge configuration, dataset checksum,
seed, query budget, and configuration fingerprint.
