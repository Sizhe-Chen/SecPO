# Training SecPO

Run commands from the repository root. The released training launchers cover
Llama-3.1-8B online SecPO and Qwen3.6-27B offline SecPO.

## Generate model-specific response labels

**Generate a separate set of response labels with each base model you plan to
fine-tune.** Optimized injections can transfer between models; response labels
and rendered chat prompts remain model-specific.

- `chosen_input` is the complete trusted instruction plus clean input. Generate
  `chosen` by presenting this complete text to the selected base model.
- `rejected_input` is the attacker target request, presented independently as
  an ordinary user request to generate `rejected`. Do not obtain this label by
  hoping an injection succeeds against the clean model.
- Keep the original example identity and attack placement. Render training
  `prompt` with the target model's trust-role template: trusted task as `user`,
  untrusted data as `input`. Switching model families requires rerendering
  these prompts before label generation and preparation.

The generator reads raw JSON/JSONL records containing `chosen_input`,
`rejected_input`, and the already rendered attacked `prompt`. It replaces only
`chosen` and `rejected`, preserves the other fields, and refuses to overwrite
an existing output. It uses the selected base model's native chat template
for the two independent response requests, with thinking disabled for SecPO.
It does not convert an existing Qwen training prompt into a Llama prompt.

```bash
python scripts/generate_response_labels.py \
  --model Qwen/Qwen3.6-27B \
  --input /path/to/qwen36_raw_preferences.json \
  --output outputs/data/qwen36_regenerated_labels.jsonl

python scripts/prepare_preferences.py \
  --input outputs/data/qwen36_regenerated_labels.jsonl \
  --output outputs/data/qwen36_regenerated_preferences.jsonl.gz \
  --expected-records 19157

DATA_FILE=outputs/data/qwen36_regenerated_preferences.jsonl.gz \
  bash scripts/train_qwen36_27b_offline.sh
```

Repeat generation with each base model and its corresponding raw input file.
For Llama, use `--model meta-llama/Llama-3.1-8B-Instruct`, then add
`--online-pair` to preference preparation. Use `--help` for batching and
generation limits. Regenerated labels create a new dataset and are not an
exact replay of the archived responses.

This follows the per-model self-generated label construction in
[Meta-SecAlign's generator](https://github.com/facebookresearch/Meta_SecAlign/blob/main/utils.py#L49).
The supplied Qwen archive already has Qwen-generated labels; those labels
were retained when optimized Llama-targeted injections replaced earlier
injections. `python setup.py` preserves that exact archive's training behavior.
See [dataset construction](../data/DATASET_CARD.md#construction-and-provenance).

This pairwise label workflow is for SecPO. SecOPD instead uses online teacher
distributions; its training loop is not implemented by these launchers.

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
fails if the match is missing or ambiguous. Read [the dataset card](../data/DATASET_CARD.md) before
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
