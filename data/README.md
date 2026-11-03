# Preference data

This directory contains two distinct artifacts:

- `qwen36_offline_dataset/`: five binary parts plus a manifest that
  reconstruct the exact 19,157-record Qwen offline main-experiment dataset.
- `example_preferences.jsonl`: a tiny synthetic schema fixture used only by
  tests.
- `dtap_qwen36_non_vm_tasks.jsonl`: the exact 1,838-row malicious-indirect
  non-VM task-selection manifest used by the released DTap evaluation.

Read `DATASET_CARD.md` before sharing or processing the experimental data.
It records checksums, provenance, known artifacts, content-review findings,
and unresolved licensing requirements.

## Published training datasets

- [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset)
- [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset)

The release card for each Hugging Face dataset governs that published
artifact. These links do not resolve or waive the separate redistribution,
privacy, and institutional-review gates for the exact Qwen archive bundled in
this directory; consult `DATASET_CARD.md` before publishing this repository.

## Trainer contract

For another base model, first [generate that model's own response labels](../docs/TRAINING.md#generate-model-specific-response-labels)
with [`generate_response_labels.py`](../scripts/generate_response_labels.py).
The raw `chosen_input` already contains the clean task and input;
`rejected_input` is answered separately as the attacker target request.
`prepare_preferences.py` validates existing responses; it does not generate
them or rerender prompts for another model family.

The trainer reads JSON arrays, JSON Lines, and their `.json.gz` or
`.jsonl.gz` equivalents. Each prepared record must provide the four strings
below:

- `prompt`: fully rendered attacked prompt ending immediately before the
  assistant response.
- `clean_prompt`: the same prompt with the adversarial injection removed and
  all other bytes unchanged.
- `chosen`: response to the trusted instruction.
- `rejected`: response to the injected instruction.

An optional string `id` is retained for provenance checks.

Raw generator output can use `rejected_input_whole` instead of
`clean_prompt`. `scripts/prepare_preferences.py` removes exactly one occurrence
from `prompt`. It rejects zero or multiple matches, empty injections, empty
responses, or an explicitly supplied clean prompt that is identical to the
attacked prompt.

Prompts are expected to be pre-rendered. The trainer does not invent or modify
chat roles. Main-experiment prompts use `user` for the trusted instruction and
the model-specific `input` role for untrusted data.

## Exact Qwen data workflow

```bash
python scripts/reconstruct_dataset.py \
  --manifest data/qwen36_offline_dataset/manifest.json \
  --output /tmp/qwen36_offline_preferences.raw.json
python scripts/prepare_preferences.py \
  --input /tmp/qwen36_offline_preferences.raw.json \
  --output /tmp/qwen36_offline_preferences.jsonl.gz \
  --expected-records 19157
DATA_FILE=/tmp/qwen36_offline_preferences.jsonl.gz \
  bash scripts/train_qwen36_27b_offline.sh
```

Preparation retains the exact attacked prompt and responses but drops
auxiliary generator-only fields. It derives `clean_prompt` by one literal
deletion, matching the benign-reference training path.

## Llama online PAIR data workflow

The raw Llama rows use `rejected_input` as the attack goal and do not carry a
stable public ID. Prepare them with the opt-in online contract:

```bash
python scripts/prepare_preferences.py \
  --input /path/to/llama31_raw_preferences.json \
  --output /tmp/llama31_online_preferences.jsonl \
  --expected-records 19157 \
  --online-pair
```

This mode strips surrounding goal whitespace and maps it to `attack_goal`.
When `id` is absent, it hashes a domain-separated canonical encoding of
`prompt`, `chosen`, and `rejected`: each UTF-8 value is prefixed by its unsigned
eight-byte big-endian byte length, and the public ID is `pair-` plus the first
24 lowercase hexadecimal SHA-256 characters. Duplicate derived IDs fail the
whole conversion. The default preparation mode is unchanged.

## Exact DTap task selection

`dtap_qwen36_non_vm_tasks.jsonl` is a selection manifest, not copied benchmark
payload data. Its SHA-256 is
`ca0f2fc5728491561e4645ed25390b6311fc9831ecb9b72660d5f513fd6d8098`.
Every row identifies one Git-tracked task in DTap commit
`e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` with `domain`, `type`,
`task_id`, `threat_model`, and `risk_category`. The release loader requires
exactly 1,838 unique malicious/indirect rows, rejects macOS and Windows, and
verifies both the raw-file checksum and ordered semantic checksum before
sharding. See `docs/DTAP.md` for the external benchmark setup.
