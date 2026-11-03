# Dataset card: SecPO Qwen3.6-27B offline preferences

## Release status

This directory contains a lossless, split archive of the exact preference
dataset used as input to the Qwen3.6-27B offline main experiment. It is an
exact experimental asset, not a newly generated or cleaned derivative.

**Publication gate:** no dataset license is granted by this review copy.
Obtain institutional approval for redistribution before uploading the
repository. The source-data lineage includes Stanford Alpaca, whose data is
published under CC BY-NC 4.0. A conservative candidate for the derived
dataset is therefore CC BY-NC 4.0 with the upstream attributions below, but
the authors must confirm that choice and any model-output terms rather than
inferring a license from this card.

Related published training datasets are available on Hugging Face:

- [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset)
- [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset)

Each Hugging Face release card governs the terms of its published artifact.
Their availability does not itself grant permission to redistribute this
bundled exact archive or clear the privacy and institutional-review gate
above.

## Artifact identity

- Logical records: 19,157
- Raw format: one UTF-8 JSON array
- Raw size: 99,314,808 bytes
- Raw SHA-256:
  `cd349baa406eae0bf7e4ee2d938d087d6656469d88076d68ba76baeb7e5ec669`
- Deterministic gzip size: 31,771,837 bytes
- Deterministic gzip SHA-256:
  `f2515db72ca5058f190f1b5d32d70915bdfd91d90629269f6a2965ac052f18be`

The gzip stream has timestamp zero and no embedded source filename. It is
split into five ordinary Git files of at most 7,000,000 bytes so each remains
below a conservative 7,000,000-byte regular-Git hosting budget.
`qwen36_offline_dataset/manifest.json` records the ordered part sizes and
SHA-256 digests. The parts are binary and must not be edited.

Reconstruct and verify the original bytes:

```bash
python scripts/reconstruct_dataset.py \
  --manifest data/qwen36_offline_dataset/manifest.json \
  --output /tmp/qwen36_offline_preferences.raw.json
```

The command refuses to overwrite an existing output and verifies every part,
the complete gzip stream, and the reconstructed raw file.

## Record schema

Every raw record has exactly these fields:

- `prompt`: rendered attacked Qwen chat prompt.
- `chosen`: preferred response to the trusted task.
- `rejected`: response associated with following the injected task.
- `rejected_input_whole`: exact injected text removed to form the benign
  reference prompt.
- `chosen_input`: complete trusted instruction plus clean input.
- `rejected_input`: injection seed retained from an earlier construction
  stage; it is not consumed by the released trainer.
- `chosen_data`: trusted untrusted-data field retained for provenance checks.
- `inject_at_the_end`: whether the injected text was placed after the trusted
  data.

All fields except `inject_at_the_end` are nonempty strings. Each `prompt`
contains `rejected_input_whole` exactly once, and `chosen` differs from
`rejected`. There are no duplicate complete records.

The trainer intentionally consumes a smaller public schema. Prepare it
without modifying the archived source:

```bash
python scripts/audit_release_dataset.py \
  /tmp/qwen36_offline_preferences.raw.json \
  --schema raw \
  --expected-records 19157
python scripts/prepare_preferences.py \
  --input /tmp/qwen36_offline_preferences.raw.json \
  --output /tmp/qwen36_offline_preferences.jsonl.gz \
  --expected-records 19157
python scripts/audit_release_dataset.py \
  /tmp/qwen36_offline_preferences.jsonl.gz \
  --schema prepared \
  --expected-records 19157
```

The prepared gzip is deterministic and contains only `prompt`,
`clean_prompt`, `chosen`, and `rejected`. Point `DATA_FILE` at that file when
running `scripts/train_qwen36_27b_offline.sh`.

## Construction and provenance

The dataset begins with examples having both instruction and input fields
from the cleaned Alpaca lineage. A second instruction is inserted before or
after the untrusted input, and the result is rendered with the Qwen chat
template. For this optimized dataset:

1. Saved strong Llama-targeted attacks replaced the earlier injection text.
2. `chosen` and `rejected` were carried over by record index from the earlier
   Qwen response-generation stage; they were not regenerated after that
   optimized-prompt replacement.
3. `rejected_input` retains the earlier seed while
   `rejected_input_whole` contains the actual saved attack used by training.

Relevant upstream sources:

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), including
  its CC BY-NC 4.0 data notice.
- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned),
  whose dataset card describes the cleaning steps and currently labels the
  dataset CC BY 4.0.
- [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B).

The differing license labels in the Alpaca lineage are the reason this card
does not make an automatic licensing grant.

## Known artifacts and reproduction behavior

- The Qwen configuration filters prompts over 512 tokens. With the released
  tokenizer/configuration, 12 of the 19,157 archived records are filtered,
  leaving 19,145 training examples.
- The clean prompt used by training is the literal result of deleting
  `rejected_input_whole` once. Relative to the auxiliary `chosen_data` field,
  some records contain boundary-whitespace or rendering residue. Three
  especially inconsistent auxiliary-field cases occur at zero-based indices
  4,011, 12,327, and 18,366. The released preparation path preserves the
  training behavior instead of silently repairing these records.
- Two `chosen` strings are extreme character-length outliers (205,478 and
  74,728 characters). They are retained for exactness.

These details matter when comparing record counts or creating a sanitized
derivative. A modified corpus should receive a distinct name and checksum.

## Privacy, safety, and content warning

The corpus contains adversarial instructions and arbitrary generated text.
Treat every field as untrusted: do not execute embedded code or commands,
follow links, or reuse apparent credentials or contact details.

No known author-specific term, credential signature, private key, or duplicate
record was detected. The bundled field scan reports 484 email-like occurrences
and 26 local-path-like occurrences. A broader row-level heuristic pass flagged
144 rows containing email-like text, 491 containing URLs, 34 containing
IP-address-like text, 343 containing phone-number-like text, and 19 unique
strings that pass a payment-card checksum. These signals have false positives
and are likely to include synthetic/model-generated values, but synthetic
origin is not proof that a value is harmless or fictional.

Before public upload, either manually review and clear the flagged content or
record an institutional decision to release the exact corpus with this
warning. Do not claim that the dataset contains no personal information based
only on automated scanning.

## Intended use and limitations

The intended use is reproducing the benign-reference SecPO Qwen main training
experiment and studying text prompt-injection defenses. It is not a general
instruction-tuning dataset, a benchmark, or an evaluation set. No claim is
made that the generated responses are correct, safe, representative, or free
of copyrighted or personal material.
