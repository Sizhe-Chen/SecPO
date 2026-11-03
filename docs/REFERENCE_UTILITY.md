# AlpacaEval2 and SEP Utility

This release freezes the two pairwise utility protocols used for the main
Llama-3.1-8B and Qwen3.6-27B evaluations. The evaluated model's outputs are
the candidate side of both comparisons.

| Suite | Rows | Comparator outputs | Primary metric |
| --- | ---: | --- | --- |
| AlpacaEval2 | 805 | `gpt-4o-2024-05-13` | raw win rate |
| SEP Utility | 9,160 | `meta-llama/Meta-Llama-3-8B-Instruct` | ordinary win rate |

Both suites use AlpacaEval's dataset-independent `get_winrate` aggregation
and report `win_rate`. Length-controlled win rate is a different metric and
must not replace these raw win-rate results.

## External reference files

The comparator-output files are not bundled. Their redistribution rights were
not established by the local research snapshot, and generated model text can
carry terms separate from the evaluation code. Obtain approved copies, then
let the runner verify the complete bytes and every row before making an
evaluator call:

| Expected basename | Rows | Generator label | SHA-256 |
| --- | ---: | --- | --- |
| `alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json` | 805 | `gpt-4o-2024-05-13` | `653c131d58e8d5802203c85e80ca3fb4b378f22e2a0280ad5cd910ff0bee9535` |
| `SEP_dataset_test_Meta-Llama-3-8B-Instruct copy.json` | 9,160 | `meta-llama/Meta-Llama-3-8B-Instruct` | `6defcfdb3ac9da4a03eff21403a0ac58ce52c0c1dc45c23fc48b17a2bba0808c` |

The filename is provenance documentation; the runner accepts a renamed copy
only when its full SHA-256, row count, generator labels, and identities match.
For SEP, it also checks that `instruction` is exactly `instruction_only` plus
the optional blank line and `input`. Candidate and reference identities must
be unique and occur in identical order, preventing positional misjudgment.

The AlpacaEval package is an external dependency. Its package metadata in the
audited environment labels the code Apache-2.0 and the evaluation data CC
BY-NC 4.0. Confirm the upstream revision and data terms independently before
redistributing either reference file. The local snapshot contained no clear
license record for the SEP dataset or its Llama-generated comparator outputs,
so this release treats them as external even though their exact identities are
known.

## Commands

First generate candidate JSON outputs in benchmark order. Each row needs a
string `output` and one non-empty `generator`. AlpacaEval2 rows are matched by
`instruction`. SEP rows are matched by `instruction`, `instruction_only`,
`input`, and `injection`.

Supply an AlpacaEval evaluator configuration at runtime. It may be a named
configuration in the independently installed package or a reviewed local
configuration file. No provider, endpoint, or credential is stored here.

```bash
export SECPO_UTILITY_ANNOTATORS_CONFIG=<alpaca-eval-annotator-name-or-config-path>

python scripts/evaluate_reference_utility.py \
  --config configs/evaluate_alpacaeval2_utility.json \
  --candidate /path/to/alpacaeval2_candidate_outputs.json \
  --reference /approved/path/to/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json \
  --output-dir results/alpacaeval2_utility

python scripts/evaluate_reference_utility.py \
  --config configs/evaluate_sep_utility.json \
  --candidate /path/to/sep_candidate_outputs.json \
  --reference "/approved/path/to/SEP_dataset_test_Meta-Llama-3-8B-Instruct copy.json" \
  --output-dir results/sep_utility
```

Commands perform full preflight and print the exact `alpaca_eval` argument
vector by default. Add `--execute` to run it. `--cache-path` optionally binds
a caller-managed annotation cache. The evaluator inherits no credential
variables by default. If the reviewed evaluator configuration requires a
variable that is already set in the parent shell, pass only that name with a
repeatable `--pass-env NAME` option, for example:

```bash
python scripts/evaluate_reference_utility.py \
  --config configs/evaluate_alpacaeval2_utility.json \
  --candidate /path/to/alpacaeval2_candidate_outputs.json \
  --reference /approved/path/to/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json \
  --output-dir results/alpacaeval2_utility \
  --execute \
  --pass-env EVALUATOR_API_TOKEN
```

The command fails if a named variable is unset. Unnamed keys, tokens,
passwords, secrets, and credentials remain absent from the child process and
are never persisted by SecPO code.

The comparator model and evaluator model are separate concepts: GPT-4o and
Llama-3-8B identify the fixed reference responses, not the runtime judge. A
scientific result should additionally record the AlpacaEval version, evaluator
configuration and checksum, provider model revision, decoding settings,
candidate checksum, and output artifacts without recording credentials.
