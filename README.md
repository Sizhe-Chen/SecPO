# SecPO: Principled Adversarial Training for Prompt Injection Security

[Sizhe Chen](https://sizhe-chen.github.io), [Yibo Peng](https://openreview.net/profile?id=~Yibo_Peng3), Jaewon Chang, [Chawin Sitawarin](https://openreview.net/profile?id=~Chawin_Sitawarin1), [David Wagner](https://people.eecs.berkeley.edu/~daw)

[![Paper](https://img.shields.io/badge/Paper-a8c66c)](https://arxiv.org/abs/2610.41998) [![Llama SecPO](https://img.shields.io/badge/Llama%20SecPO--8B-FFD21E)](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) [![Qwen SecPO](https://img.shields.io/badge/Qwen%20SecPO--27B-FFD21E)](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) [![Qwen SecOPD](https://img.shields.io/badge/Qwen%20SecOPD--27B-FFD21E)](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Thinking-SecOPD)

SecPO combines a clean-input reference objective with adversarial training to
secure LLMs against prompt injection while preserving task utility. On
Qwen3.6-27B, AgentDojo adaptive ASR@800 drops from **78.8% with Meta-SecAlign
to 8.8% with SecPO**. Our released optimized injections can be reused to train
other models without repeating attack generation.

# Environment Setup

- Use Linux, Python 3.11+, and CUDA GPUs. The main Qwen training run uses eight
  H200s; Llama uses eight for training and eight for online attack generation.
  The 8B demo loads one model at a time; allow roughly 16 GB for BF16 weights
  plus memory for generation. See [training settings](REPRODUCIBILITY.md).
- Install the package and prepare the bundled Qwen preference data:

```bash
git clone https://github.com/Sizhe-Chen/SecPO.git
cd SecPO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python setup.py
```

- `setup.py` verifies and prepares the archive at
  `outputs/data/qwen36_offline_preferences.jsonl.gz`. It preserves identical
  existing output. Model downloads use your Hugging Face access; accept the
  base model's terms and authenticate with `hf auth login` for gated Llama.
- Install [benchmark dependencies](third_party/README.md) for the suites you
  run. Attacker, victim, and judge services are configured in the linked
  evaluation guides.

# Demo

[`demo.py`](demo.py) compares the undefended Llama-3.1-8B with the released
SecPO model on clean and attacked movie-summary requests. It replays a saved
adaptive injection that attempts to redirect an *Avengers: Endgame* summary into a
robotic-gardener story. This is an optimized training example, not a held-out
evaluation. Trusted instructions use `user`; untrusted content uses `input`.

```bash
python demo.py
```

Use `python demo.py --show-prompt` to inspect the example without loading models.

# Evaluation

## AgentDojo Genetic Attack

Our released [AttackerMovesSecond implementation](src/secpo/attacks/agentdojo_exact_first.py)
makes strong adaptive evaluation affordable: **MiniMax-M3.1-Flash with high
reasoning** generates attacks, and **DeepSeek-V4.1-Flash-1017 with reasoning
disabled** scores failed candidates. Evaluating **all 80 AgentDojo cases at
ASR@800 costs less than $200** in this setup, even when every case exhausts
its search budget. Early stopping reduces the cost further. AgentDojo's exact
task checker determines success; API charges depend on provider pricing.

After [setting up AgentDojo and the model services](docs/AGENTDOJO.md#low-cost-attackermovessecond-section-46), run:

```bash
python scripts/evaluate_agentdojo_qwen36_27b.py \
  --config configs/evaluate_agentdojo_search_qwen36_secpo.json \
  adaptive --checkout /path/to/agentdojo \
  --backend-config configs/evaluate_agentdojo_search_qwen36_secpo.json \
  --output-dir results/agentdojo_search --resume
```

The main comparison below uses the original **Gemini-2.5-Pro** attack
configuration, separately from the cheaper configuration above. Thinking is
disabled for all three Qwen3.6-27B models.

| AgentDojo | Undefended | Meta-SecAlign | SecPO |
| --- | ---: | ---: | ---: |
| Genetic adaptive ASR@800 ↓ | 81.3% | 78.8% | **8.8%** |
| Benign utility ↑ | 89.7% | 87.6% | **90.7%** |

See the [configuration](configs/evaluate_agentdojo_search_qwen36_secpo.json)
and [AgentDojo guide](docs/AGENTDOJO.md) for the original search interface,
static evaluation, checkpoints, and resume behavior.

## SEP Static and Adaptive Attacks

[`evaluate_sep_static.py`](scripts/evaluate_sep_static.py) runs six static
attacks and model-specific completion attacks.
[`evaluate_sep_adaptive.py`](scripts/evaluate_sep_adaptive.py) runs Genetic,
PAIR, and TAP with supplied attacker, victim, and judge backends. Start with
the [commands and judging protocol](docs/EVALUATION.md#running-sep-and-utility-evaluations).

Across Genetic ASR@800, TAP ASR@800, and PISmith ASR@10, mean SEP ASR falls
from **89.6% to 10.5%** for Llama and **92.6% to 9.2%** for Qwen, comparing
Meta-SecAlign with SecPO. See [all results and evaluation sizes](docs/PERFORMANCE.md).

## Utility and Other Agent Benchmarks

- **MMLU-Pro, GPQA Diamond, GSM8K, and Minerva MATH:** use
  [`evaluate_lm_eval_utility.py`](scripts/evaluate_lm_eval_utility.py) with the
  [Llama](configs/evaluate_lm_eval_llama31_8b.json) or
  [Qwen](configs/evaluate_lm_eval_qwen36_27b.json) configuration.
- **AlpacaEval2 and SEP utility:** generate each evaluated model's outputs,
  then use [`evaluate_reference_utility.py`](scripts/evaluate_reference_utility.py)
  with the fixed comparator outputs. Both report **raw win rate**; see the
  [reference evaluation guide](docs/REFERENCE_UTILITY.md).
- **AgentDyn and DTaP:** follow the [AgentDyn](docs/AGENTDYN.md) and
  [DTaP](docs/DTAP.md) setup and evaluation guides.

Mean utility across the six instruction-following, science, and math
benchmarks is **47.5% vs. 46.9%** for Llama SecPO vs. undefended, and
**89.1% vs. 88.9%** for Qwen. Full tables are in [Performance](docs/PERFORMANCE.md).

# Defensive Fine-Tuning (SecPO)

**Each base model must generate its own preferred and rejected response
labels.** Reuse our optimized injections, but regenerate `chosen` and
`rejected` with the model you will fine-tune. Use its native tokenizer and
chat template as well. The supplied Qwen archive already contains Qwen labels.

[`generate_response_labels.py`](scripts/generate_response_labels.py) generates
both responses separately from `chosen_input` (the complete clean task) and
`rejected_input` (the attacker target request). For a new model, follow the
[model-specific label-generation workflow](docs/TRAINING.md#generate-model-specific-response-labels)
before preparing preferences and training.

For the bundled Qwen3.6-27B offline run:

```bash
DATA_FILE=outputs/data/qwen36_offline_preferences.jsonl.gz \
  bash scripts/train_qwen36_27b_offline.sh
```

For Llama-3.1-8B online training, prepare Llama-specific labels, start the
[current-policy PAIR workers](docs/LLAMA31_CURRENT_POLICY_PAIR.md), then run
[`train_llama31_8b_online.sh`](scripts/train_llama31_8b_online.sh). See
[Training](docs/TRAINING.md) for the objective, data commands, and model merge.
Released data: [Llama](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset)
and [Qwen](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset).

# Code Acknowledgements

SecPO builds on [Meta-SecAlign](https://github.com/facebookresearch/Meta_SecAlign)
and [SecAlign](https://github.com/facebookresearch/SecAlign). Evaluation
integrates [AgentDojo](https://github.com/ethz-spylab/agentdojo), AgentDyn,
DTaP, Qwen-Agent, and lm-evaluation-harness; see
[third-party acknowledgements](docs/THIRD_PARTY.md) and
[implementation provenance](docs/SOURCE_TO_RELEASE_MAP.md).

Project-owned code is [Apache-2.0](LICENSE). Models, datasets, and third-party
components retain their own terms. [Citation](CITATION.bib) ·
[Reproducibility](REPRODUCIBILITY.md) · [Release scope](docs/RELEASE_SCOPE.md).
