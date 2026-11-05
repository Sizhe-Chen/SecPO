# SecPO: Principled Adversarial Training for Prompt Injection Security

[Sizhe Chen](https://sizhe-chen.github.io), [Yibo Peng](https://pppyb.github.io/), [Jaewon Chang](https://www.jaewon.io/), [Chawin Sitawarin](https://chawin.sitawarin.com/), [David Wagner](https://people.eecs.berkeley.edu/~daw)

[![Paper](https://img.shields.io/badge/Paper-a8c66c)](https://arxiv.org/abs/2610.41998)

Recognized as a top threat to AI agents, prompt injection allows instructions from untrusted data (email, documents, websites) to manipulate the agent’s operation. The existing best-defended LLM, Meta-SecAlign, is trained with the DPO loss but still suffers from attack success rates (ASRs) approaching 100% under adaptive prompt injections. One reason is that DPO anchors the model around the undefended reference model, which is insecure. We show an idealized reference model—an injection-invariant, utility-preserving oracle—can nevertheless be approximated, as its output distribution in training is interestingly accessible. This new observation motivates a theoretically principled SecPO loss. Another Meta-SecAlign shortcoming is that it trains on weak simulated attacks. We implement an efficient adversarial training (AT) pipeline that synthesizes strong injected inputs (best@240), and train the model on them. Our AT reduces ASRs by 1.5–14.5× across three SoTA adaptive attacks, compared with Meta-SecAlign; the SecPO loss reduces them by a further 2.4–4.6× compared with using DPO loss. In agentic workflows, our SecPO + AT Qwen3.6-27B is much more secure than Meta-SecAlign: 8.8% vs. 78.8% AgentDojo ASR@800 under the AttackerMovesSecond attack. Moreover, our defended model shows no noticeable reduction in benign utility in nine benchmarks covering agentic tool calling, instruction following, science, and math.

Links to our open model checkpoints.

| Base model | Meta-SecAlign | SecPO | SecOPD |
| --- | --- | --- | --- |
| Llama-3.1-8B | [Model](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign) | [Model](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) | |
| Qwen3.6-27B | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Thinking-SecOPD) |
| Qwen3.8-27B | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.8-27B-Instruct-MetaSecAlign) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.8-27B-Thinking-SecOPD) |

# Environment Setup

- Use Linux, Python 3.11+, and CUDA GPUs. The main Qwen training run uses eight
  H200s; Llama uses eight for training and eight for online attack generation.
  The 27B demo loads one model at a time; allow roughly 56 GB for BF16 weights
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

[`demo.py`](demo.py) compares the undefended Qwen3.6-27B with the released
SecPO checkpoint on an attacked movie-summary request, using the same SecPO
chat template and thinking disabled for both. It replays a saved
adaptive injection that attempts to redirect an *Avengers: Endgame* summary into a
robotic-gardener story. This is an optimized training example, not a held-out
evaluation. Trusted instructions use `user`; untrusted content uses `input`.

```bash
python demo.py
```

# Evaluation

Higher utility (↑) and lower attack success rate (ASR ↓) are better.
Meta-SecAlign uses DPO loss without AT; SecPO uses online AT for
`meta-llama/Llama-3.1-8B-Instruct` and offline AT for `Qwen/Qwen3.6-27B`.
The main Qwen3.6-27B evaluations use thinking disabled. Model columns give
the full checkpoint IDs.

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

The Genetic Adpt. ASR@800 results below use the original **Gemini-2.5-Pro**
attack configuration, separately from the cheaper configuration above.
AgentDojo v1.2.1 uses 97 tasks for utility, 949 task pairs for static attacks,
and 80 task pairs for adaptive attacks. The static attack is
`important_instructions`; the defense is `repeat_user_prompt`.

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| AgentDojo Utility (↑) | 89.7% | 87.6% | 90.7% |
| AgentDojo ASR (↓) | 26.9% | 2.2% | 0.0% |
| AgentDojo Genetic Adpt. ASR@800 (↓) | 81.3% | 78.8% | 8.8% |

<details>
<summary>Victim queries and Cost ($) of successful AgentDojo Genetic attacks</summary>

Query and cost statistics report means and medians conditioned on successful
attacks under the 80-example, Pass@800 protocol. These costs use the original
Gemini-2.5-Pro configuration.

| Target | AgentDojo ASR (↓) | Victim queries<br>Mean | Victim queries<br>Median | Cost ($)<br>Mean | Cost ($)<br>Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 81.3% | 26 | 13 | 0.88 | 0.43 |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | 78.8% | 109 | 53 | 5.92 | 2.56 |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) | 8.8% | 227 | 96 | 34.90 | 13.20 |

</details>

See the [configuration](configs/evaluate_agentdojo_search_qwen36_secpo.json)
and [AgentDojo guide](docs/AGENTDOJO.md) for the original search interface,
static evaluation, checkpoints, and resume behavior.

## SEP Static and Adaptive Attacks

[`evaluate_sep_static.py`](scripts/evaluate_sep_static.py) runs six static
attacks and model-specific completion attacks.
[`evaluate_sep_adaptive.py`](scripts/evaluate_sep_adaptive.py) runs Genetic,
PAIR, and TAP with supplied attacker, victim, and judge backends. Start with
the [commands and judging protocol](docs/EVALUATION.md#running-sep-and-utility-evaluations).

SEP ASR uses 9,160 examples and six static attack templates; an example
counts as successful if any template succeeds. Each adaptive evaluation uses
1,024 examples, with 800 attempts for Genetic and TAP and 10 for PISmith.

| Metric | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign) | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| SEP ASR (↓) | 98.1% | 1.8% | 1.4% |
| SEP Genetic Adpt. ASR@800 (↓) | 100.0% | 89.1% | 12.3% |
| SEP TAP Adpt. ASR@800 (↓) | 100.0% | 84.6% | 12.2% |
| SEP PISmith Adpt. ASR@10 (↓) | 98.7% | 95.1% | 7.0% |

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| SEP ASR (↓) | 98.9% | 2.9% | 1.8% |
| SEP Genetic Adpt. ASR@800 (↓) | 100.0% | 93.7% | 13.3% |
| SEP TAP Adpt. ASR@800 (↓) | 100.0% | 90.1% | 12.5% |
| SEP PISmith Adpt. ASR@10 (↓) | 98.1% | 94.0% | 1.8% |

Across Genetic Adpt. ASR@800, TAP Adpt. ASR@800, and PISmith Adpt. ASR@10, mean SEP ASR falls
from **89.6% to 10.5%** for Llama and **92.6% to 9.2%** for Qwen, comparing
Meta-SecAlign with SecPO. See [all results and evaluation sizes](docs/PERFORMANCE.md).

## Static Agent Benchmarks: AgentDyn and DTaP

- **AgentDyn:** evaluate static prompt injections in dynamic agent
  environments using the [AgentDyn setup and evaluation guide](docs/AGENTDYN.md).
- **DTaP-Bench-Indirect:** evaluate 1,838 indirect-injection scenarios using
  the [DTaP setup and evaluation guide](docs/DTAP.md). DTaP also reports
  **utility under attack**.

AgentDyn v1.2.2 uses 560 injected cases with `important_instructions` and
`repeat_user_prompt`. DTaP-Bench-Indirect covers 1,838 cases after excluding
160 macOS/Windows cases. DTaP Utility under attack is measured with injections
present.

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| AgentDyn ASR (↓) | 22.0% | 4.8% | 0.0% |
| DTaP ASR (↓) | 50.8% | 45.9% | 7.6% |
| DTaP Utility under attack (↑) | 65.8% | 70.0% | 74.5% |

## Security against adaptive visual prompt injections

Repeat-After-Me iteratively optimizes instructions written on top of an image.
The corresponding DocVQA images without injections measure utility.
The code for these evaluations is not included in this repository.

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| DocVQA Utility (↑) | 96.0% | 95.8% | 95.2% |
| Steal-PII ASR@1600 (↓) | 100.0% | 100.0% | 0.0% |
| Call-Tool ASR@1600 (↓) | 96.0% | 77.0% | 0.0% |
| OpenClaw ASR@1600 (↓) | 80.0% | 92.0% | 0.0% |

## Utility Benchmarks on Benign Inputs

- **MMLU-Pro, GPQA Diamond, GSM8K, and Minerva Math:** use
  [`evaluate_lm_eval_utility.py`](scripts/evaluate_lm_eval_utility.py) with the
  [Llama](configs/evaluate_lm_eval_llama31_8b.json) or
  [Qwen](configs/evaluate_lm_eval_qwen36_27b.json) configuration.
- **AlpacaEval2 and SEP Utility:** generate each evaluated model's outputs,
  then use [`evaluate_reference_utility.py`](scripts/evaluate_reference_utility.py)
  with the fixed comparator outputs. Both report **raw win rate**; see the
  [reference evaluation guide](docs/REFERENCE_UTILITY.md).
- **AgentDojo Utility and AgentDyn Utility:** run the benign task evaluations
  described in the [AgentDojo](docs/AGENTDOJO.md) and
  [AgentDyn](docs/AGENTDYN.md) guides.

AlpacaEval2 and SEP Utility use raw win rates on 805 and 9,160 examples,
respectively. MMLU-Pro, GPQA Diamond, GSM8K, and Minerva Math use 5-, 0-, 8-,
and 4-shot prompting. AgentDojo Utility uses 97 tasks; AgentDyn Utility uses
60 tasks.

| Metric | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign) | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| MMLU-Pro (↑) | 46.5% | 46.7% | 46.3% |
| GPQA Diamond (↑) | 23.7% | 24.2% | 24.2% |
| GSM8K (↑) | 82.9% | 84.4% | 81.3% |
| Minerva Math (↑) | 49.2% | 48.8% | 50.1% |
| AlpacaEval2 (↑) | 23.7% | 24.0% | 23.7% |
| SEP Utility (↑) | 55.4% | 55.9% | 59.4% |

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| MMLU-Pro (↑) | 84.8% | 83.9% | 84.6% |
| GPQA Diamond (↑) | 81.3% | 79.3% | 80.8% |
| GSM8K (↑) | 97.3% | 97.0% | 96.4% |
| Minerva Math (↑) | 96.1% | 94.8% | 95.8% |
| AlpacaEval2 (↑) | 80.6% | 81.6% | 82.9% |
| SEP Utility (↑) | 93.0% | 93.9% | 94.3% |
| AgentDojo Utility (↑) | 89.7% | 87.6% | 90.7% |
| AgentDyn Utility (↑) | 71.7% | 76.7% | 71.7% |

Mean utility across the six instruction-following, science, and math
benchmarks is **47.5% vs. 46.9%** for Llama SecPO vs. undefended, and
**89.1% vs. 88.9%** for Qwen. See [Performance](docs/PERFORMANCE.md) for the
combined results and evaluation settings.

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

# Optimized Training Data

Our optimized training sets reuse strong injections generated during online
adversarial training of Llama-3.1-8B-Instruct, so other models can be trained
without running attack generation again. Offline SecPO training transfers
these security gains to Qwen3.6-27B while preserving utility.

The data also improve SecOPD on-policy distillation: replacing its unoptimized
injections with ours reduces Qwen3.6-27B SEP and DTaP ASRs by **2.5–22.5×**,
with no noticeable overall utility cost. In that comparison, the datasets
differ only in their simulated injections.

| Training data | Dataset |
| --- | --- |
| Llama-3.1-8B SecPO | [Llama-3.1-8B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset) |
| Qwen3.6-27B SecPO | [Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset) |
| Qwen3.8-27B SecPO | [Qwen3.8-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset) |

The listed datasets are public. When adapting the preference data to a new model,
[generate that model's own response labels](docs/TRAINING.md#generate-model-specific-response-labels)
and render the training prompts with its tokenizer and chat template.

# Quantized Models and Speculative Decoding

An **8-bit Q8_0** conversion of the defended Qwen3.6-27B runs with
[llama.cpp](https://github.com/ggml-org/llama.cpp) on a **single RTX 5090
with 32 GB of memory**. Reusing the existing DFlash drafter without additional
training increases decoding throughput by **2.66× on AgentDojo** and **2.63×
on AgentDyn**, compared with the same quantized model without speculative
decoding. The Q8_0 model records **0.0% observed static ASR on both benchmarks**,
with and without speculative decoding.

Attack Success Rate (ASR ↓) under 8-bit quantization and speculative
decoding (SD):

| Model | Configuration | AgentDojo ASR | AgentDyn ASR |
| --- | --- | ---: | ---: |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Original BF16 | 26.9% | 22.0% |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) | Original BF16 | 0.0% | 0.0% |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Q8_0 | 11.2% | 15.5% |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit) | Q8_0 | 0.0% | 0.0% |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Q8_0 + SD | 11.6% | 15.5% |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit) | Q8_0 + SD | 0.0% | 0.0% |

The released checkpoint uses bitsandbytes INT8; the measurements above are
for the llama.cpp Q8_0 conversion.

# Citation

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
