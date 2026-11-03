# SecPO: Principled Adversarial Training for Prompt Injection Security

Recognized as a top threat to AI agents, prompt injection allows instructions from untrusted data (email, documents, websites) to manipulate the agent’s operation. The existing best-defended LLM, Meta-SecAlign, is trained with the DPO loss but still suffers from attack success rates (ASRs) approaching 100% under adaptive prompt injections. One reason is that DPO anchors the model around the undefended reference model, which is insecure. We show an idealized reference model—an injection-invariant, utility-preserving oracle—can nevertheless be approximated, as its output distribution in training is interestingly accessible. This new observation motivates a theoretically principled SecPO loss. Another Meta-SecAlign shortcoming is that it trains on weak simulated attacks. We implement an efficient adversarial training (AT) pipeline that synthesizes strong injected inputs (best@240), and train the model on them. Our AT reduces ASRs by 1.5–14.5× across three SoTA adaptive attacks, compared with Meta-SecAlign; the SecPO loss reduces them by a further 2.4–4.6× compared with using DPO loss. In agentic workflows, our SecPO + AT Qwen3.6-27B is much more secure than Meta-SecAlign: 8.8% vs. 78.8% AgentDojo ASR@800 under the AttackerMovesSecond attack. Moreover, our defended model shows no noticeable reduction in benign utility in nine benchmarks covering agentic tool calling, instruction following, science, and math.

[Sizhe Chen](https://sizhe-chen.github.io), [Yibo Peng](https://pppyb.github.io/), [Jaewon Chang](https://www.jaewon.io/), [Chawin Sitawarin](https://chawin.sitawarin.com/), [David Wagner](https://people.eecs.berkeley.edu/~daw)

[Paper](https://arxiv.org/abs/2610.41998)

Links to our open model checkpoints.

| Base model | Meta-SecAlign | SecPO | SecOPD |
| --- | --- | --- | --- |
| Llama-3.1-8B | [Model](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign) | [Model](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) | |
| Qwen3.6-27B | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Thinking-SecOPD) |
| Qwen3.8-27B | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.8-27B-Instruct-MetaSecAlign) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO) | [Model](https://huggingface.co/Sizhe-Chen/Qwen3.8-27B-Thinking-SecOPD) |

The Qwen3.8-27B Meta-SecAlign repository is a public placeholder; weights are pending.

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

## Static Agent Benchmarks: AgentDyn and DTaP

- **AgentDyn:** evaluate static prompt injections in dynamic agent
  environments using the [AgentDyn setup and evaluation guide](docs/AGENTDYN.md).
- **DTaP-Bench-Indirect:** evaluate 1,838 indirect-injection scenarios using
  the [DTaP setup and evaluation guide](docs/DTAP.md). DTaP also reports
  **utility under attack**.

For Qwen3.6-27B with thinking disabled, static ASR falls from **4.8% to 0.0%**
on AgentDyn and **45.9% to 7.6%** on DTaP, comparing Meta-SecAlign with SecPO.
See [Performance](docs/PERFORMANCE.md) for the complete comparisons.

## Utility Benchmarks on Benign Inputs

- **MMLU-Pro, GPQA Diamond, GSM8K, and Minerva MATH:** use
  [`evaluate_lm_eval_utility.py`](scripts/evaluate_lm_eval_utility.py) with the
  [Llama](configs/evaluate_lm_eval_llama31_8b.json) or
  [Qwen](configs/evaluate_lm_eval_qwen36_27b.json) configuration.
- **AlpacaEval2 and SEP utility:** generate each evaluated model's outputs,
  then use [`evaluate_reference_utility.py`](scripts/evaluate_reference_utility.py)
  with the fixed comparator outputs. Both report **raw win rate**; see the
  [reference evaluation guide](docs/REFERENCE_UTILITY.md).
- **AgentDojo and AgentDyn utility:** run the benign task evaluations
  described in the [AgentDojo](docs/AGENTDOJO.md) and
  [AgentDyn](docs/AGENTDYN.md) guides.

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
| Qwen3.6-27B SecPO and optimized-data SecOPD | [Qwen3.6-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset) |
| Qwen3.8-27B SecPO | [Qwen3.8-27B-Instruct-SecPO-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset) |
| Qwen3.8-27B SecOPD | [Qwen3.8-27B-Thinking-SecOPD-trainset](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.8-27B-Thinking-SecOPD-trainset) |

All four datasets are public. When adapting the preference data to a new model,
[generate that model's own response labels](docs/TRAINING.md#generate-model-specific-response-labels)
and render the training prompts with its tokenizer and chat template.

# Quantized Inference

An **8-bit Q8_0** conversion of the defended Qwen3.6-27B runs with
[llama.cpp](https://github.com/ggml-org/llama.cpp) on a **single RTX 5090
with 32 GB of memory**. Reusing the existing DFlash drafter without additional
training increases decoding throughput by **2.66× on AgentDojo** and **2.63×
on AgentDyn**, compared with the same quantized model without speculative
decoding. The Q8_0 model records **0.0% observed static ASR on both benchmarks**,
with and without speculative decoding.

The separately released
[bitsandbytes INT8 checkpoint](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit)
uses a different quantization format; the measurements above are for the
llama.cpp Q8_0 conversion.
