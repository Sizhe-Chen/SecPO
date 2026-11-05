# Performance

Scores are percentages, except victim query counts, costs, and throughput
speedups. Higher utility and lower attack success rate (ASR) are better.
The Qwen3.6-27B results below use thinking disabled. Meta-SecAlign uses DPO
without adversarial training. SecPO uses online adversarial training for
Llama and reuses the optimized injections offline for Qwen.

## Llama-3.1-8B-Instruct

### Utility

| Metric | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign) | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| MMLU-Pro (↑) | 46.5% | 46.7% | 46.3% |
| GPQA Diamond (↑) | 23.7% | 24.2% | 24.2% |
| GSM8K (↑) | 82.9% | 84.4% | 81.3% |
| Minerva Math (↑) | 49.2% | 48.8% | 50.1% |
| AlpacaEval2 (↑) | 23.7% | 24.0% | 23.7% |
| SEP Utility (↑) | 55.4% | 55.9% | 59.4% |

### SEP

| Metric | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign) | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| SEP ASR (↓) | 98.1% | 1.8% | 1.4% |
| Genetic Adpt. ASR@800 (↓) | 100.0% | 89.1% | 12.3% |
| TAP Adpt. ASR@800 (↓) | 100.0% | 84.6% | 12.2% |
| PISmith Adpt. ASR@10 (↓) | 98.7% | 95.1% | 7.0% |

## Qwen3.6-27B

### Utility

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
| DTaP Utility under attack (↑) | 65.8% | 70.0% | 74.5% |

### SEP

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| SEP ASR (↓) | 98.9% | 2.9% | 1.8% |
| Genetic Adpt. ASR@800 (↓) | 100.0% | 93.7% | 13.3% |
| TAP Adpt. ASR@800 (↓) | 100.0% | 90.1% | 12.5% |
| PISmith Adpt. ASR@10 (↓) | 98.1% | 94.0% | 1.8% |

### AgentDojo

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| AgentDojo ASR (↓) | 26.9% | 2.2% | 0.0% |
| Genetic Adpt. ASR@800 (↓) | 81.3% | 78.8% | 8.8% |

### AgentDyn and DTaP

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| AgentDyn ASR (↓) | 22.0% | 4.8% | 0.0% |
| DTaP ASR (↓) | 50.8% | 45.9% | 7.6% |

### DTaP subsets

The following scores are ASR (↓). The semantically close subset contains
481 cases, and the multiple injections subset contains 123 cases.

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| Semantically close subset | 58.8% | 55.1% | 12.3% |
| Multiple injections subset | 51.2% | 33.3% | 3.3% |

### Visual prompt injection

DocVQA measures utility. Repeat-After-Me evaluates three visual injection
goals with up to 1,600 attempts each.

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) |
| --- | ---: | ---: | ---: |
| DocVQA Utility (↑) | 96.0% | 95.8% | 95.2% |
| Steal-PII ASR@1600 (↓) | 100.0% | 100.0% | 0.0% |
| Call-Tool ASR@1600 (↓) | 96.0% | 77.0% | 0.0% |
| OpenClaw ASR@1600 (↓) | 80.0% | 92.0% | 0.0% |

### AgentDojo attack queries and cost

The original Gemini-2.5-Pro attack configuration evaluates 80 task pairs
with up to 800 attempts each. Query and cost statistics are conditioned
on successful attacks. They describe the original evaluation configuration;
the released inexpensive attack configuration has separate pricing.

| Target | ASR (↓) | Victim queries<br>Mean | Victim queries<br>Median | Cost ($)<br>Mean | Cost ($)<br>Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 81.3% | 26 | 13 | 0.88 | 0.43 |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-MetaSecAlign) | 78.8% | 109 | 53 | 5.92 | 2.56 |
| [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO) | 8.8% | 227 | 96 | 34.90 | 13.20 |

## White-box attacks

These Llama-3.1-8B-Instruct evaluations require access beyond the main
black-box threat model: Basic Adaptive uses model embedding weights, and
GCG uses complete model weights. Scores are ASR (↓).

| Loss | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)<br>Undef. | [Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-MetaSecAlign)<br>DPO | [Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO)<br>SecPO |
| --- | ---: | ---: | ---: |
| Simulated Attack | – | Static | Adaptive |
| Basic Adaptive | 97.1% | 33.0% | 10.6% |
| GCG | 87.0% | 20.7% | 5.8% |

## Quantization and speculative decoding

The Q8_0 evaluations use llama.cpp on one RTX 5090 with 32 GB of memory.
SD denotes speculative decoding with a DFlash drafter reused without
additional training. Scores are ASR (↓).

### Original BF16

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)<br>Undef. | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO)<br>Ours |
| --- | ---: | ---: |
| AgentDojo ASR | 26.9% | 0.0% |
| AgentDyn ASR | 22.0% | 0.0% |

### Q8_0

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)<br>Undef. | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO)<br>Ours |
| --- | ---: | ---: |
| AgentDojo ASR | 11.2% | 0.0% |
| AgentDyn ASR | 15.5% | 0.0% |

### Q8_0 + SD

| Metric | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)<br>Undef. | [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO)<br>Ours |
| --- | ---: | ---: |
| AgentDojo ASR | 11.6% | 0.0% |
| AgentDyn ASR | 15.5% | 0.0% |

DFlash increases throughput by **2.66x** on AgentDojo and **2.63x** on
AgentDyn relative to the same Q8_0 model without speculative decoding.
The public [Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit](https://huggingface.co/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-8Bit)
checkpoint uses bitsandbytes INT8. The Q8_0 results above describe the
llama.cpp quantization evaluated here.

## Evaluation details

Across the three adaptive SEP attacks, mean ASR falls from **89.6% to
10.5%** for Llama (**8.5x lower**) and from **92.6% to 9.2%** for Qwen
(**10.1x lower**). Mean utility across the six instruction-following,
science, and math benchmarks is **47.5%** for Llama SecPO versus **46.9%**
undefended, and **89.1%** for Qwen SecPO versus **88.9%** undefended.

SEP ASR covers 9,160 examples and aggregates six static attack templates:
an example counts as compromised if any template succeeds. Adaptive SEP
uses 1,024 examples, with 800 attempts for Genetic/TAP and 10 for PISmith.
AlpacaEval2 and SEP utility are raw win rates.

AgentDojo evaluates utility on 97 tasks and the static attack on 949 task
pairs. Its adaptive evaluation uses 80 task pairs and up to 800 attempts
each. AgentDyn evaluates utility on 60 tasks and the static attack on 560
injected cases. Both use the `repeat_user_prompt` defense and the
`important_instructions` static attack.

DTaP covers 1,838 indirect-injection cases after excluding 160 macOS and
Windows cases. Its utility is measured **under attack**. A measured 0.0%
ASR means no successful attacks were observed in that evaluation.

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
