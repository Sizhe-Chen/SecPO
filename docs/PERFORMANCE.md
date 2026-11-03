# Performance

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
