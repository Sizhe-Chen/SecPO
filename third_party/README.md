# External benchmark dependencies

This release does not vendor AgentDojo, AgentDyn, DTap, their Git histories,
benchmark data, task environments, or generated runs. Each integration
accepts only its audited external checkout.

## AgentDojo

- repository: `https://github.com/ethz-spylab/agentdojo.git`
- commit: `d3640b5b03a88eb44dad96852e1de1ef437d836e`
- package version at that commit: `0.1.30`
- license: MIT
- audited `LICENSE` SHA-256:
  `4285a071f2d382338e52b4fb0a186d952984a34d43a33d8872e1a1d8cb43401e`

The upstream MIT license and attribution were verified from the pinned local
checkout. Copyright is held by the AgentDojo authors listed in that license.
Users must retain the upstream license when redistributing AgentDojo or a
substantial portion of it.

## Optional patch

[`patches/agentdojo.patch`](patches/agentdojo.patch) is a minimal, portable
resumption fix. It makes two cached-result reads treat a truncated JSON/UTF-8
record as incomplete, so an interrupted evaluation is rerun instead of
aborting. It changes only `src/agentdojo/benchmark.py`, has no provider or
credential code, has SHA-256
`96c66d764cb7df5c3d20a4235575f25034c7cf6650f8887df2fc5fe65d39bab0`,
and was verified with `git apply --check` against the exact commit above.

The patch is not required by the SecPO native-tool adapter. A pristine checkout
or a checkout containing exactly this patch is accepted; any other tracked or
untracked change fails verification.

```bash
git clone https://github.com/ethz-spylab/agentdojo.git
git -C agentdojo checkout d3640b5b03a88eb44dad96852e1de1ef437d836e
git -C agentdojo apply /path/to/SecPO/third_party/patches/agentdojo.patch
python scripts/evaluate_agentdojo_qwen36_27b.py verify --checkout agentdojo
```

The broader patch in the research workspace was intentionally not copied: it
mixed the Qwen adapter with unrelated hosted-provider and compatibility work.
This release instead keeps its Qwen OpenAI-compatible adapter in
`src/secpo/evaluation/agentdojo_qwen36.py`.

## AgentDyn

- repository: `https://github.com/SaFo-Lab/AgentDyn.git`
- commit: `5353cf7615b135cace8d07c8f12dac53a16b6db3`
- package version: `0.1.35`
- license: MIT
- audited `LICENSE` SHA-256:
  `4285a071f2d382338e52b4fb0a186d952984a34d43a33d8872e1a1d8cb43401e`
- reviewed patch SHA-256:
  `43a8e090daa08854d93cc8ea613d203354b004da645253b90380b3e59a7075f6`
- reviewed logical patched-tree SHA-256:
  `c0cc7b493d3239c57e7de622f398a8e04f8b6f102b690ca31d99f43695fb353f`
- Qwen role-template SHA-256:
  `0ae8886ac5955f0bbb9b67da4065b701902891ff67249baf8e4cfd4c01d55aa7`

The patch is `patches/agentdyn.patch`; the matching template is
`../templates/qwen36_agentdyn_input_role.jinja`. The verifier requires the
exact patched tree and rejects untracked or additional changes:

```bash
git clone https://github.com/SaFo-Lab/AgentDyn.git agentdyn
git -C agentdyn checkout 5353cf7615b135cace8d07c8f12dac53a16b6db3
git -C agentdyn apply /path/to/SecPO/third_party/patches/agentdyn.patch
python scripts/evaluate_agentdyn_qwen36_27b.py \
  --checkout agentdyn verify
```

See `../docs/AGENTDYN.md` for the native-tool, trusted-`user`,
untrusted-`input`, and runtime-endpoint contracts.

## PI-Smith

- repository: `https://github.com/albert-y1n/PISmith.git`
- commit: `16865a73b1930ac98fc90d517f1ab20d532f912b`
- license: MIT
- `patches/pismith.patch` SHA-256:
  `efe029cfbad71792268ff37cac9d1fd3c59073ac281dd63a728f385f4c957ba1`
- `patches/pismith_agentdojo.patch` SHA-256:
  `0296b79b24e903b53344dff4d1fcf86115cbc968d064ff1c27c08289a670863f`
- Qwen3.8 input-role template `../templates/qwen38_input_role.jinja` SHA-256:
  `3439e5dc3169221799ef9fa7cc3290aa7b19436c3fe7bf589ae4a736131b0c45`

`patches/pismith.patch` changes PI-Smith training for the recorded runs: the
untrusted data is sent in the Qwen `input` role, thinking-mode targets are
closed after a timeout, a SEP file can be used as training data, and the
AgentDojo reward adds a dense judge score for failed attacks.

`patches/pismith_agentdojo.patch` applies to the AgentDojo commit above. It
is a different patch from `patches/agentdojo.patch` and is used only by
PI-Smith: it reads the local model's address from `AGENTDOJO_VLLM_URL`, which
PI-Smith sets from the target URL, instead of a fixed port; it lets the
local-model pipeline return tool results in the `tool` or `input` role, adds Qwen thinking controls and tool-call parsing, and adds
the v1.2.2 task suites from AgentDojo `v0.1.35`
(`a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`). It also makes
the workspace injection tasks 10 to 12 return a failed attack, instead of
raising `TypeError`, when the checked email carries a calendar event as an
attachment.

`../scripts/setup_pismith.sh` clones both repositories at these commits,
applies the patches, and installs the environments pinned in
`../requirements/pismith_*.lock.txt`. See `../docs/PISMITH.md`.

## DTap and Qwen-Agent

- DTap repository: `https://github.com/AI-secure/DecodingTrust-Agent.git`
- DTap commit: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- DTap package version: `0.2.12`
- DTap license: Apache-2.0
- audited DTap `LICENSE` SHA-256:
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- cleaned release DTap patch SHA-256:
  `c63e22b067719827741e4847932ce6629010509a28d0f0fb4ea8833485b3185d`
- audited research-source patch SHA-256:
  `e80552a24cf3db6a905a0007c94b5295189cb4c45956a89fb6e648dbd0fca684`
- Qwen role-template SHA-256:
  `5fe53e6107dcd645c9a73457a694b158f5fb4ab839927a56867ec9a2c127cd90`
- Qwen-Agent: `qwen-agent==0.0.34`, Apache-2.0, license SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`

The reviewed files are `patches/dtap.patch`,
`../templates/dtap_qwen36_input_role.jinja`, and the release modules
`../src/secpo/evaluation/dtap_qwen_agent_{adapter,backend}.py`. The adapter is a
clean extraction of the final Qwen-Agent function loop and trust-preserving
transport; its SHA-256 is
`4d9293fdf8f0ed7d03a0e55a46de7a81db22b30f56da2501cb8486d27d21ac9b`.
The backend invokes the external benchmark's task setup, MCP lifecycle, and
judges through its complete `eval.evaluation` lifecycle. The model-service
address is mandatory runtime configuration through `DTAP_QWEN_BASE_URL`, and
thinking is frozen off. The 1,838-row task-selection manifest is bundled at
`../data/dtap_qwen36_non_vm_tasks.jsonl`; task payloads remain Git-tracked in
the pinned external checkout. Research-task judging requires explicit
provider-neutral runtime endpoint/model/credential settings and persists only
their non-secret identity digest. Docker Compose v2, prewarmed `uv` projects,
the Playwright Chromium bundle, task environments, model serving, and weights
remain external. See `../docs/DTAP.md` for verification and execution.
