# Third-party software, models, and data

The repository `LICENSE` grants Apache-2.0 rights only for SecPO-owned code and
documentation. It does not license datasets, model weights, benchmarks,
external patches, or third-party components. Every external item below remains
subject to its own terms, attribution, access controls, acceptable-use
requirements, and redistribution restrictions. Before publication, maintainers
must confirm authority to apply Apache-2.0 to the project-owned material and
prepare a third-party notice containing exact immutable revisions.

## Dependency plan

| Item | Release treatment | Required final record |
| --- | --- | --- |
| Base language models | External; no weights or cache views in this repository | Public model identifier, exact revision, license, access instructions, and artifact checksums where permitted |
| Transformers, PEFT, Accelerate, Datasets, PyTorch, DeepSpeed | Package dependencies | Exact tested versions or lock file and upstream licenses |
| Evaluation harness | External dependency | Exact package revision, task revisions, prompt/few-shot settings, and local patches |
| AlpacaEval 2 | External evaluation service/package | Exact revision, evaluator configuration, reference data provenance, and applicable terms |
| AgentDojo | Pinned external dependency plus a minimal patch | Immutable upstream commit, upstream license, patch file, patch checksum, and application/test instructions |
| AgentDyn | Pinned external dependency plus a reviewed compatibility patch and role template | Immutable upstream commit, MIT license, patch/template checksums, and verification instructions |
| DTap | Pinned external dependency plus a reviewed Qwen adapter patch and role template | Immutable upstream commit, Apache-2.0 license, exact task-selection contract, patch/template checksums, and verification instructions |
| PI-Smith | Pinned external dependency plus reviewed patches for PI-Smith and for its AgentDojo checkout | Immutable upstream commits, MIT license, patch and template checksums, pinned environment lock files, and setup instructions |
| Qwen-Agent | Pinned DTap runtime dependency used by the released bridge | Package version and Apache-2.0 license are pinned; the clean trust adapter is released while model serving remains external |
| SEP attack templates and benchmark data | Static transforms included; benchmark data remains external unless separately approved | Template and data origin, benchmark revision, citations, license, and redistribution decision |
| Genetic/MAP-Elites, PAIR, and TAP reference implementations | Clean provider-neutral implementation included; provenance gate remains | Origin, license compatibility, retained concepts, and algorithm citations |
| Judge and attacker models | External runtime dependencies | Public model identifiers, immutable revisions, decoding parameters, and access requirements; no credentials or private endpoints |

### AgentDojo integration

Do not copy a development checkout, nested repository metadata, caches, or
results into this repository. The intended release shape is:

1. declare one immutable AgentDojo upstream commit;
2. keep upstream code external;
3. store only the small reviewed integration patch under
   `third_party/patches/agentdojo.patch`;
4. document a deterministic patch-application command;
5. test utility, static-security, and Genetic evaluation against the pinned
   tree; and
6. record the upstream and patched-tree checksums.

The accepted AgentDojo revision is
`d3640b5b03a88eb44dad96852e1de1ef437d836e` (package `0.1.30`). Its MIT
license SHA-256 is
`4285a071f2d382338e52b4fb0a186d952984a34d43a33d8872e1a1d8cb43401e`.
The verifier accepts only a pristine checkout at that revision or exactly the
reviewed one-file patch documented under `third_party/`.

### AgentDyn and DTap integrations

AgentDyn is pinned to commit
`5353cf7615b135cace8d07c8f12dac53a16b6db3` (package `0.1.35`) under the
MIT license. DTap is pinned to commit
`e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` (package `0.2.12`) under
Apache-2.0, with `qwen-agent==0.0.34` as a separately licensed Apache-2.0
runtime dependency. Their reviewed patches live at
`third_party/patches/agentdyn.patch` and `third_party/patches/dtap.patch`;
their Qwen role templates live under `templates/`.

Neither upstream tree is vendored. The verifiers reject the wrong commit,
license, patch, template, untracked file, or additional tracked change. The
release also excludes historical runtime layers, scheduler policy, service
leases, provider routing, and generated state. See `AGENTDYN.md`, `DTAP.md`,
and `../third_party/README.md` for checksums and setup commands.
The DTap patch carries the paired stall-resistant lower-level Qwen input-role
base and registers the clean release-owned final Qwen-Agent adapter. The
released backend invokes DTap's complete evaluation, task setup, Compose/MCP
lifecycle, trajectories, judges, and verified shutdown. The exact 1,838-row
selection manifest is bundled, while task payloads remain in the pinned DTap
Git tree. The external research-task judge is configured only at runtime with
provider-neutral settings; no endpoint, model alias, or credential is baked
into the patch. Containers, prewarmed runtime dependencies, the judge service,
the evaluated checkpoint, and model service are still external and are not
implied to be bundled.

### Adaptive-attack provenance

The released Genetic/MAP-Elites, PAIR, and TAP modules are clean,
provider-neutral implementations rather than copies of the research scripts.
That fact does not by itself settle provenance: an older Genetic source carried
an upstream copyright header, while the origin and licensing of one
MAP-Elites research helper were not clear in the audited workspace. Confirm
the applicable origins, compatible licenses, required notices, and citations
before publicly distributing the adaptive modules under Apache-2.0. The
project license covers only SecPO-owned contributions; it neither relicenses
third-party material nor establishes an unencumbered redistribution right.
Until that review is recorded, the adaptive modules remain a separate
publication gate.

The static SEP module freezes attack templates and two model-specific fake
delimiters derived during the research evaluation. Confirm the benchmark and
template provenance and include the required citation or notice. The release
does not need private model caches to use the frozen values, but a later claim
that independently regenerating them is supported would require a documented
embedding-search procedure and exact tokenizer/model revision.

## Preference-data provenance gate

The bundled Qwen preference archive is a derived research dataset. Its card
documents its immediate construction, exact checksum, known auxiliary-field
inconsistencies, and automated content review. Before public redistribution,
maintainers must independently confirm:

- the license and attribution chain for the cleaned instruction source;
- permission to redistribute model-generated chosen and rejected text;
- the terms of every model used to create or optimize injections;
- whether any synthetic PII-like strings require removal or institutional
  review; and
- that the release description does not imply every response was regenerated
  after the optimized-injection replacement when it was carried over by
  record index.

Automated scans are evidence, not a guarantee that the data contains no
personal or restricted material. Keep the dataset license separate from the
code license, and do not publish the archive until both decisions are
affirmative.

## Vendoring rule

Prefer declared dependencies and small patches over copied upstream trees. If
vendoring is unavoidable, remove nested version-control metadata and generated
state, preserve license and notice files, record the exact source revision and
checksum, and run the same release validator over the vendored files after
adding narrow allow rules.
