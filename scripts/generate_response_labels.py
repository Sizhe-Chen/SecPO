#!/usr/bin/env python3
"""Generate model-specific SecPO response labels from raw preference records.

``chosen_input`` must already contain the complete trusted instruction AND
clean input. ``rejected_input`` is the attacker's target request. Each is
answered separately as an ordinary user message by the selected base model,
with thinking disabled. Only ``chosen`` and ``rejected`` are replaced.

This command does NOT reformat the attacked ``prompt`` or derive new attacks.
The input prompts must already use the selected model family's training chat
template. Do not use a Qwen-rendered raw file to train Llama by changing only
``--model``. For a new model family, construct its raw prompts first. Run
``prepare_preferences.py`` afterward to validate/minimize the training data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence


class LabelGenerationError(ValueError):
    """An input or generated response cannot form a SecPO preference pair."""


def load_records(source: Path) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise LabelGenerationError("input contains no records")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(decoded, dict):
        decoded = [decoded]
    if not isinstance(decoded, list) or not decoded:
        raise LabelGenerationError("input must be a JSON object, array, or JSONL")
    for index, row in enumerate(decoded):
        if not isinstance(row, dict):
            raise LabelGenerationError(f"record {index} must be an object")
        for key in ("prompt", "chosen_input", "rejected_input"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise LabelGenerationError(
                    f"record {index} requires a non-empty string {key!r}"
                )
        if not any(key in row for key in ("clean_prompt", "rejected_input_whole")):
            raise LabelGenerationError(
                f"record {index} needs clean_prompt or rejected_input_whole "
                "for subsequent preference preparation"
            )
    return decoded


def regenerate_records(
    records: Sequence[dict[str, Any]],
    generate: Callable[[list[str]], list[str]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Pair independent requests, replacing only response fields in new rows."""
    if batch_size < 1:
        raise LabelGenerationError("batch_size must be positive")
    result = []
    for offset in range(0, len(records), batch_size):
        rows = records[offset : offset + batch_size]
        requests = [row[key] for row in rows for key in ("chosen_input", "rejected_input")]
        responses = generate(requests)
        if len(responses) != len(requests):
            raise LabelGenerationError("generator returned the wrong response count")
        for index, row in enumerate(rows):
            chosen, rejected = responses[2 * index : 2 * index + 2]
            if any(not isinstance(value, str) or not value.strip() for value in (chosen, rejected)):
                raise LabelGenerationError(f"record {offset + index} has an empty response")
            if chosen == rejected:
                raise LabelGenerationError(
                    f"record {offset + index} has identical chosen/rejected responses; "
                    "review the requests or regenerate with another seed"
                )
            result.append({**row, "chosen": chosen, "rejected": rejected})
    return result


class TransformersLabelGenerator:
    def __init__(self, args: argparse.Namespace):
        # Keep --help, data validation, and unit tests independent of GPU packages.
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

        self.torch = torch
        set_seed(args.seed)
        common = {"revision": args.revision, "trust_remote_code": args.trust_remote_code}
        config = AutoConfig.from_pretrained(args.model, **common)
        if getattr(config, "is_encoder_decoder", False):
            raise LabelGenerationError("SecPO requires a decoder-only base model")
        multimodal = getattr(config, "vision_config", None) is not None or any(
            "ConditionalGeneration" in name or "ImageTextToText" in name
            for name in (config.architectures or [])
        )
        if multimodal:
            from transformers import AutoModelForImageTextToText

            loader = AutoModelForImageTextToText
        else:
            loader = AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
        if args.chat_template is not None:
            self.tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
        if not self.tokenizer.chat_template:
            raise LabelGenerationError("base tokenizer has no chat template; provide --chat-template")
        if self.tokenizer.eos_token_id is None or not self.tokenizer.eos_token:
            raise LabelGenerationError("base tokenizer must define an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = loader.from_pretrained(
            args.model, config=config, dtype=args.dtype,
            device_map=args.device_map, **common,
        ).eval()
        configured_eos = self.model.generation_config.eos_token_id
        self.eos_ids = list(dict.fromkeys(
            ([configured_eos] if isinstance(configured_eos, int) else list(configured_eos or []))
            + [self.tokenizer.eos_token_id]
        ))
        self.generation = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "eos_token_id": self.eos_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": 1,
            "return_dict_in_generate": False,
        }
        if args.temperature > 0:
            self.generation.update(temperature=args.temperature, top_p=args.top_p)

    def __call__(self, requests: list[str]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": request}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for request in requests
        ]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False,
            add_special_tokens=False, return_attention_mask=True,
        )
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_width = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.generation)
        responses = []
        for sequence in outputs:
            tokens = sequence[prompt_width:].tolist()
            end = next((i for i, token in enumerate(tokens) if token in self.eos_ids), None)
            if end is None:
                raise LabelGenerationError(
                    "response reached the token limit without EOS; increase --max-new-tokens"
                )
            response = self.tokenizer.decode(tokens[:end], skip_special_tokens=True)
            if not response.strip():
                raise LabelGenerationError("model generated an empty response")
            responses.append(response + self.tokenizer.eos_token)
        return responses


def write_records(records: Sequence[dict[str, Any]], destination: Path) -> None:
    """Publish a complete UTF-8 file without ever replacing an existing path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            if destination.suffix.lower() == ".jsonl":
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(list(records), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard-link publication is atomic and refuses overwrite,
        # including a destination created while model generation was running.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("--model", required=True, help="Base model to generate its own labels; not a LoRA adapter")
    result.add_argument("--input", required=True, type=Path, help="Raw JSON/JSONL already rendered for this model family")
    result.add_argument("--output", required=True, type=Path, help="New .json or .jsonl file; existing files are never overwritten")
    result.add_argument("--revision", help="Base model/tokenizer revision; pin a commit for reproducibility")
    result.add_argument("--chat-template", type=Path, help="Optional native template for label requests only; does not change stored prompts")
    result.add_argument("--batch-size", type=int, default=2, help="Preference rows per batch (twice as many independent requests; default: 2)")
    result.add_argument("--max-new-tokens", type=int, default=8192)
    result.add_argument("--temperature", type=float, default=0.8, help="Use 0 for greedy generation (default: 0.8)")
    result.add_argument("--top-p", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    result.add_argument("--device-map", default="auto")
    result.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code only when you trust its pinned revision")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if os.path.lexists(args.output):
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.output.suffix.lower() not in (".json", ".jsonl"):
        raise LabelGenerationError("output must end with .json or .jsonl")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise LabelGenerationError("--batch-size and --max-new-tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise LabelGenerationError("--temperature must be finite and non-negative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise LabelGenerationError("--top-p must be in (0, 1]")
    records = load_records(args.input)
    generator = TransformersLabelGenerator(args)
    labeled = regenerate_records(records, generator, batch_size=args.batch_size)
    write_records(labeled, args.output)
    print(f"Wrote {len(labeled)} model-specific preference pairs to {args.output}")
    print("Next: validate this file with scripts/prepare_preferences.py.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LabelGenerationError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from None
