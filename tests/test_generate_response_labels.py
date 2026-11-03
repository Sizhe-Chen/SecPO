from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_response_labels.py"
SPEC = importlib.util.spec_from_file_location("response_labels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
labels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(labels)


def record(index=0):
    return {
        "id": f"row-{index}",
        "prompt": f"target-native attacked prompt {index}",
        "chosen_input": f"Trusted task {index}\n\nClean input {index}",
        "chosen_data": f"Clean input {index}",
        "rejected_input": f"Attack target {index}",
        "rejected_input_whole": f"optimized injection {index}",
        "chosen": "old chosen",
        "rejected": "old rejected",
        "metadata": {"position": "end", "index": index},
    }


def test_requests_are_separate_and_only_labels_are_replaced():
    rows = [record(i) for i in range(3)]
    original = deepcopy(rows)
    batches = []

    def generate(requests):
        batches.append(requests)
        return [f"Response to {request}<eos>" for request in requests]

    result = labels.regenerate_records(rows, generate, batch_size=2)
    assert batches == [
        [rows[0]["chosen_input"], rows[0]["rejected_input"], rows[1]["chosen_input"], rows[1]["rejected_input"]],
        [rows[2]["chosen_input"], rows[2]["rejected_input"]],
    ]
    assert rows == original
    for before, after in zip(rows, result):
        assert after["chosen"] == f"Response to {before['chosen_input']}<eos>"
        assert after["rejected"] == f"Response to {before['rejected_input']}<eos>"
        assert {k: v for k, v in before.items() if k not in ("chosen", "rejected")} == {
            k: v for k, v in after.items() if k not in ("chosen", "rejected")
        }


class FakeTensor(list):
    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, device):
        assert device == "cpu"
        return self

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return FakeTensor(value) if isinstance(key, slice) else value

    def tolist(self):
        return list(self)


def fake_generator(*, terminated=True):
    chats = []
    captured = {}

    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            chats.append((messages, kwargs))
            return f"native:{messages[0]['content']}"

        def __call__(self, prompts, **kwargs):
            assert prompts == ["native:trusted", "native:attack target"]
            assert kwargs == {
                "return_tensors": "pt", "padding": True, "truncation": False,
                "add_special_tokens": False, "return_attention_mask": True,
            }
            return {"input_ids": FakeTensor([[0, 1], [2, 3]]), "attention_mask": FakeTensor([[0, 1], [1, 1]])}

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            assert len(ids) == 1
            return {11: "chosen answer", 12: "rejected answer"}[ids[0]]

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

        def generate(self, **kwargs):
            captured.update(kwargs)
            ending = [99, 0] if terminated else []
            return [FakeTensor([0, 1, 11] + ending), FakeTensor([2, 3, 12] + ending)]

    generator = object.__new__(labels.TransformersLabelGenerator)
    generator.torch = SimpleNamespace(inference_mode=nullcontext)
    generator.tokenizer = Tokenizer()
    generator.model = Model()
    generator.eos_ids = [99]
    generator.generation = {"max_new_tokens": 16}
    return generator, chats, captured


def test_native_chat_thinking_off_attention_mask_and_eos():
    generate, chats, captured = fake_generator()
    assert generate(["trusted", "attack target"]) == ["chosen answer<eos>", "rejected answer<eos>"]
    assert [messages for messages, _ in chats] == [
        [{"role": "user", "content": "trusted"}],
        [{"role": "user", "content": "attack target"}],
    ]
    assert all(options == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False} for _, options in chats)
    assert captured["attention_mask"] == [[0, 1], [1, 1]]


def test_truncated_generation_is_not_mislabeled_as_complete():
    generate, _, _ = fake_generator(terminated=False)
    with pytest.raises(labels.LabelGenerationError, match="increase --max-new-tokens"):
        generate(["trusted", "attack target"])


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_round_trip_preserves_metadata_and_refuses_overwrite(tmp_path, suffix):
    path = tmp_path / f"labeled{suffix}"
    rows = [record(0), record(1)]
    labels.write_records(rows, path)
    assert labels.load_records(path) == rows
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        labels.write_records([record(2)], path)
    assert path.read_bytes() == first
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_output_is_rejected_before_loading_model(tmp_path, monkeypatch):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([record()]), encoding="utf-8")

    def forbidden_load(args):
        pytest.fail("must not load a model for an existing destination")

    monkeypatch.setattr(labels, "TransformersLabelGenerator", forbidden_load)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        labels.main(["--model", "example/base", "--input", str(source), "--output", str(source)])


def test_missing_label_requests_are_rejected_instead_of_parsing_attacked_prompt(tmp_path):
    row = record()
    del row["chosen_input"]
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(labels.LabelGenerationError, match="chosen_input"):
        labels.load_records(source)


def test_bad_response_count_and_identical_labels_are_rejected():
    with pytest.raises(labels.LabelGenerationError, match="response count"):
        labels.regenerate_records([record()], lambda requests: ["one"], batch_size=1)
    with pytest.raises(labels.LabelGenerationError, match="identical"):
        labels.regenerate_records([record()], lambda requests: ["same", "same"], batch_size=1)


@pytest.mark.parametrize("multimodal", [False, True])
def test_model_loader_uses_selected_base_and_correct_architecture(monkeypatch, multimodal):
    captured = []
    config = SimpleNamespace(
        is_encoder_decoder=False,
        architectures=["QwenForConditionalGeneration" if multimodal else "LlamaForCausalLM"],
        vision_config={} if multimodal else None,
    )
    tokenizer = SimpleNamespace(
        chat_template="native template", eos_token_id=99, eos_token="<eos>",
        pad_token_id=0, padding_side="right",
    )

    def loader(name):
        def from_pretrained(model_id, **kwargs):
            captured.append((name, model_id, kwargs))
            if name == "config":
                return config
            if name == "tokenizer":
                return tokenizer
            model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99, 100]))
            model.eval = lambda: model
            return model
        return SimpleNamespace(from_pretrained=from_pretrained)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=loader("config"), AutoTokenizer=loader("tokenizer"),
        AutoModelForCausalLM=loader("causal"), AutoModelForImageTextToText=loader("multimodal"),
        set_seed=lambda seed: None,
    ))
    args = labels.parser().parse_args([
        "--model", "selected/base", "--revision", "pinned-revision",
        "--input", "raw.json", "--output", "labels.json",
    ])
    generator = labels.TransformersLabelGenerator(args)
    assert [name for name, _, _ in captured] == ["config", "tokenizer", "multimodal" if multimodal else "causal"]
    assert all(model_id == "selected/base" for _, model_id, _ in captured)
    assert all(kwargs["revision"] == "pinned-revision" for _, _, kwargs in captured)
    assert all(kwargs["trust_remote_code"] is False for _, _, kwargs in captured)
    assert tokenizer.padding_side == "left"
    assert generator.eos_ids == [99, 100]


def test_cli_writes_model_specific_labels_and_preserves_raw_fields(tmp_path, monkeypatch):
    source = tmp_path / "raw.json"
    destination = tmp_path / "labeled.jsonl"
    source.write_text(json.dumps([record()]), encoding="utf-8")
    original = source.read_bytes()

    def fake_loader(args):
        assert args.model == "selected/base"
        return lambda requests: [f"{request}<eos>" for request in requests]

    monkeypatch.setattr(labels, "TransformersLabelGenerator", fake_loader)
    assert labels.main([
        "--model", "selected/base", "--input", str(source), "--output", str(destination),
    ]) == 0
    written = labels.load_records(destination)[0]
    assert written == {
        **record(), "chosen": record()["chosen_input"] + "<eos>",
        "rejected": record()["rejected_input"] + "<eos>",
    }
    assert source.read_bytes() == original
