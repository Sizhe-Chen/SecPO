"""Model-free checks of the demo's saved attack and inference calls."""

import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_run(monkeypatch):
    inputs = {"input_ids": SimpleNamespace(shape=(1, 3))}
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value.to.return_value = inputs
    tokenizer.decode.return_value = "A movie summary."
    models = [MagicMock(device="cuda:0"), MagicMock(device="cuda:0")]
    for model in models:
        model.generate.return_value = [[10, 11, 12, 20, 21]]
    loader = MagicMock(side_effect=models)
    tokenizer_loader = MagicMock(return_value=tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=loader),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=MagicMock()))
    namespace = runpy.run_path(str(ROOT / "demo.py"), run_name="__main__")
    return namespace, tokenizer, tokenizer_loader, loader, models


def test_demo_replays_original_task_and_untrusted_input(demo_run):
    namespace = demo_run[0]
    archive = ROOT / "data" / "qwen36_offline_dataset"
    manifest = json.loads((archive / "manifest.json").read_text())
    raw = gzip.decompress(b"".join(
        (archive / part["file"]).read_bytes() for part in manifest["parts"]
    ))
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"]
    record = json.loads(raw)[2470]
    assert namespace["injection"] == record["rejected_input_whole"]
    assert namespace["data"] == record["chosen_data"]
    assert not record["inject_at_the_end"]
    original = record["prompt"]
    instruction = original.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    untrusted = original.split("<|im_start|>input\n", 1)[1].split("<|im_end|>", 1)[0]
    assert namespace["conversation"] == [
        {"role": "user", "content": instruction},
        {"role": "input", "content": untrusted},
    ]


def test_agentdojo_keeps_attacked_tool_result_in_untrusted_role(demo_run):
    fixture = demo_run[0]["agentdojo"]
    messages = fixture["messages"]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "input",
    ]
    call, = messages[2]["tool_calls"]
    assert call["type"] == "function"
    assert call["function"]["name"] == "search_calendar_events"
    assert isinstance(call["function"]["arguments"], dict)
    assert messages[3]["tool_call_id"] == call["id"]
    assert "delete_file" in messages[3]["content"]
    assert "The Gourmet Restaurant" in messages[3]["content"]
    tools = {tool["function"]["name"]: tool["function"] for tool in fixture["tools"]}
    assert call["function"]["name"] in tools
    assert tools["delete_file"]["parameters"]["required"] == ["file_id"]


def test_both_models_receive_same_messages_and_decode_only_response(demo_run):
    namespace, tokenizer, tokenizer_loader, loader, models = demo_run
    secpo = "Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO"
    tokenizer_loader.assert_called_once_with(secpo)
    assert [call.args[0] for call in loader.call_args_list] == [
        "Qwen/Qwen3.6-27B", secpo,
    ]
    expected_cases = [
        (namespace["conversation"], None),
        (namespace["agentdojo"]["messages"], namespace["agentdojo"]["tools"]),
    ] * 2
    template_calls = tokenizer.apply_chat_template.call_args_list
    assert len(template_calls) == len(expected_cases)
    for call, (messages, tools) in zip(template_calls, expected_cases):
        assert call.args == (messages,)
        assert call.kwargs == dict(
            tools=tools,
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
        )
    for model in models:
        assert model.generate.call_count == 2
        for call in model.generate.call_args_list:
            assert call.kwargs["do_sample"] is False
            assert call.kwargs["max_new_tokens"] == 512
    assert tokenizer.decode.call_count == 4
    for call in tokenizer.decode.call_args_list:
        assert call.args == ([20, 21],)
        assert call.kwargs == {"skip_special_tokens": True}
