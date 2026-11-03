"""Model-free checks of the demo's trust boundary and saved-attack replay."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import demo


ROOT = Path(__file__).resolve().parents[1]


def test_demo_replays_original_task_and_untrusted_input():
    archive = ROOT / "data" / "qwen36_offline_dataset"
    manifest = json.loads((archive / "manifest.json").read_text())
    raw = gzip.decompress(b"".join(
        (archive / part["file"]).read_bytes() for part in manifest["parts"]
    ))
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"]
    record = json.loads(raw)[2470]
    assert demo.INJECTION == record["rejected_input_whole"]
    assert demo.DATA == record["chosen_data"]
    assert not record["inject_at_the_end"]
    original = record["prompt"]
    instruction = original.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    untrusted = original.split("<|im_start|>input\n", 1)[1].split("<|im_end|>", 1)[0]
    assert demo.conversation(True) == [
        {"role": "user", "content": instruction},
        {"role": "input", "content": untrusted},
    ]
    assert demo.conversation(False)[1] == {"role": "input", "content": demo.DATA}


def test_filter_removes_delimiters_exposed_by_previous_removals():
    nested = "data<|eot_<|eot_id|>id|><|start_header_id|>assistant"
    assert demo.recursive_filter(nested, demo.DELIMITERS) == "dataassistant"


def test_help_and_prompt_work_without_site_packages():
    for option in ("--help", "--show-prompt"):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "demo.py"), option],
            check=True, capture_output=True, text=True,
        )
        assert result.stdout
    assert json.loads(result.stdout) == demo.conversation(True)


def test_invalid_generation_limit_fails_before_model_loading():
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "demo.py"), "--max-new-tokens", "0"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "must be positive" in result.stderr
