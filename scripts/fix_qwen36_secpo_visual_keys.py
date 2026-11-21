"""Build a servable copy of Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO at revision 726da6b.

That revision names the 333 vision-tower tensors model.language_model.visual.*; the base Qwen3.6 and the
MetaSecAlign upload use model.visual.*. Every other file is symlinked unchanged; only the visual
tensors' names are rewritten, and their bytes and the final key set are verified.
Usage: python fix_qwen36_secpo_visual_keys.py SRC_SNAPSHOT REFERENCE_SNAPSHOT DST_DIR
"""
import json
import os
import sys
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

BAD = "model.language_model.visual."
GOOD = "model.visual."

src, ref, dst = map(Path, sys.argv[1:])
dst.mkdir(parents=True, exist_ok=False)

index = json.loads((src / "model.safetensors.index.json").read_text())
bad_shards = {shard for key, shard in index["weight_map"].items() if key.startswith(BAD)}
if len(bad_shards) != 1:
    raise RuntimeError(f"expected visual tensors in one shard, got {bad_shards}")
bad_shard = bad_shards.pop()

for entry in src.iterdir():
    if entry.name not in {bad_shard, "model.safetensors.index.json"}:
        os.symlink(entry.resolve(), dst / entry.name)

with safe_open(src / bad_shard, framework="pt") as f:
    metadata = f.metadata()
    tensors = {(GOOD + k[len(BAD):] if k.startswith(BAD) else k): f.get_tensor(k) for k in f.keys()}
save_file(tensors, dst / bad_shard, metadata=metadata)

index["weight_map"] = {(GOOD + k[len(BAD):] if k.startswith(BAD) else k): v for k, v in index["weight_map"].items()}
(dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

# Verify: renamed shard has the same bytes per tensor, and the key set matches the reference upload.
with safe_open(src / bad_shard, framework="pt") as old, safe_open(dst / bad_shard, framework="pt") as new:
    if old.metadata() != new.metadata():
        raise RuntimeError("shard metadata changed")
    for k in old.keys():
        nk = GOOD + k[len(BAD):] if k.startswith(BAD) else k
        a, b = old.get_tensor(k), new.get_tensor(nk)
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
            raise RuntimeError(f"tensor changed: {k}")
ref_keys = set(json.loads((ref / "model.safetensors.index.json").read_text())["weight_map"])
if set(index["weight_map"]) != ref_keys:
    raise RuntimeError(f"key set differs from reference: {sorted(set(index['weight_map']) ^ ref_keys)[:10]}")
renamed = sum(k.startswith(GOOD) for k in index["weight_map"])
print(f"FIX_OK shard={bad_shard} renamed={renamed} total_keys={len(index['weight_map'])}")
