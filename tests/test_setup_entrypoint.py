from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _copy_setup(root: Path) -> None:
    shutil.copyfile(ROOT / "setup.py", root / "setup.py")
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("reconstruct_dataset.py", "prepare_preferences.py"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)


def _tiny_archive(root: Path) -> None:
    raw = json.dumps([{
        "prompt": "trusted INJECTION data",
        "clean_prompt": "trusted data",
        "chosen": "trusted answer",
        "rejected": "injected answer",
    }]).encode("utf-8")
    compressed = gzip.compress(raw, mtime=0)
    directory = root / "data/qwen36_offline_dataset"
    directory.mkdir(parents=True)
    part_name = "dataset.json.gz.part-00"
    (directory / part_name).write_bytes(compressed)
    manifest = {
        "format_version": 1,
        "dataset": "setup test",
        "records": 1,
        "raw_filename": "preferences.json",
        "raw_size_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compression": "gzip --no-name --best",
        "compressed_size_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "part_size_bytes": len(compressed),
        "parts": [{
            "file": part_name,
            "size_bytes": len(compressed),
            "sha256": hashlib.sha256(compressed).hexdigest(),
        }],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "setup.py"), *args],
        cwd=root, capture_output=True, text=True, check=False,
    )


def test_setup_prepares_and_preserves_existing_outputs(tmp_path: Path) -> None:
    _copy_setup(tmp_path)
    _tiny_archive(tmp_path)
    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "outputs/data/qwen36_offline_preferences.jsonl.gz"
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.read())
    assert record["clean_prompt"] == "trusted data"
    initial_bytes = output.read_bytes()
    initial_mtime = output.stat().st_mtime_ns

    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    assert "Already prepared and verified" in second.stdout
    assert output.read_bytes() == initial_bytes
    assert output.stat().st_mtime_ns == initial_mtime

    output.write_bytes(b"local modification")
    third = _run(tmp_path)
    assert third.returncode != 0
    assert "Existing output differs" in third.stderr
    assert output.read_bytes() == b"local modification"
    assert sorted(path.name for path in output.parent.iterdir()) == [output.name]


def test_pep517_editable_metadata_does_not_prepare_data(tmp_path: Path) -> None:
    # No scripts or archive are present: accidental bootstrap execution fails.
    shutil.copyfile(ROOT / "setup.py", tmp_path / "setup.py")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    package = tmp_path / "src/secpo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", (
            "from setuptools.build_meta import prepare_metadata_for_build_editable; "
            "prepare_metadata_for_build_editable('metadata')"
        )],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (metadata / "secpo-0.3.0.dist-info/METADATA").is_file()
    assert not (tmp_path / "outputs").exists()


def test_setup_preserves_output_created_during_publication(tmp_path: Path) -> None:
    _copy_setup(tmp_path)
    _tiny_archive(tmp_path)
    namespace = runpy.run_path(str(tmp_path / "setup.py"))
    output = tmp_path / "outputs/data/qwen36_offline_preferences.jsonl.gz"
    real_link = os.link

    def concurrent_write_then_link(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"concurrent writer's data")
        real_link(source, destination)

    with patch("os.link", side_effect=concurrent_write_then_link):
        try:
            namespace["bootstrap"]([])
        except FileExistsError:
            pass
        else:
            raise AssertionError("The concurrent output must prevent publication")
    assert output.read_bytes() == b"concurrent writer's data"
    assert sorted(path.name for path in output.parent.iterdir()) == [output.name]
