#!/usr/bin/env python3
"""Prepare SecPO data, or delegate packaging commands to setuptools.

Run ``python setup.py`` for the bundled Qwen training data. Build frontends
continue to use the package metadata and dependencies in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/data/qwen36_offline_preferences.jsonl.gz"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Verify and prepare the bundled Qwen SecPO preference data. "
            "This command does not download models or generate responses."
        ),
        epilog=(
            "For another base model, generate its own response labels; see "
            "docs/TRAINING.md#generate-model-specific-response-labels. "
            "See REPRODUCIBILITY.md#data-construction for the archived Qwen "
            "label provenance. Packaging commands such as "
            "egg_info and bdist_wheel are delegated to setuptools."
        ),
    )
    result.add_argument(
        "--output", type=Path,
        help=(
            "Destination .jsonl or .jsonl.gz. "
            "Default: outputs/data/qwen36_offline_preferences.jsonl.gz."
        ),
    )
    return result


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_script(name: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *arguments],
        check=True,
    )


def bootstrap(arguments: list[str]) -> int:
    cli = parser()
    args = cli.parse_args(arguments)
    destination = (args.output or DEFAULT_OUTPUT).absolute()
    if not str(destination).endswith((".jsonl", ".jsonl.gz")):
        cli.error("--output must end in .jsonl or .jsonl.gz")
    if destination.is_symlink():
        cli.error("--output must not be a symbolic link")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Prepare beside the destination for atomic publication. Compare existing
    # output and use a hard link so concurrent writers are never overwritten.
    with tempfile.TemporaryDirectory(
        prefix=".secpo-setup-", dir=destination.parent,
    ) as temporary:
        directory = Path(temporary)
        manifest_path = ROOT / "data/qwen36_offline_dataset/manifest.json"
        source = directory / "preferences.raw.json"
        run_script(
            "reconstruct_dataset.py", "--manifest", str(manifest_path),
            "--output", str(source),
        )
        expected_records = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["records"]
        prepared = directory / destination.name
        run_script(
            "prepare_preferences.py",
            "--input", str(source), "--output", str(prepared),
            "--expected-records", str(expected_records),
        )

        if destination.exists():
            if not destination.is_file() or sha256(destination) != sha256(prepared):
                raise ValueError(
                    f"Existing output differs; choose a new --output: {destination}"
                )
            print(f"Already prepared and verified: {destination}")
        else:
            os.link(prepared, destination)
            print(f"Prepared training data: {destination}")

    print(
        "For another base model, generate its own response labels: "
        "docs/TRAINING.md#generate-model-specific-response-labels.\n"
        "Archived Qwen label provenance: "
        "REPRODUCIBILITY.md#data-construction."
    )
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if (
        not arguments
        or arguments[0] in {"bootstrap", "-h", "--help"}
        or arguments[0].split("=", 1)[0] == "--output"
    ):
        if arguments and arguments[0] == "bootstrap":
            arguments = arguments[1:]
        try:
            raise SystemExit(bootstrap(arguments))
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"error: {error}") from None
    else:
        # setuptools.build_meta executes this file for editable installs,
        # metadata, source distributions, and wheels. Never prepare data there.
        from setuptools import setup

        setup()
