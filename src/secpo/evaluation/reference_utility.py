"""Exact, provider-neutral AlpacaEval2 and SEP utility orchestration.

The comparator outputs are external data.  This module verifies their full
SHA-256 digest, row count, generator label, and row-by-row alignment before it
builds an ``alpaca_eval`` command.  It contains no provider client, endpoint,
credential, or private path.  The evaluator configuration is supplied at
runtime by CLI option or environment variable.

Both protocols report raw pairwise win rate with ``get_winrate``:

* AlpacaEval2 evaluates 805 examples against the
  research snapshot's fixed GPT-4o-2024-05-13 comparator outputs.
* SEP Utility evaluates 9,160 examples against
  Meta-Llama-3-8B-Instruct comparator outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from secpo.subprocess_env import build_subprocess_environment


CONFIG_SCHEMA_VERSION = "secpo-reference-utility-v1"
SUPPORTED_SUITES = ("alpacaeval2", "sep")
DEFAULT_ANNOTATORS_CONFIG_ENV = "SECPO_UTILITY_ANNOTATORS_CONFIG"

_CONFIG_KEYS = {
    "schema_version",
    "suite",
    "expected_rows",
    "identity_keys",
    "reference",
    "metric",
    "alpaca_eval",
}
_REFERENCE_KEYS = {
    "expected_filename",
    "sha256",
    "generator",
    "redistribution",
}
_METRIC_KEYS = {"primary", "fn_metric", "sort_by", "metric_kwargs"}
_ALPACA_EVAL_KEYS = {
    "annotators_config_env",
    "annotation_kwargs",
    "input_keys",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ReferenceUtilityError(RuntimeError):
    """Raised when an evaluation input violates the frozen protocol."""


def _require_exact_keys(
    value: Mapping[str, Any], *, expected: set[str], label: str
) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise ValueError(f"{label} has " + "; ".join(details))


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty NUL-free string")
    return value.strip()


@dataclass(frozen=True)
class ReferenceSpec:
    expected_filename: str
    sha256: str
    generator: str
    redistribution: str


@dataclass(frozen=True)
class MetricSpec:
    primary: str
    fn_metric: str | None
    sort_by: str | None
    metric_kwargs: Mapping[str, Any] | None


@dataclass(frozen=True)
class AlpacaEvalSpec:
    annotators_config_env: str
    annotation_kwargs: Mapping[str, Any]
    input_keys: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceUtilityConfig:
    suite: str
    expected_rows: int
    identity_keys: tuple[str, ...]
    reference: ReferenceSpec
    metric: MetricSpec
    alpaca_eval: AlpacaEvalSpec


@dataclass(frozen=True)
class PreflightReport:
    suite: str
    expected_rows: int
    candidate_path: str
    candidate_sha256: str
    candidate_generator: str
    reference_path: str
    reference_sha256: str
    reference_generator: str
    identity_keys: tuple[str, ...]
    primary_metric: str


def parse_reference_utility_config(value: Any) -> ReferenceUtilityConfig:
    """Strictly parse one frozen utility configuration."""

    if not isinstance(value, Mapping):
        raise ValueError("reference utility config must be a JSON object")
    _require_exact_keys(value, expected=_CONFIG_KEYS, label="config")
    if value["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {value['schema_version']!r}")

    suite = _nonempty_string(value["suite"], label="suite")
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"suite must be one of {SUPPORTED_SUITES}")
    expected_rows = value["expected_rows"]
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive integer")

    identity_value = value["identity_keys"]
    if not isinstance(identity_value, list) or not identity_value:
        raise ValueError("identity_keys must be a non-empty JSON array")
    identity_keys = tuple(
        _nonempty_string(item, label=f"identity_keys[{index}]")
        for index, item in enumerate(identity_value)
    )
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("identity_keys must be unique")

    reference_value = value["reference"]
    if not isinstance(reference_value, Mapping):
        raise ValueError("reference must be a JSON object")
    _require_exact_keys(
        reference_value, expected=_REFERENCE_KEYS, label="reference"
    )
    expected_filename = _nonempty_string(
        reference_value["expected_filename"], label="reference.expected_filename"
    )
    if Path(expected_filename).name != expected_filename:
        raise ValueError("reference.expected_filename must be one basename")
    reference_sha256 = _nonempty_string(
        reference_value["sha256"], label="reference.sha256"
    )
    if not _SHA256_RE.fullmatch(reference_sha256):
        raise ValueError("reference.sha256 must be 64 lowercase hexadecimal digits")
    reference = ReferenceSpec(
        expected_filename=expected_filename,
        sha256=reference_sha256,
        generator=_nonempty_string(
            reference_value["generator"], label="reference.generator"
        ),
        redistribution=_nonempty_string(
            reference_value["redistribution"], label="reference.redistribution"
        ),
    )
    if reference.redistribution != "external":
        raise ValueError("reference.redistribution must remain 'external'")

    metric_value = value["metric"]
    if not isinstance(metric_value, Mapping):
        raise ValueError("metric must be a JSON object")
    _require_exact_keys(metric_value, expected=_METRIC_KEYS, label="metric")
    primary = _nonempty_string(metric_value["primary"], label="metric.primary")
    for optional_key in ("fn_metric", "sort_by"):
        optional_value = metric_value[optional_key]
        if optional_value is not None and not isinstance(optional_value, str):
            raise ValueError(f"metric.{optional_key} must be a string or null")
    metric_kwargs = metric_value["metric_kwargs"]
    if metric_kwargs is not None and not isinstance(metric_kwargs, Mapping):
        raise ValueError("metric.metric_kwargs must be an object or null")
    metric = MetricSpec(
        primary=primary,
        fn_metric=metric_value["fn_metric"],
        sort_by=metric_value["sort_by"],
        metric_kwargs=dict(metric_kwargs) if metric_kwargs is not None else None,
    )

    alpaca_value = value["alpaca_eval"]
    if not isinstance(alpaca_value, Mapping):
        raise ValueError("alpaca_eval must be a JSON object")
    _require_exact_keys(
        alpaca_value, expected=_ALPACA_EVAL_KEYS, label="alpaca_eval"
    )
    env_name = _nonempty_string(
        alpaca_value["annotators_config_env"],
        label="alpaca_eval.annotators_config_env",
    )
    if not _ENV_RE.fullmatch(env_name):
        raise ValueError("alpaca_eval.annotators_config_env is not a safe env name")
    annotation_kwargs = alpaca_value["annotation_kwargs"]
    if not isinstance(annotation_kwargs, Mapping):
        raise ValueError("alpaca_eval.annotation_kwargs must be an object")
    input_keys_value = alpaca_value["input_keys"]
    if not isinstance(input_keys_value, list) or not input_keys_value:
        raise ValueError("alpaca_eval.input_keys must be a non-empty array")
    input_keys = tuple(
        _nonempty_string(item, label=f"alpaca_eval.input_keys[{index}]")
        for index, item in enumerate(input_keys_value)
    )
    alpaca_eval = AlpacaEvalSpec(
        annotators_config_env=env_name,
        annotation_kwargs=dict(annotation_kwargs),
        input_keys=input_keys,
    )

    expected_protocols = {
        "alpacaeval2": {
            "rows": 805,
            "identity": ("instruction",),
            "reference": ReferenceSpec(
                expected_filename=(
                    "alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json"
                ),
                sha256=(
                    "653c131d58e8d5802203c85e80ca3fb4b378f22e2a0280ad5cd910ff0bee9535"
                ),
                generator="gpt-4o-2024-05-13",
                redistribution="external",
            ),
            "primary": "win_rate",
            "fn_metric": "get_winrate",
            "sort_by": "win_rate",
            "metric_kwargs": None,
        },
        "sep": {
            "rows": 9160,
            "identity": ("instruction", "instruction_only", "input", "injection"),
            "reference": ReferenceSpec(
                expected_filename=(
                    "SEP_dataset_test_Meta-Llama-3-8B-Instruct copy.json"
                ),
                sha256=(
                    "6defcfdb3ac9da4a03eff21403a0ac58ce52c0c1dc45c23fc48b17a2bba0808c"
                ),
                generator="meta-llama/Meta-Llama-3-8B-Instruct",
                redistribution="external",
            ),
            "primary": "win_rate",
            "fn_metric": "get_winrate",
            "sort_by": "win_rate",
            "metric_kwargs": None,
        },
    }
    expected = expected_protocols[suite]
    actual = {
        "rows": expected_rows,
        "identity": identity_keys,
        "reference": reference,
        "primary": metric.primary,
        "fn_metric": metric.fn_metric,
        "sort_by": metric.sort_by,
        "metric_kwargs": metric.metric_kwargs,
    }
    if actual != expected:
        raise ValueError(f"{suite} protocol fields do not match the frozen defaults")
    if alpaca_eval.input_keys != identity_keys:
        raise ValueError("alpaca_eval.input_keys must equal identity_keys")
    if alpaca_eval.annotation_kwargs != {"is_ordered": True}:
        raise ValueError("annotation_kwargs must freeze ordered comparison")
    if alpaca_eval.annotators_config_env != DEFAULT_ANNOTATORS_CONFIG_ENV:
        raise ValueError(
            f"annotators config env must be {DEFAULT_ANNOTATORS_CONFIG_ENV}"
        )

    return ReferenceUtilityConfig(
        suite=suite,
        expected_rows=expected_rows,
        identity_keys=identity_keys,
        reference=reference,
        metric=metric,
        alpaca_eval=alpaca_eval,
    )


def load_reference_utility_config(path: str | Path) -> ReferenceUtilityConfig:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read utility config {source}: {error}") from error
    return parse_reference_utility_config(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReferenceUtilityError(f"missing {label}: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReferenceUtilityError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, list):
        raise ReferenceUtilityError(f"{label} must be a JSON array: {path}")
    if any(not isinstance(row, dict) for row in value):
        raise ReferenceUtilityError(f"every {label} row must be a JSON object")
    return value


def _row_identity(
    row: Mapping[str, Any],
    *,
    keys: Sequence[str],
    label: str,
    index: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        value = row.get(key)
        if not isinstance(value, str):
            raise ReferenceUtilityError(
                f"{label} row {index} has no string {key!r}"
            )
        values.append(value)
    return tuple(values)


def _validate_canonical_sep_instruction(
    row: Mapping[str, Any], *, label: str, index: int
) -> None:
    instruction = row["instruction"]
    instruction_only = row["instruction_only"]
    input_data = row["input"]
    expected = instruction_only if input_data == "" else instruction_only + "\n\n" + input_data
    if instruction != expected:
        raise ReferenceUtilityError(
            f"{label} row {index} is not canonical: instruction must equal "
            "instruction_only plus optional blank line and input"
        )


def preflight_reference_utility(
    config: ReferenceUtilityConfig,
    *,
    candidate_path: str | Path,
    reference_path: str | Path,
) -> PreflightReport:
    """Validate exact comparator identity and candidate/reference alignment."""

    candidate = Path(candidate_path).expanduser().resolve()
    reference = Path(reference_path).expanduser().resolve()
    reference_hash = sha256_file(reference) if reference.is_file() else ""
    if reference_hash != config.reference.sha256:
        raise ReferenceUtilityError(
            "reference SHA-256 mismatch: expected "
            f"{config.reference.sha256}, got {reference_hash or 'missing'}"
        )

    candidate_rows = _load_rows(candidate, label="candidate outputs")
    reference_rows = _load_rows(reference, label="reference outputs")
    for label, rows in (
        ("candidate outputs", candidate_rows),
        ("reference outputs", reference_rows),
    ):
        if len(rows) != config.expected_rows:
            raise ReferenceUtilityError(
                f"{label} has {len(rows)} rows; expected {config.expected_rows}"
            )

    candidate_identities: list[tuple[str, ...]] = []
    reference_identities: list[tuple[str, ...]] = []
    candidate_generators: set[str] = set()
    for label, rows, identities, require_reference_generator in (
        ("candidate", candidate_rows, candidate_identities, False),
        ("reference", reference_rows, reference_identities, True),
    ):
        for index, row in enumerate(rows):
            identity = _row_identity(
                row,
                keys=config.identity_keys,
                label=label,
                index=index,
            )
            identities.append(identity)
            output = row.get("output")
            if not isinstance(output, str):
                raise ReferenceUtilityError(
                    f"{label} row {index} has no string 'output'"
                )
            generator = row.get("generator")
            if not isinstance(generator, str) or not generator.strip():
                raise ReferenceUtilityError(
                    f"{label} row {index} has no non-empty generator"
                )
            if require_reference_generator:
                if generator != config.reference.generator:
                    raise ReferenceUtilityError(
                        f"reference row {index} generator is {generator!r}; expected "
                        f"{config.reference.generator!r}"
                    )
            else:
                candidate_generators.add(generator)
            if config.suite == "sep":
                _validate_canonical_sep_instruction(row, label=label, index=index)

    if len(set(candidate_identities)) != config.expected_rows:
        raise ReferenceUtilityError("candidate identities are not unique")
    if len(set(reference_identities)) != config.expected_rows:
        raise ReferenceUtilityError("reference identities are not unique")
    if candidate_identities != reference_identities:
        raise ReferenceUtilityError(
            "candidate and reference identities are not aligned in identical order"
        )
    if len(candidate_generators) != 1:
        raise ReferenceUtilityError(
            f"candidate outputs must have one generator, got {sorted(candidate_generators)}"
        )

    return PreflightReport(
        suite=config.suite,
        expected_rows=config.expected_rows,
        candidate_path=str(candidate),
        candidate_sha256=sha256_file(candidate),
        candidate_generator=next(iter(candidate_generators)),
        reference_path=str(reference),
        reference_sha256=reference_hash,
        reference_generator=config.reference.generator,
        identity_keys=config.identity_keys,
        primary_metric=config.metric.primary,
    )


def build_alpaca_eval_command(
    config: ReferenceUtilityConfig,
    report: PreflightReport,
    *,
    output_dir: str | Path,
    annotators_config: str,
    alpaca_eval_bin: str = "alpaca_eval",
    cache_path: str | Path | None = None,
) -> list[str]:
    """Build the argument-vector command after a successful preflight."""

    if not isinstance(annotators_config, str) or not annotators_config.strip():
        raise ValueError("annotators_config must be supplied at runtime")
    if any(character in annotators_config for character in "\x00\r\n"):
        raise ValueError("annotators_config must not contain NUL or newlines")
    if not isinstance(alpaca_eval_bin, str) or not alpaca_eval_bin.strip():
        raise ValueError("alpaca_eval_bin must be non-empty")
    output = Path(output_dir).expanduser().resolve()
    command = [
        alpaca_eval_bin,
        "--is_overwrite_leaderboard",
        "--annotators_config",
        annotators_config,
        "--reference_outputs",
        report.reference_path,
        "--model_outputs",
        report.candidate_path,
        "--output_path",
        str(output),
        "--annotation_kwargs",
        repr(dict(config.alpaca_eval.annotation_kwargs)),
        "--input_keys",
        repr(list(config.alpaca_eval.input_keys)),
        "--is_store_missing_annotations",
        "False",
        "--precomputed_leaderboard",
        "None",
    ]
    if cache_path is not None:
        command.extend(
            ("--caching_path", str(Path(cache_path).expanduser().resolve()))
        )
    if config.metric.metric_kwargs is not None:
        command.extend(
            ("--metric_kwargs", repr(dict(config.metric.metric_kwargs)))
        )
    if config.metric.fn_metric is not None:
        command.extend(("--fn_metric", config.metric.fn_metric))
    if config.metric.sort_by is not None:
        command.extend(("--sort_by", config.metric.sort_by))
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--alpaca-eval-bin", default="alpaca_eval")
    parser.add_argument(
        "--annotators-config",
        help=(
            "Runtime AlpacaEval evaluator name or config path. Defaults to "
            f"${DEFAULT_ANNOTATORS_CONFIG_ENV}; no provider is baked in."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute after preflight. Without this flag, print the command only.",
    )
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Pass exactly this parent environment variable to AlpacaEval; "
            "repeat for additional names. No credentials are passed by default."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = load_reference_utility_config(args.config)
    annotators_config = args.annotators_config or os.getenv(
        config.alpaca_eval.annotators_config_env, ""
    )
    if not annotators_config.strip():
        parser.error(
            "provide --annotators-config or set "
            f"{config.alpaca_eval.annotators_config_env}"
        )
    report = preflight_reference_utility(
        config,
        candidate_path=args.candidate,
        reference_path=args.reference,
    )
    command = build_alpaca_eval_command(
        config,
        report,
        output_dir=args.output_dir,
        annotators_config=annotators_config,
        alpaca_eval_bin=args.alpaca_eval_bin,
        cache_path=args.cache_path,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(shlex.join(command))
    if not args.execute:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_path is not None:
        args.cache_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        command,
        check=True,
        env=build_subprocess_environment(
            pass_env=tuple(args.pass_env),
            require_pass_env=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
