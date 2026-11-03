from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import secpo.evaluation.reference_utility as reference_utility
from secpo.evaluation.reference_utility import (
    PreflightReport,
    ReferenceUtilityError,
    build_alpaca_eval_command,
    load_reference_utility_config,
    parse_reference_utility_config,
    preflight_reference_utility,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_config(parsed, *, rows: int, reference_hash: str):
    from dataclasses import replace

    return replace(
        parsed,
        expected_rows=rows,
        reference=replace(parsed.reference, sha256=reference_hash),
    )


def test_committed_configs_freeze_exact_references_and_metrics() -> None:
    alpaca = load_reference_utility_config(
        ROOT / "configs/evaluate_alpacaeval2_utility.json"
    )
    sep = load_reference_utility_config(ROOT / "configs/evaluate_sep_utility.json")

    assert alpaca.expected_rows == 805
    assert alpaca.reference.generator == "gpt-4o-2024-05-13"
    assert alpaca.reference.sha256 == (
        "653c131d58e8d5802203c85e80ca3fb4b378f22e2a0280ad5cd910ff0bee9535"
    )
    assert alpaca.metric.primary == "win_rate"
    assert alpaca.metric.fn_metric == "get_winrate"
    assert alpaca.metric.sort_by == "win_rate"
    assert alpaca.metric.metric_kwargs is None

    assert sep.expected_rows == 9160
    assert sep.reference.generator == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert sep.reference.sha256 == (
        "6defcfdb3ac9da4a03eff21403a0ac58ce52c0c1dc45c23fc48b17a2bba0808c"
    )
    assert sep.metric.primary == "win_rate"
    assert sep.metric.fn_metric == "get_winrate"
    assert sep.metric.sort_by == "win_rate"
    assert sep.metric.metric_kwargs is None


@pytest.mark.parametrize("suite", ["sep", "alpacaeval2"])
def test_parser_rejects_length_control_in_raw_win_rate_protocol(suite: str) -> None:
    value = json.loads(
        (ROOT / f"configs/evaluate_{suite}_utility.json").read_text(encoding="utf-8")
    )
    value["metric"] = {
        "primary": "length_controlled_winrate",
        "fn_metric": None,
        "sort_by": None,
        "metric_kwargs": {"save_weights_dir": None},
    }
    with pytest.raises(ValueError, match="frozen defaults"):
        parse_reference_utility_config(value)


def test_parser_rejects_alternate_comparator_identity() -> None:
    value = json.loads(
        (ROOT / "configs/evaluate_alpacaeval2_utility.json").read_text(
            encoding="utf-8"
        )
    )
    value["reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen defaults"):
        parse_reference_utility_config(value)


def test_preflight_verifies_hash_generator_and_order(tmp_path: Path) -> None:
    reference_rows = [
        {"instruction": "one", "output": "r1", "generator": "gpt-4o-2024-05-13"},
        {"instruction": "two", "output": "r2", "generator": "gpt-4o-2024-05-13"},
    ]
    candidate_rows = [
        {"instruction": "one", "output": "c1", "generator": "candidate/model"},
        {"instruction": "two", "output": "c2", "generator": "candidate/model"},
    ]
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference_hash = _write(reference, reference_rows)
    _write(candidate, candidate_rows)
    parsed = load_reference_utility_config(
        ROOT / "configs/evaluate_alpacaeval2_utility.json"
    )
    config = _small_config(parsed, rows=2, reference_hash=reference_hash)

    report = preflight_reference_utility(
        config, candidate_path=candidate, reference_path=reference
    )
    assert report.candidate_generator == "candidate/model"
    assert report.reference_sha256 == reference_hash

    reordered = list(reversed(candidate_rows))
    _write(candidate, reordered)
    with pytest.raises(ReferenceUtilityError, match="identical order"):
        preflight_reference_utility(
            config, candidate_path=candidate, reference_path=reference
        )


def test_preflight_rejects_reference_drift(tmp_path: Path) -> None:
    rows = [
        {"instruction": "one", "output": "r1", "generator": "wrong-reference"},
        {"instruction": "two", "output": "r2", "generator": "wrong-reference"},
    ]
    candidate_rows = [
        {"instruction": "one", "output": "c1", "generator": "candidate/model"},
        {"instruction": "two", "output": "c2", "generator": "candidate/model"},
    ]
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference_hash = _write(reference, rows)
    _write(candidate, candidate_rows)
    parsed = load_reference_utility_config(
        ROOT / "configs/evaluate_alpacaeval2_utility.json"
    )
    config = _small_config(parsed, rows=2, reference_hash=reference_hash)

    with pytest.raises(ReferenceUtilityError, match="generator"):
        preflight_reference_utility(
            config, candidate_path=candidate, reference_path=reference
        )

    config = _small_config(parsed, rows=2, reference_hash="f" * 64)
    with pytest.raises(ReferenceUtilityError, match="SHA-256 mismatch"):
        preflight_reference_utility(
            config, candidate_path=candidate, reference_path=reference
        )


def test_commands_use_raw_win_rate_for_both_suites(tmp_path: Path) -> None:
    alpaca = load_reference_utility_config(
        ROOT / "configs/evaluate_alpacaeval2_utility.json"
    )
    sep = load_reference_utility_config(ROOT / "configs/evaluate_sep_utility.json")
    from secpo.evaluation.reference_utility import PreflightReport

    def report(config):
        return PreflightReport(
            suite=config.suite,
            expected_rows=config.expected_rows,
            candidate_path=str(tmp_path / "candidate.json"),
            candidate_sha256="a" * 64,
            candidate_generator="candidate/model",
            reference_path=str(tmp_path / "reference.json"),
            reference_sha256=config.reference.sha256,
            reference_generator=config.reference.generator,
            identity_keys=config.identity_keys,
            primary_metric=config.metric.primary,
        )

    alpaca_command = build_alpaca_eval_command(
        alpaca,
        report(alpaca),
        output_dir=tmp_path / "alpaca",
        annotators_config="runtime-config",
    )
    assert "--metric_kwargs" not in alpaca_command
    assert alpaca_command[alpaca_command.index("--fn_metric") + 1] == "get_winrate"
    assert alpaca_command[alpaca_command.index("--sort_by") + 1] == "win_rate"
    assert "{'is_ordered': True}" in alpaca_command

    sep_command = build_alpaca_eval_command(
        sep,
        report(sep),
        output_dir=tmp_path / "sep",
        annotators_config="runtime-config",
    )
    assert sep_command[sep_command.index("--fn_metric") + 1] == "get_winrate"
    assert sep_command[sep_command.index("--sort_by") + 1] == "win_rate"
    assert "--metric_kwargs" not in sep_command


def test_execute_passes_only_explicit_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_reference_utility_config(
        ROOT / "configs/evaluate_sep_utility.json"
    )
    report = PreflightReport(
        suite=config.suite,
        expected_rows=config.expected_rows,
        candidate_path=str(tmp_path / "candidate.json"),
        candidate_sha256="a" * 64,
        candidate_generator="candidate/model",
        reference_path=str(tmp_path / "reference.json"),
        reference_sha256=config.reference.sha256,
        reference_generator=config.reference.generator,
        identity_keys=config.identity_keys,
        primary_metric=config.metric.primary,
    )
    monkeypatch.setattr(
        reference_utility,
        "preflight_reference_utility",
        lambda *args, **kwargs: report,
    )
    monkeypatch.setattr(
        reference_utility,
        "build_alpaca_eval_command",
        lambda *args, **kwargs: ["alpaca_eval", "evaluate"],
    )
    monkeypatch.setenv("APPROVED_JUDGE_TOKEN", "approved-value")
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-nvidia-key")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "unrelated-password")
    observed = []
    monkeypatch.setattr(
        reference_utility.subprocess,
        "run",
        lambda command, **kwargs: observed.append((command, kwargs["env"])),
    )

    base_args = [
        "--config",
        str(ROOT / "configs/evaluate_sep_utility.json"),
        "--candidate",
        str(tmp_path / "candidate.json"),
        "--reference",
        str(tmp_path / "reference.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--annotators-config",
        "runtime-config",
        "--execute",
    ]
    assert reference_utility.main(
        [*base_args, "--pass-env", "APPROVED_JUDGE_TOKEN"]
    ) == 0
    environment = observed[-1][1]
    assert environment["APPROVED_JUDGE_TOKEN"] == "approved-value"
    assert "NVIDIA_API_KEY" not in environment
    assert "UNRELATED_DATABASE_PASSWORD" not in environment

    assert reference_utility.main(base_args) == 0
    assert "APPROVED_JUDGE_TOKEN" not in observed[-1][1]


def test_release_files_contain_no_provider_route_or_secret() -> None:
    paths = [
        ROOT / "configs/evaluate_alpacaeval2_utility.json",
        ROOT / "configs/evaluate_sep_utility.json",
        ROOT / "src/secpo/evaluation/reference_utility.py",
        ROOT / "scripts/evaluate_reference_utility.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).casefold()
    assert "http://" not in text
    assert "https://" not in text
    assert "api_key" not in text
    assert "/" + "lustre" + "/" not in text
