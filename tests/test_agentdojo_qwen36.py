from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import secpo.evaluation.agentdojo_qwen36 as integration
from secpo.attacks import GeneticConfig
from secpo.evaluation.agentdojo_qwen36 import (
    AgentDojoConfig,
    AgentDojoEpisode,
    load_config,
    paper_genetic_tasks,
    run_genetic,
    verify_agentdojo_checkout,
)


RELEASE_ROOT = Path(__file__).parents[1]
CONFIG = RELEASE_ROOT / "configs" / "evaluate_agentdojo_qwen36_27b.json"
PATCH = RELEASE_ROOT / "third_party" / "patches" / "agentdojo.patch"


def test_public_config_is_pinned_and_contains_no_endpoint() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = load_config(CONFIG)

    assert config.upstream_commit == integration.UPSTREAM_COMMIT
    assert config.benchmark_version == "v1.2.1"
    assert config.base_url_env == "AGENTDOJO_BASE_URL"
    assert "base_url" not in raw["model"]
    loopback = "127" + ".0.0.1"
    assert loopback not in CONFIG.read_text(encoding="utf-8")
    assert raw["adaptive_genetic"]["sample_count"] == 80


def test_endpoint_is_required_and_resolved_only_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    monkeypatch.delenv(config.base_url_env, raising=False)
    with pytest.raises(RuntimeError, match=config.base_url_env):
        config.resolve_base_url()

    monkeypatch.setenv(config.base_url_env, "https://example.invalid/v1")
    assert config.resolve_base_url() == "https://example.invalid/v1"
    embedded_credential = "https://" + "user:" + "secret" + "@example.invalid/v1"
    monkeypatch.setenv(config.base_url_env, embedded_credential)
    with pytest.raises(ValueError, match="credentials"):
        config.resolve_base_url()


def test_config_rejects_embedded_credential_fields() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["model"]["api_key"] = "not-a-real-value"
    with pytest.raises(ValueError, match="credential"):
        AgentDojoConfig.from_mapping(value)


def test_paper_genetic_manifest_is_exactly_80_unique_pairs() -> None:
    tasks = paper_genetic_tasks(goal="perform the injection task")
    identities = {
        (task.suite, task.user_task_id, task.injection_task_id) for task in tasks
    }

    assert len(tasks) == 80
    assert len(identities) == 80
    assert {task.suite for task in tasks} == {"slack", "travel", "workspace"}


def test_genetic_success_requires_exact_agentdojo_checker() -> None:
    task = paper_genetic_tasks(goal="perform the injection task")[0]

    def unused_proposer(request):
        raise AssertionError("exact initial seed should stop the search")

    def episode(task, injection):
        return AgentDojoEpisode(
            response=f"episode for {injection}",
            exact_success=injection == "exact",
            utility=True,
            dense_score=9.0,
        )

    result = run_genetic(
        task=task,
        proposer=unused_proposer,
        episode=episode,
        initial_injections=("high-heuristic", "exact"),
        config=GeneticConfig(
            steps=2,
            candidates_per_step=2,
            num_islands=2,
            query_budget=2,
            early_stop_score=10.0,
        ),
    )

    assert result["pass_at_n"] is True
    assert result["pass_at_n_denominator"] == 2
    assert result["best"]["injection"] == "exact"
    assert result["best"]["score"] == 10.0


def test_dense_score_never_declares_attack_success() -> None:
    task = paper_genetic_tasks(goal="perform the injection task")[0]

    def proposer(request):
        return [f"candidate-{request.step}"]

    result = run_genetic(
        task=task,
        proposer=proposer,
        episode=lambda task, injection: {
            "response": "failed episode",
            "exact_success": False,
            "utility": True,
            "dense_score": 9.0,
        },
        initial_injections=("seed",),
        config=GeneticConfig(
            steps=1,
            candidates_per_step=1,
            num_islands=1,
            query_budget=2,
            early_stop_score=10.0,
        ),
    )

    assert result["pass_at_n"] is False
    assert result["exact_success_candidate_ids"] == []
    assert result["best"]["score"] == 9.0


def test_episode_scores_require_explicit_booleans() -> None:
    with pytest.raises(TypeError, match="explicit booleans"):
        AgentDojoEpisode.from_value(
            {
                "response": "candidate",
                "exact_success": "false",
                "utility": True,
            }
        )
    with pytest.raises(TypeError, match="explicit booleans"):
        AgentDojoEpisode.from_value(
            AgentDojoEpisode(
                response="candidate",
                exact_success=1,
                utility=True,
            )
        )


def test_checkout_verifier_accepts_only_pin_license_and_known_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "agentdojo"
    (checkout / "src" / "agentdojo").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (checkout / "src" / "agentdojo" / "benchmark.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    license_bytes = b"audited MIT fixture\n"
    (checkout / "LICENSE").write_bytes(license_bytes)
    monkeypatch.setattr(
        integration,
        "UPSTREAM_LICENSE_SHA256",
        hashlib.sha256(license_bytes).hexdigest(),
    )

    status = ""

    def fake_git(root, *arguments, check=True):
        if arguments[:2] == ("rev-parse", "HEAD"):
            stdout = integration.UPSTREAM_COMMIT + "\n"
            return subprocess.CompletedProcess([], 0, stdout, "")
        if arguments and arguments[0] == "status":
            return subprocess.CompletedProcess([], 0, status, "")
        raise AssertionError(arguments)

    monkeypatch.setattr(integration, "_git", fake_git)
    verified = verify_agentdojo_checkout(checkout)
    assert verified["pristine"] is True

    status = "?? private.env"
    with pytest.raises(RuntimeError, match="untracked"):
        verify_agentdojo_checkout(checkout, optional_patch=PATCH)


def test_optional_patch_is_minimal_and_identity_neutral() -> None:
    text = PATCH.read_text(encoding="utf-8")
    modified = {
        line.removeprefix("+++ b/")
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }

    assert modified == {"src/agentdojo/benchmark.py"}
    lowered = text.casefold()
    assert "api_key" not in lowered
    assert ("/" + "home" + "/") not in lowered
    assert ("/" + "lustre" + "/") not in lowered


def test_checkout_verifier_rejects_extra_edit_inside_patched_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    benchmark = checkout / "src" / "agentdojo" / "benchmark.py"
    benchmark.parent.mkdir(parents=True)
    original_lines = [f"VALUE_{index} = {index}\n" for index in range(20)]
    benchmark.write_text("".join(original_lines), encoding="utf-8")
    (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    license_bytes = b"fixture license\n"
    (checkout / "LICENSE").write_bytes(license_bytes)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reviewed_lines = list(original_lines)
    reviewed_lines[1] = "VALUE_1 = 200\n"
    benchmark.write_text("".join(reviewed_lines), encoding="utf-8")
    patch = tmp_path / "reviewed.patch"
    patch.write_bytes(
        subprocess.run(
            ["git", "-C", str(checkout), "diff", "--binary"],
            check=True,
            capture_output=True,
        ).stdout
    )
    monkeypatch.setattr(integration, "UPSTREAM_COMMIT", commit)
    monkeypatch.setattr(
        integration,
        "UPSTREAM_LICENSE_SHA256",
        hashlib.sha256(license_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        integration,
        "OPTIONAL_PATCH_SHA256",
        hashlib.sha256(patch.read_bytes()).hexdigest(),
    )

    assert verify_agentdojo_checkout(
        checkout, optional_patch=patch
    )["optional_patch_applied"] is True

    extra_lines = list(reviewed_lines)
    extra_lines[-1] = "VALUE_19 = 999\n"
    benchmark.write_text("".join(extra_lines), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact reviewed patched tree"):
        verify_agentdojo_checkout(checkout, optional_patch=patch)
