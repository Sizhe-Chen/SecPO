from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from secpo.attacks import agentdojo_exact_first as pipeline
from secpo.attacks import agentdojo_genetic as original
from secpo.evaluation.agentdojo_models import OpenAIChatConfig, OpenAIChatModel
from secpo.evaluation.agentdojo_qwen36 import (
    AgentDojoConfig, AgentDojoEpisode, FreshAgentDojoEpisodeRunner,
    default_genetic_seeds, paper_genetic_tasks,
)
from secpo.evaluation.agentdojo_search import backend_from_config, run_paper80_search


TASK = paper_genetic_tasks(goal="synthetic goal")[0]


def proposal(request):
    return [f"mutation-{request.step}-{i}" for i in range(request.count)]


def victim(task, injection):
    return AgentDojoEpisode("trace:" + injection, False, True)


def score(task, injection, episode):
    value = 1 + int(hashlib.sha256(injection.encode()).hexdigest()[:12], 16) / 2**48
    return replace(episode, dense_score=value)


def run(**kwargs):
    defaults = dict(task=TASK, context={}, victim=victim, scorer=score,
                    proposer=proposal, identity={"fixture": 1}, save=lambda state: None,
                    steps=3, candidates=8, selection_seed=1234)
    defaults.update(kwargs)
    return pipeline.search(**defaults)


def test_success_stops_at_exact_candidate_without_critic_or_replay():
    calls = []

    def exact(task, injection):
        calls.append(injection)
        return AgentDojoEpisode("trace", len(calls) == 3, True)

    def forbidden(*args):
        pytest.fail("success must stop before mutation or paid scoring")

    result = run(victim=exact, scorer=forbidden, proposer=forbidden)
    assert result["search_success"] is True
    assert result["success_metric"] == "search_asr"
    assert result["victim_episodes"] == len(calls) == 3
    assert result["winner"]["candidate_id"] == "s001-c002"


def test_failed_generation_checks_all_victims_before_critic_including_last():
    events = []
    result = run(
        steps=2, victim=lambda t, i: events.append("victim") or victim(t, i),
        scorer=lambda t, i, e: events.append("critic") or score(t, i, e),
    )
    assert events == (["victim"] * 8 + ["critic"] * 8) * 2
    assert result["search_success"] is False
    assert result["victim_episodes"] == 16


@pytest.mark.parametrize("phase", ["victim", "critic"])
def test_mid_generation_resume_does_not_repeat_committed_calls(phase):
    saved = []
    calls = {"victim": [], "critic": []}
    paused = False

    def check():
        if not paused and len(calls[phase]) == 3:
            raise RuntimeError("injected interruption")

    kwargs = dict(
        victim=lambda t, i: calls["victim"].append(i) or victim(t, i),
        scorer=lambda t, i, e: calls["critic"].append(i) or score(t, i, e),
        save=saved.append, check=check,
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        run(**kwargs)
    paused = True
    resumed = run(resume=json.loads(json.dumps(saved[-1])), **kwargs)
    uninterrupted = run()
    assert resumed["selection"] == uninterrupted["selection"]
    expected = [i for batch in uninterrupted["batches"] for i in batch["injections"]]
    assert calls == {"victim": expected, "critic": expected}


def test_parent_history_archive_and_rng_match_original_policy():
    expected_requests, actual_requests = [], []

    def record(destination, request):
        destination.append((request.parent_injection, request.parents, request.sampling_method))
        return proposal(request)

    expected = original.run_agentdojo_genetic_search(
        goal=TASK.goal, initial_injections=default_genetic_seeds(TASK.goal),
        proposer=lambda r: record(expected_requests, r),
        victim=lambda i: victim(TASK, i).response,
        scorer=lambda i, response: score(TASK, i, AgentDojoEpisode(response, False, True)).dense_score,
        config=original.AgentDojoGeneticConfig(steps=12, seed=1234),
    )
    actual = run(steps=12, proposer=lambda r: record(actual_requests, r))
    assert actual_requests == expected_requests
    actual_by_id = {r["candidate_id"]: r["injection"] for b in actual["batches"] for r in b["records"]}
    expected_by_id = {r.candidate_id: r.injection for r in expected.evaluations}
    assert [[actual_by_id[i] for i in group] for group in actual["selection"]["islands"]] == [
        [expected_by_id[i] for i in group] for group in expected.state["islands"]
    ]
    assert actual["selection"]["rng"] == original._tuples(expected.state["random_state"])
    assert [(r["cell"], actual_by_id[r["id"]]) for r in actual["selection"]["archive"]] == [
        (row["cell"], expected_by_id[row["candidate_id"]]) for row in expected.state["archive"]
    ]


def test_resume_rejects_tampering_and_changed_identity_before_calls():
    saved = run()
    damaged = copy.deepcopy(saved)
    damaged["batches"][0]["records"][0]["score"] = 10
    for overrides in ({"resume": damaged}, {"resume": saved, "identity": {"fixture": 2}}):
        with pytest.raises(ValueError, match="identity or digest"):
            run(victim=lambda *a: pytest.fail("must not call victim"), **overrides)


def test_episode_errors_abort_without_counting_defense_success():
    with pytest.raises(RuntimeError, match="Episode error"):
        run(victim=lambda *a: AgentDojoEpisode("error", False, False, metadata={"episode_error": "timeout"}))


def test_critic_cannot_change_exact_outcome():
    with pytest.raises(ValueError, match="Critic changed"):
        run(scorer=lambda t, i, e: replace(e, exact_success=True))


def fixture_runner(monkeypatch):
    monkeypatch.setenv("AGENTDOJO_BASE_URL", "https://victim.example.invalid/v1")
    episode = object.__new__(FreshAgentDojoEpisodeRunner)
    episode.config = AgentDojoConfig()
    episode.transcript_max_chars = 16000
    episode.tool_result_max_chars = 3000
    episode.dense_scorer = lambda *a: 1.0
    episode.task_context = lambda task: task.to_dict()
    episode.replay = lambda *a: AgentDojoEpisode("trace", True, True)
    return episode


def test_paper80_search_resume_skips_paid_calls_and_rejects_changed_run(tmp_path, monkeypatch):
    episode = fixture_runner(monkeypatch)
    calls = []
    episode.replay = lambda *a: calls.append(a) or AgentDojoEpisode("trace", True, True)
    kwargs = dict(tasks=paper_genetic_tasks(goal="goal"), proposer=proposal,
                  episode=episode, output_dir=tmp_path, steps=2)
    result = run_paper80_search(**kwargs)
    assert result["completed_pairs"] == result["verified_successes"] == len(calls) == 80
    assert result["search_asr_so_far"] == 1.0
    assert result["final_replay_queries"] == 0
    assert run_paper80_search(resume=True, **kwargs) == result
    assert len(calls) == 80
    with pytest.raises(ValueError, match="output exists"):
        run_paper80_search(**kwargs)
    with pytest.raises(ValueError, match="identity differs"):
        run_paper80_search(resume=True, run_identity={"changed": True}, **kwargs)


def test_subset_reports_full_manifest_denominator(tmp_path, monkeypatch):
    result = run_paper80_search(
        tasks=paper_genetic_tasks(goal="goal"), proposer=proposal,
        episode=fixture_runner(monkeypatch), output_dir=tmp_path, task_indices=[0, 1],
    )
    assert result["search_asr_so_far"] == 2 / 80
    assert result["selected_search_asr"] == 1
    assert result["complete"] and not result["complete_paper80"]


def test_openrouter_provider_route_is_preserved_without_serializing_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    root = Path(__file__).parents[1]
    raw = json.loads((root / "configs/evaluate_agentdojo_search_qwen36_secpo.json").read_text())
    backend = backend_from_config(raw["models"])
    requests = []

    def transport(url, payload, headers, timeout):
        requests.append(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    for model in (backend.attacker, backend.judge):
        client = OpenAIChatModel(model.config, transport=transport)
        assert client.complete("system", "user") == "ok"
    assert requests[0]["provider"]["only"] == ["minimax"]
    assert requests[1]["provider"]["only"] == ["deepinfra/fp8"]
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[1]["reasoning_effort"] == "none"
    assert all(p["max_tokens"] == 16384 and not p["provider"]["allow_fallbacks"] for p in requests)
    assert "fixture-only" not in json.dumps(backend.config_identity())
    assert OpenAIChatConfig(model="fixture").provider is None


def test_cli_search_only_uses_profile_and_has_no_final_replay(tmp_path, monkeypatch, capsys):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("search_cli", root / "scripts/evaluate_agentdojo_qwen36_27b.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    episode = fixture_runner(monkeypatch)
    monkeypatch.setattr(cli, "FreshAgentDojoEpisodeRunner", lambda **kw: episode)
    monkeypatch.setattr(cli, "load_verified_paper_genetic_tasks",
                        lambda *a, **kw: (paper_genetic_tasks(goal="goal"), {"fixture": True}))
    profile = root / "configs/evaluate_agentdojo_search_qwen36_secpo.json"
    assert cli.main([
        "--config", str(profile), "adaptive", "--checkout", str(tmp_path),
        "--backend-config", str(profile), "--search-only", "--output-dir", str(tmp_path / "out"),
        "--task-index", "0", "--steps", "1", "--query-budget", "8",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["search_asr_so_far"] == 1 / 80
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["final_replay_queries"] == 0
