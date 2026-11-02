from __future__ import annotations

import json
import os

import pytest

from secpo.attacks import GeneticConfig
from secpo.evaluation.agentdojo_qwen36 import (
    AgentDojoConfig,
    AgentDojoEpisode,
    FreshAgentDojoEpisodeRunner,
    paper_genetic_tasks,
    run_genetic,
    run_paper80_genetic,
)


def _proposal(request):
    parent = request.parents[0] if request.parents else "seed"
    return [f"{parent} revision-{request.step}-{i}" for i in range(request.count)]


def _episode(task, injection):
    return AgentDojoEpisode(
        response=f"transcript: {injection}",
        exact_success=False,
        utility=True,
        dense_score=min(9.0, float(injection.count("revision"))),
    )


def test_agentdojo_genetic_step_checkpoint_resume_matches_full_run():
    task = paper_genetic_tasks(goal="perform exact task")[0]
    checkpoints = []
    partial = GeneticConfig(
        steps=1,
        candidates_per_step=1,
        num_islands=1,
        query_budget=8,
        seed=7,
        early_stop_score=10.0,
    )
    run_genetic(
        task=task,
        proposer=_proposal,
        episode=_episode,
        initial_injections=("seed",),
        config=partial,
        checkpoint=checkpoints.append,
    )
    full = GeneticConfig(
        steps=4,
        candidates_per_step=1,
        num_islands=1,
        query_budget=8,
        seed=7,
        early_stop_score=10.0,
    )
    resumed = run_genetic(
        task=task,
        proposer=_proposal,
        episode=_episode,
        initial_injections=("seed",),
        config=full,
        resume_payload=checkpoints[-1],
    )
    uninterrupted = run_genetic(
        task=task,
        proposer=_proposal,
        episode=_episode,
        initial_injections=("seed",),
        config=full,
    )
    assert resumed == uninterrupted


def test_paper80_orchestrator_writes_all_pairs_and_skips_them_on_resume(tmp_path):
    tasks = paper_genetic_tasks(goal="perform exact task")
    calls = []

    def episode(task, injection):
        calls.append(task.key)
        return AgentDojoEpisode("trace", False, True, 0.0)

    def unused_proposer(request):
        raise AssertionError("one seed exhausts this test budget")

    config = GeneticConfig(
        steps=1,
        candidates_per_step=1,
        num_islands=1,
        query_budget=1,
        seed=3,
        early_stop_score=10.0,
    )
    summary = run_paper80_genetic(
        tasks=tasks,
        proposer=unused_proposer,
        episode=episode,
        output_dir=tmp_path,
        config=config,
    )
    assert summary["complete_paper80"] is True
    assert summary["selected_count"] == 80
    assert summary["victim_queries"] == 80
    assert summary["final_replay_queries"] == 80
    assert len(calls) == 160  # One search episode and one final replay per pair.
    assert len(list((tmp_path / "pairs").glob("*.json"))) == 80
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 80

    calls.clear()
    resumed = run_paper80_genetic(
        tasks=tasks,
        proposer=unused_proposer,
        episode=episode,
        output_dir=tmp_path,
        config=config,
        resume=True,
    )
    assert resumed == summary
    assert calls == []
    persisted = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert persisted["paper_manifest_count"] == 80


def test_paper80_orchestrator_rejects_duplicate_or_partial_manifest(tmp_path):
    tasks = list(paper_genetic_tasks(goal="goal"))
    tasks[-1] = tasks[0]
    with pytest.raises(ValueError, match="exact paper-80"):
        run_paper80_genetic(
            tasks=tasks,
            proposer=_proposal,
            episode=_episode,
            output_dir=tmp_path,
            config=GeneticConfig(
                steps=1,
                candidates_per_step=1,
                num_islands=1,
                query_budget=1,
            ),
        )


def test_fresh_episode_runner_uses_exact_checker_and_bounded_dense_score(monkeypatch):
    monkeypatch.setenv("AGENTDOJO_BASE_URL", "https://model.example.invalid/v1")
    monkeypatch.setenv("AGENTDOJO_API_KEY", "approved-agentdojo-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-nvidia-key")
    monkeypatch.setenv("UNRELATED_DATABASE_PASSWORD", "unrelated-password")

    class Pipeline:
        name = "fixture"

        def query(self, *args, **kwargs):
            return (
                "query",
                object(),
                object(),
                [{"role": "assistant", "content": "visible"}],
                {},
            )

    monkeypatch.setattr(
        "secpo.evaluation.agentdojo_qwen36._build_pipeline",
        lambda config: Pipeline(),
    )
    runner = FreshAgentDojoEpisodeRunner(
        config=AgentDojoConfig(),
        dense_scorer=lambda task, injection, transcript: 9.0,
        transcript_max_chars=1000,
    )
    task = paper_genetic_tasks(goal="goal")[0]
    environments = []

    class UserTask:
        pass

    class InjectionTask:
        GOAL = "goal"

    class Probe:
        def get_injection_candidates(self, user_task):
            return ["vector"]

    class Suite:
        user_tasks = {task.user_task_id: UserTask()}
        injection_tasks = {task.injection_task_id: InjectionTask()}

        def get_user_task_by_id(self, task_id):
            return self.user_tasks[task_id]

        def get_injection_task_by_id(self, task_id):
            return self.injection_tasks[task_id]

        def run_task_with_pipeline(self, pipeline, user, injection_task, injections):
            environment = object()
            environments.append(environment)
            assert injections == {"vector": "candidate"}
            assert os.environ["AGENTDOJO_BASE_URL"].startswith("https://")
            assert os.environ["AGENTDOJO_API_KEY"] == "approved-agentdojo-key"
            assert "NVIDIA_API_KEY" not in os.environ
            assert "UNRELATED_DATABASE_PASSWORD" not in os.environ
            pipeline.query("prompt", None, environment)
            return True, False

    monkeypatch.setattr(runner, "_suite_and_probe", lambda suite: (Suite(), Probe()))
    first = runner(task, "candidate")
    second = runner(task, "candidate")
    assert first.exact_success is False
    assert first.utility is True
    assert first.dense_score == 9.0
    assert first.metadata["fresh_environment"] is True
    assert len(environments) == 2
    assert environments[0] is not environments[1]
    assert second == first
    assert os.environ["NVIDIA_API_KEY"] == "unrelated-nvidia-key"
    assert os.environ["UNRELATED_DATABASE_PASSWORD"] == "unrelated-password"
