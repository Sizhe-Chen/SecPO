"""Model-free regression tests for the AgentDojo research-search port."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from secpo.attacks.base import CandidateEvaluation
from secpo.attacks.genetic import GeneticConfig
from secpo.attacks import agentdojo_genetic as search


def config(**overrides):
    return replace(search.AgentDojoGeneticConfig(), **overrides)


def proposal(request):
    return [f"mutation-{request.step}-{index}" for index in range(request.count)]


def run(**overrides):
    kwargs = dict(
        goal="synthetic goal", initial_injections=tuple(f"seed-{i}" for i in range(8)),
        proposer=proposal, victim=lambda injection: "trace:" + injection,
        scorer=lambda injection, response: 1.0,
    )
    kwargs.update(overrides)
    return search.run_agentdojo_genetic_search(**kwargs)


def item(text, score=1.0, ordinal=0):
    return search._Individual(CandidateEvaluation(
        candidate_id=f"test-{ordinal}", injection=text, response="trace", score=score, step=1,
    ))


def test_seed_generation_is_first_of_exactly_100_batches_and_800_queries():
    requests = []
    result = run(proposer=lambda request: requests.append(request) or proposal(request))
    assert len(result.evaluations) == 800
    assert result.completed_steps == 100
    assert len(requests) == 99
    assert [request.step for request in requests] == list(range(2, 101))
    assert {candidate.step for candidate in result.evaluations[:8]} == {1}
    assert result.state["query_budget_exhausted"] is True


def test_success_completes_its_eight_candidate_batch():
    calls = []
    result = run(
        victim=lambda injection: calls.append(injection) or "trace",
        scorer=lambda injection, response: 10.0 if injection == "seed-0" else 1.0,
    )
    assert len(calls) == 8
    assert result.completed_steps == 1
    assert result.best.injection == "seed-0"
    assert result.stopped_early is True


def test_duplicate_generation_refills_only_missing_slots():
    requests = []
    def propose(request):
        requests.append(request)
        if request.retry_round == 0:
            return [request.parents[0], "", "  first-child  ", "first-child"]
        return [f"fresh-{index}" for index in range(request.count)]
    result = run(config=config(steps=2), proposer=propose)
    assert len(result.evaluations) == 16
    assert [request.count for request in requests] == [8, 7]
    assert result.evaluations[8].injection == "first-child"
    assert requests[0].parent is not None
    assert requests[0].parent_injection == requests[0].parent.injection


def test_exhausted_duplicate_generation_fails_without_a_short_checkpoint():
    checkpoints, calls, requests = [], [], []
    def duplicates(request):
        requests.append(request)
        return request.parents
    with pytest.raises(RuntimeError, match="No incomplete generation"):
        run(
            config=config(steps=2, format_retry_rounds=2), proposer=duplicates,
            victim=lambda injection: calls.append(injection) or "trace",
            checkpoint=checkpoints.append,
        )
    assert len(requests) == 3
    assert len(calls) == 8
    assert len(checkpoints) == 1
    assert checkpoints[0]["completed_steps"] == 1


def test_cross_generation_repetition_is_not_globally_deduplicated():
    # With one history slot the low-scored seed is absent from the selected
    # context. The archive allows it to be evaluated again in another batch.
    def repeat_old(request):
        assert "seed-0" not in request.parents
        return ["seed-0"] + [f"child-{index}" for index in range(7)]
    result = run(
        config=config(steps=2, max_history=1), proposer=repeat_old,
        scorer=lambda injection, response: 0.5 if injection == "seed-0" else 1.0,
    )
    assert sum(candidate.injection == "seed-0" for candidate in result.evaluations) == 2
    assert len(result.evaluations) == 16


def test_archive_uses_character_similarity_and_1000_character_normalization():
    candidate = item("x" * 128)
    assert search._feature_cell(candidate, [[candidate]], config(), random.Random(3)) == (1, 9)
    assert search._feature_cell(candidate, [[item("y" * 128)]], config(), random.Random(3)) == (1, 0)
    assert search._character_similarity("aab", "abb") == 1.0
    assert search._character_similarity("A", "a") == 0.0


@pytest.mark.parametrize("draw,expected", [(0.19, "exploration"), (0.20, "exploitation"), (0.899, "exploitation"), (0.901, "random")])
def test_parent_probability_boundaries(draw, expected):
    class ControlledRandom(random.Random):
        def random(self):
            return draw
    candidate = item("one")
    _, method = search._select_parent(0, [[candidate]], {(0, 0): candidate}, config(), ControlledRandom(7))
    assert method == expected


def test_first_mutation_uses_second_island_and_search_does_not_migrate():
    requests = []
    result = run(config=config(steps=11), proposer=lambda request: requests.append(request) or proposal(request))
    islands = result.state["islands"]
    by_id = {candidate.candidate_id: candidate for candidate in result.evaluations}
    # Source scheduling: seed=step1; mutation step2 goes to island1.
    for island_index, ids in enumerate(islands):
        for candidate_id in ids:
            candidate = by_id[candidate_id]
            if candidate.step > 1:
                assert (candidate.step - 1) % 5 == island_index
    assert all(len(request.history) <= 10 for request in requests)
    assert requests[0].sampling_method in {"exploration", "exploitation", "random"}


def test_serialized_resume_restores_exact_rng_heap_archive_order_and_metadata():
    partial = run(
        config=config(steps=7),
        evaluation_metadata=lambda injection, response: {"judge_suggestion": "try tools"},
    )
    serialized = json.loads(json.dumps(partial.state, sort_keys=True))
    resumed = run(
        config=config(steps=20), resume_state=serialized,
        evaluation_metadata=lambda injection, response: {"judge_suggestion": "try tools"},
    )
    uninterrupted = run(
        config=config(steps=20),
        evaluation_metadata=lambda injection, response: {"judge_suggestion": "try tools"},
    )
    assert resumed == uninterrupted
    assert resumed.best.metadata["judge_suggestion"] == "try tools"


@pytest.mark.parametrize("field,value", [("random_state", []), ("completed_steps", 99), ("islands", [])])
def test_checkpoint_corruption_is_rejected_before_any_model_call(field, value):
    state = run(config=config(steps=2)).state
    state[field] = value
    def forbidden(*args):
        pytest.fail("corrupt resume must not make a model call")
    with pytest.raises(ValueError, match="digest"):
        run(config=config(steps=3), resume_state=state, victim=forbidden, proposer=forbidden)


def test_resume_rejects_new_goal_seed_or_search_policy():
    state = run(config=config(steps=1)).state
    for changed in ({"goal": "different"}, {"initial_injections": ["different"]},
                    {"config": config(steps=2, random_prob=0.2, exploitation_prob=0.6)}):
        with pytest.raises(ValueError, match="does not match"):
            run(resume_state=state, **changed)


def test_errored_episodes_consume_budget_but_never_enter_selection():
    requests = []
    result = run(
        config=config(steps=2),
        scorer=lambda injection, response: 10.0 if injection == "seed-0" else 1.0,
        evaluation_metadata=lambda injection, response: {
            "episode_error": "synthetic failure" if injection == "seed-0" else None,
        },
        proposer=lambda request: requests.append(request) or proposal(request),
    )
    assert len(result.evaluations) == 16
    assert result.evaluations[0].score == 0.0
    assert result.best.injection != "seed-0"
    assert result.stopped_early is False
    assert "seed-0" not in requests[0].parents
    assert requests[0].parent.injection != "seed-0"
    assert all(result.evaluations[0].candidate_id not in ids for ids in result.state["islands"])


def test_all_runtime_errors_preserve_diagnostics_without_a_successful_result():
    checkpoints = []
    with pytest.raises(RuntimeError, match="no error-free"):
        run(
            config=config(steps=1), checkpoint=checkpoints.append,
            evaluation_metadata=lambda injection, response: {"episode_error": "synthetic failure"},
        )
    assert len(checkpoints[0]["evaluations"]) == 8
    assert checkpoints[0]["best_candidate_id"] is None


def test_legacy_config_normalizes_to_research_policy_without_generic_migration():
    legacy = GeneticConfig(steps=2, candidates_per_step=1, num_islands=1, query_budget=8, seed=7)
    normalized = search.normalize_config(legacy)
    assert normalized.exploration_prob == 0.2
    assert normalized.num_elite_inspirations == 10
    result = run(config=legacy, initial_injections=["seed"])
    assert len(result.evaluations) == 2
    assert result.completed_steps == 2


def test_partial_generation_budget_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        config(query_budget=799)


@pytest.mark.parametrize("capacity", [4, 100])
def test_research_source_differential_when_archive_is_available(capacity):
    """Compare actual archived heap, descriptors, RNG, and mutation selection.

    Set SECPO_AGENTDOJO_ARCHIVE to the archived ``implementation`` directory.
    Only the pure search classes are executed; provider imports are excluded.
    Unique scores make this comparison independent of the prototype's set-order
    ties, the explicitly documented reproducibility repair in the port.
    """
    location = os.getenv("SECPO_AGENTDOJO_ARCHIVE")
    if not location:
        pytest.skip("optional differential test needs the archived research source")
    source = Path(location) / "minimal_apo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"EliteIsland", "IslandDatabase", "MAPElites", "get_inspirations",
             "jaccard_distance", "SamplingMethod", "IslandGeneticAlgorithm"}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    import enum
    import heapq
    import numpy
    class ReferenceIndividual:
        def __init__(self, value, fitness=None, **kwargs):
            self.value, self.fitness, self.kwargs = value, fitness, kwargs
        def __lt__(self, other):
            return (self.fitness if self.fitness is not None else -math.inf) < (other.fitness if other.fitness is not None else -math.inf)
        def __eq__(self, other):
            return self.value == other.value
        def __hash__(self):
            return hash(self.value)
    import math
    reference_rng = random.Random(1234)
    namespace = {
        "Individual": ReferenceIndividual, "heapq": heapq, "random": reference_rng,
        "enum": enum, "np": numpy,
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    reference_config = SimpleNamespace(
        exploration_prob=0.2, exploitation_prob=0.7, random_prob=0.1,
        num_elite_inspirations=10, num_neighbor_inspirations=5, num_random_inspirations=5,
        map_elites_tiebreak="none", sample_elites=False, elites_sample_temperature=1.0,
    )
    # Source isinstance guards only need this nominal config type.
    namespace["IslandGeneticAlgorithmConfig"] = SimpleNamespace
    ReferenceAlgorithm = namespace["IslandGeneticAlgorithm"]
    algorithm = ReferenceAlgorithm.__new__(ReferenceAlgorithm)
    algorithm.config = reference_config
    algorithm.database = namespace["IslandDatabase"](num_islands=5, island_capacity=capacity)
    algorithm.elites = namespace["MAPElites"](algorithm.database, num_bins=10)
    algorithm.init_vals = [f"seed-{index}" for index in range(8)]
    algorithm.cur_island_key, algorithm.i = "0", 0
    expected_requests = []
    def reference_mutate(parent, population):
        history = list(population)
        if parent not in history:
            history.append(parent)
        history.sort(key=lambda individual: individual.fitness, reverse=True)
        history = history[:10]
        step = algorithm.i + 1
        expected_requests.append((parent.value, tuple(individual.value for individual in history)))
        return [ReferenceIndividual(f"mutation-{step}-{index}") for index in range(8)]
    algorithm.mutator = SimpleNamespace(
        attacker_pipeline=SimpleNamespace(set_thinking_budget=lambda value: None), mutate=reference_mutate,
    )
    def score(injection, response):
        return 1 + int(hashlib.sha256(injection.encode()).hexdigest()[:12], 16) / 2**48
    for step in range(12):
        candidates = algorithm.step()
        for index, candidate in enumerate(candidates):
            candidate.fitness = score(candidate.value, "")
            key = str(index % 5) if step == 0 else algorithm.cur_island_key
            if not algorithm.database.exists(candidate, key=key):
                algorithm.database.add(candidate, key=key)
            algorithm.elites.add(candidate)
    actual_requests = []
    actual = run(
        config=config(steps=12, island_capacity=capacity), scorer=score,
        proposer=lambda request: actual_requests.append((request.parent.injection, request.parents)) or proposal(request),
    )
    assert actual_requests == expected_requests
    actual_by_id = {candidate.candidate_id: candidate.injection for candidate in actual.evaluations}
    assert [[actual_by_id[candidate_id] for candidate_id in ids] for ids in actual.state["islands"]] == [
        [individual.value for individual in algorithm.database.get(key).population] for key in algorithm.database.keys
    ]
    assert [(entry["cell"], actual_by_id[entry["candidate_id"]]) for entry in actual.state["archive"]] == [
        (list(cell), individual.value) for cell, individual in algorithm.elites.archive.items()
    ]
    assert search._tuples(actual.state["random_state"]) == reference_rng.getstate()
