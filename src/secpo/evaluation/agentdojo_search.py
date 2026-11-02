"""Portable entry point for the existing exact-first AgentDojo search loop."""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from secpo.attacks import agentdojo_exact_first as pipeline
from secpo.attacks import agentdojo_genetic
from . import agentdojo_models, agentdojo_prompts, agentdojo_qwen36, agentdojo_transport
from .agentdojo_models import AgentDojoModelBackend, OpenAIChatConfig, OpenAIChatModel
from .agentdojo_qwen36 import (
    AgentDojoTask,
    _atomic_write_json,
    _read_json_object,
    _run_identity,
    _validate_paper_task_set,
)


def backend_from_config(config: Mapping[str, Any]) -> AgentDojoModelBackend:
    return AgentDojoModelBackend(
        OpenAIChatModel(OpenAIChatConfig(**config["attacker"])),
        OpenAIChatModel(OpenAIChatConfig(**config["judge"])),
        workers=config.get("workers", 1),
        judge_format_retries=config.get("judge_format_retries", 2),
    )


def run_paper80_search(
    *, tasks: Sequence[AgentDojoTask], proposer: Any, episode: Any,
    output_dir: str | Path, steps: int = 100, candidates: int = 8,
    format_retry_rounds: int = 15, resume: bool = False,
    task_indices: Sequence[int] | None = None,
    upstream: Mapping[str, Any] | None = None,
    run_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_paper_task_set(tasks)
    indices = list(range(80) if task_indices is None else task_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("task_indices must be non-empty and unique")
    if any(type(index) is not int or not 0 <= index < 80 for index in indices):
        raise ValueError("every task index must be between 0 and 79")
    if type(steps) is not int or steps < 1 or type(candidates) is not int or not 1 <= candidates <= 8:
        raise ValueError("steps must be positive and candidates must be between 1 and 8")
    if type(format_retry_rounds) is not int or format_retry_rounds < 0:
        raise ValueError("format_retry_rounds must be nonnegative")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Workers targeting different replicas use separate output directories.
    with (root / "search.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another search owns this output directory") from None
        identity = {
            "runtime": _run_identity(episode, proposer, run_identity),
            "victim": asdict(episode.config),
            "upstream": dict(upstream or {}),
            "sources": {
                module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (pipeline, agentdojo_genetic, agentdojo_models,
                               agentdojo_prompts, agentdojo_qwen36, agentdojo_transport)
            },
        }
        manifest = {
            "protocol": pipeline.PROTOCOL, "success_metric": "search_asr",
            "final_replay": False, "identity": identity,
            "tasks": [task.to_dict() for task in tasks], "task_indices": indices,
            "steps": steps, "candidates": candidates,
            "format_retry_rounds": format_retry_rounds,
        }
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("output exists; use --resume or a new output directory")
            if _read_json_object(manifest_path) != manifest:
                raise ValueError("search manifest identity differs")
        else:
            if any(path.name != "search.lock" for path in root.iterdir()):
                raise ValueError("output contains artifacts without a search manifest")
            _atomic_write_json(manifest_path, manifest)
        owner = getattr(proposer, "__self__", proposer)
        rows: list[dict[str, Any]] = []

        def summary() -> dict[str, Any]:
            successes = sum(row["search_success"] for row in rows)
            result = {
                "protocol": pipeline.PROTOCOL, "success_metric": "search_asr",
                "completed_pairs": len(rows), "total_pairs": 80,
                "selected_count": len(indices), "complete": len(rows) == len(indices),
                "complete_paper80": len(rows) == 80, "verified_successes": successes,
                "search_asr_so_far": successes / 80,
                "selected_search_asr": successes / len(indices),
                "final_replay_queries": 0, "pairs": rows,
            }
            _atomic_write_json(root / "summary.json", result)
            return result

        for index in indices:
            task = tasks[index]
            target = root / "tasks" / f"{index:02d}"
            checkpoint = target / "checkpoint.json"
            result_path = target / "result.json"
            if result_path.exists() and not checkpoint.exists():
                raise ValueError("completed result has no verifiable checkpoint")
            context = episode.task_context(task)
            if callable(getattr(owner, "set_task_context", None)):
                owner.set_task_context(context)
            state = pipeline.search(
                task=task, context=context, victim=episode.replay,
                scorer=lambda t, injection, cached: pipeline.score_cached_episode(episode, t, injection, cached),
                proposer=proposer, identity=identity,
                save=lambda value, path=checkpoint: _atomic_write_json(path, value),
                resume=_read_json_object(checkpoint) if resume and checkpoint.exists() else None,
                steps=steps, candidates=candidates, format_retry_rounds=format_retry_rounds,
            )
            _atomic_write_json(result_path, {"task_index": index, "result": state})
            rows.append({
                "task_index": index, "search_success": state["search_success"],
                "generations": len(state["batches"]), "victim_episodes": state["victim_episodes"],
                "result_path": str(result_path.relative_to(root)),
            })
            summary()
        return summary()
