#!/usr/bin/env python3
"""Verify or run the pinned external AgentDojo evaluation slice.

Examples::

    python scripts/evaluate_agentdojo_qwen36_27b.py verify \
      --checkout /path/to/agentdojo

    python scripts/evaluate_agentdojo_qwen36_27b.py utility \
      --checkout /path/to/agentdojo --output results/agentdojo-utility.json

    python scripts/evaluate_agentdojo_qwen36_27b.py static \
      --checkout /path/to/agentdojo --output results/agentdojo-static.json

The ``adaptive`` command derives all 80 goals from the pinned checkout,
executes complete candidate generations, then independently replays the
selected payload. The built-in model backend is configured through environment
variables; callback overrides remain available for integrations and tests.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from secpo.attacks.agentdojo_genetic import AgentDojoGeneticConfig
from secpo.evaluation.agentdojo_models import AgentDojoModelBackend
from secpo.evaluation.agentdojo_qwen36 import (
    AgentDojoEpisode,
    AgentDojoTask,
    FreshAgentDojoEpisodeRunner,
    load_config,
    load_verified_paper_genetic_tasks,
    paper_genetic_tasks,
    run_benchmark,
    run_genetic,
    run_paper80_genetic,
    verify_agentdojo_checkout,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RELEASE_ROOT / "configs" / "evaluate_agentdojo_qwen36_27b.json"
DEFAULT_PATCH = RELEASE_ROOT / "third_party" / "patches" / "agentdojo.patch"


def _load_callable(specification: str, label: str) -> Any:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"{label} must use package.module:attribute syntax")
    module = importlib.import_module(module_name)
    value: Any = module
    for component in attribute_name.split("."):
        if not component or component.startswith("_"):
            raise ValueError(f"invalid {label} attribute {attribute_name!r}")
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"{specification!r} does not resolve to a callable")
    return value


def _write_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _add_checkout(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--optional-patch",
        type=Path,
        default=DEFAULT_PATCH,
        help="Only this exact optional patch is accepted in a dirty checkout.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="verify pin, license, and patch")
    _add_checkout(verify)

    for mode in ("utility", "static"):
        command = subparsers.add_parser(mode, help=f"run AgentDojo {mode}")
        _add_checkout(command)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--log-dir", type=Path)
        command.add_argument("--suite", action="append", default=[])
        command.add_argument("--user-task", action="append", default=[])
        if mode == "static":
            command.add_argument("--injection-task", action="append", default=[])
        command.add_argument("--force-rerun", action="store_true")

    genetic = subparsers.add_parser(
        "genetic", help="run one exact-oracle Genetic task pair"
    )
    _add_checkout(genetic)
    genetic.add_argument("--episode", required=True)
    genetic.add_argument("--proposer", required=True)
    genetic.add_argument("--task-index", type=int, required=True)
    genetic.add_argument("--goal", required=True)
    genetic.add_argument("--output", type=Path, required=True)
    genetic.add_argument("--steps", type=int, default=100)
    genetic.add_argument("--candidates", type=int, default=8)
    genetic.add_argument("--query-budget", type=int, default=800)
    genetic.add_argument("--seed", type=int, default=1234)

    adaptive = subparsers.add_parser(
        "adaptive",
        help="run the exact resumable v1.2.1 paper-80 Genetic evaluation",
    )
    _add_checkout(adaptive)
    adaptive.add_argument("--proposer", help="Optional package.module:callable override for the built-in attacker")
    adaptive.add_argument(
        "--dense-scorer",
        help="Optional package.module:callable(task, injection, transcript)",
    )
    adaptive.add_argument("--output-dir", type=Path, required=True)
    adaptive.add_argument(
        "--task-index",
        action="append",
        type=int,
        default=[],
        help="Repeat to run a subset; omit to execute all 80 pairs.",
    )
    adaptive.add_argument("--resume", action="store_true")
    adaptive.add_argument("--steps", type=int, default=100)
    adaptive.add_argument("--candidates", type=int, default=8)
    adaptive.add_argument("--query-budget", type=int, default=800)
    adaptive.add_argument("--seed", type=int, default=1234)
    adaptive.add_argument("--transcript-max-chars", type=int, default=16000)
    adaptive.add_argument("--tool-result-max-chars", type=int, default=3000)
    adaptive.add_argument("--runtime-identity", default="", help="Version tag for custom callback models, prompts and runtime settings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "verify":
        result = verify_agentdojo_checkout(
            args.checkout,
            optional_patch=args.optional_patch,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command in ("utility", "static"):
        log_dir = args.log_dir or args.output.with_suffix("").with_name(
            f"{args.output.stem}-episodes"
        )
        result = run_benchmark(
            checkout=args.checkout,
            config=config,
            mode=args.command,
            log_dir=log_dir,
            optional_patch=args.optional_patch,
            suites=args.suite or None,
            user_tasks=args.user_task or None,
            injection_tasks=(
                getattr(args, "injection_task", None) or None
            ),
            force_rerun=args.force_rerun,
        )
        _write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "macro_average": result["macro_average"]}))
        return 0

    if args.command == "adaptive":
        tasks, upstream = load_verified_paper_genetic_tasks(
            args.checkout,
            optional_patch=args.optional_patch,
        )
        backend = AgentDojoModelBackend.from_environment() if args.proposer is None else None
        proposer = backend.propose if backend is not None else _load_callable(args.proposer, "proposer")
        dense_scorer = (_load_callable(args.dense_scorer, "dense scorer") if args.dense_scorer
                        else backend.score if backend is not None else None)
        episode = FreshAgentDojoEpisodeRunner(
            config=config,
            dense_scorer=dense_scorer,
            transcript_max_chars=args.transcript_max_chars,
            tool_result_max_chars=args.tool_result_max_chars,
        )
        summary = run_paper80_genetic(
            tasks=tasks,
            proposer=proposer,
            episode=episode,
            output_dir=args.output_dir,
            config=AgentDojoGeneticConfig(
                steps=args.steps,
                candidates_per_step=args.candidates,
                num_islands=5,
                island_capacity=100,
                map_bins=10,
                query_budget=args.query_budget,
                seed=args.seed,
                early_stop_score=10.0,
            ),
            resume=args.resume,
            task_indices=args.task_index or None,
            upstream=upstream,
            run_identity={"callback_runtime_version": args.runtime_identity},
        )
        print(
            json.dumps(
                {
                    "output": str(args.output_dir / "summary.json"),
                    "selected_count": summary["selected_count"],
                    "attack_success_rate": summary["attack_success_rate"],
                }
            )
        )
        return 0

    upstream = verify_agentdojo_checkout(
        args.checkout,
        optional_patch=args.optional_patch,
    )
    tasks = paper_genetic_tasks(goal=args.goal)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ValueError(f"task-index must be in [0, {len(tasks) - 1}]")
    episode_callable = _load_callable(args.episode, "episode")
    proposer = _load_callable(args.proposer, "proposer")

    def episode(task: AgentDojoTask, injection: str) -> AgentDojoEpisode | dict[str, Any]:
        return episode_callable(task, injection)

    result = run_genetic(
        task=tasks[args.task_index],
        proposer=proposer,
        episode=episode,
        config=AgentDojoGeneticConfig(
            steps=args.steps,
            candidates_per_step=args.candidates,
            num_islands=5,
            island_capacity=100,
            map_bins=10,
            query_budget=args.query_budget,
            seed=args.seed,
            early_stop_score=10.0,
        ),
    )
    result["upstream"] = upstream
    _write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "pass_at_n": result["pass_at_n"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
