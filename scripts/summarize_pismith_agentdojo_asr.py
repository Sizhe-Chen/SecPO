#!/usr/bin/env python3
"""Count the AgentDojo pairs that any PI-Smith attempt broke, over training and evaluation."""

import argparse
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True, help="agentdojo_judge_samples.jsonl of the training run")
    parser.add_argument("--evaluation", action="append", default=[], help="eval_detailed.jsonl; may be repeated")
    args = parser.parse_args()

    attempts: Counter = Counter()
    broken = set()
    for path in [args.training_log, *args.evaluation]:
        with open(path, encoding="utf-8") as rows:
            for line in rows:
                row = json.loads(line)
                pair = (row["suite_name"], row["user_task_id"], row["injection_task_id"])
                attempts[pair] += 1
                if row["security"]:
                    broken.add(pair)
    print(
        f"pairs={len(attempts)} broken={len(broken)} asr={len(broken) / len(attempts):.4f} "
        f"attempts_per_pair_min={min(attempts.values())} attempts_per_pair_max={max(attempts.values())}"
    )


if __name__ == "__main__":
    main()
