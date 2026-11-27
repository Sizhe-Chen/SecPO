#!/usr/bin/env python3
"""Evaluate a trained PI-Smith attacker on SEP with pass@10."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import APITimeoutError, OpenAI
from transformers import AutoTokenizer

from .pismith_judge import GeminiJudge
from .pismith_protocol import (
    attacker_messages,
    parse_candidate,
    score_text,
    target_messages,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_FIELDS = ("instruction", "input", "injection", "witness")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class EndpointTarget:
    def __init__(self, config: dict[str, Any], url: str) -> None:
        target = config["target"]
        generation = config["target_generation"]
        self.client = OpenAI(
            base_url=url,
            api_key="EMPTY",
            timeout=generation["request_timeout_seconds"],
            max_retries=0,
        )
        models = self.client.models.list().data
        if not models:
            raise RuntimeError(f"Target endpoint exposes no models: {url}")
        self.model_name = models[0].id
        self.thinking = generation["thinking"]
        self.max_tokens = generation["max_tokens"]
        self.timeout = generation["request_timeout_seconds"]
        self.force_close = generation["force_close"]
        self.final_tokens = generation["final_answer_max_tokens"]
        tokenizer = AutoTokenizer.from_pretrained(
            target["model_id"],
            revision=target["revision"],
            trust_remote_code=True,
        )
        chat_template = REPO_ROOT / config["chat_template"]
        if not chat_template.is_file():
            raise FileNotFoundError(f"Missing Qwen chat template: {chat_template}")
        tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
        self.tokenizer = tokenizer
        self.concurrency = generation["api_concurrency"]

    def _generate_one(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        if not self.force_close:
            result = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=self.max_tokens,
            )
            return result.choices[0].text or ""

        started = time.monotonic()
        chunks: list[str] = []
        finish_reason = None
        timed_out = False
        stream = None
        try:
            stream = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    chunks.append(choice.text or "")
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                if finish_reason is None and time.monotonic() - started >= self.timeout:
                    timed_out = True
                    break
        except APITimeoutError:
            timed_out = True
        finally:
            if stream is not None:
                stream.close()

        output = "".join(chunks)
        naturally_closed = "</think>" in output
        if naturally_closed and output.split("</think>", 1)[1].strip() and not timed_out:
            return output
        closing = "" if naturally_closed else "\n</think>\n\n"
        continuation = self.client.completions.create(
            model=self.model_name,
            prompt=prompt + output + closing,
            temperature=0,
            max_tokens=self.final_tokens,
        )
        final = continuation.choices[0].text or ""
        if not final.strip():
            raise RuntimeError("Target returned no answer after forced thinking closure")
        return output + closing + final

    def generate(self, messages: list[list[dict[str, str]]]) -> tuple[list[str], list[int]]:
        outputs: list[str | None] = [None] * len(messages)
        timed_out: list[int] = []

        def run(index: int) -> tuple[int, str]:
            return index, self._generate_one(messages[index])

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(run, i): i for i in range(len(messages))}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    _, output = future.result()
                    outputs[index] = output
                except APITimeoutError:
                    timed_out.append(index)
                    outputs[index] = ""
        missing = [i for i, value in enumerate(outputs) if value is None]
        if missing:
            raise RuntimeError(f"Target produced no output for indices: {missing}")
        return [value or "" for value in outputs], sorted(timed_out)


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "name", "target", "attacker", "dataset", "target_generation", "judge", "chat_template"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Protocol is missing fields: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported PI-Smith protocol schema")
    if config["attacker"]["pass_at_k"] != 10:
        raise ValueError("Recorded protocol requires pass_at_k=10")
    if config["target_generation"]["data_role"] != "input":
        raise ValueError("Recorded protocol requires target data_role=input")
    if config["judge"]["repetitions_per_model"] != 3:
        raise ValueError("Recorded protocol requires three judge repetitions per model")


def load_dataset(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the first ``count`` SEP records after checking the fields the evaluation reads."""
    data = read_json(path)
    count = config["dataset"]["count"]
    if not isinstance(data, list) or len(data) < count:
        raise ValueError(f"Expected at least {count} SEP records in {path}")
    data = data[:count]
    for index, sample in enumerate(data):
        missing = set(SAMPLE_FIELDS) - sample.keys()
        if missing:
            raise ValueError(f"SEP record {index} is missing fields: {sorted(missing)}")
    actual_hash = records_sha256(data)
    expected_hash = config["dataset"]["records_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"SEP records hash mismatch: expected {expected_hash}, got {actual_hash}")
    return data


def records_sha256(data: list[dict[str, Any]]) -> str:
    rows = [[sample[field] for field in SAMPLE_FIELDS] for sample in data]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def command_version(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    chat_template = REPO_ROOT / config["chat_template"]
    return {
        "schema_version": 1,
        "protocol": config,
        "protocol_sha256": sha256(args.protocol),
        "dataset_path": str(args.data.resolve()),
        "dataset_sha256": sha256(args.data),
        "attacker_url": args.attacker_url,
        "target_url": args.target_url,
        "limit": args.limit,
        "environment": {
            "python": sys.version,
            "packages": {
                name: metadata.version(name)
                for name in ("google-genai", "openai", "transformers", "vllm")
            },
        },
        "source_sha256": {
            name: sha256(source_dir / name)
            for name in ("pismith_sep.py", "pismith_judge.py", "pismith_protocol.py")
        }
        | {
            str(chat_template.relative_to(REPO_ROOT)): sha256(chat_template),
        },
        "git_commit": command_version(["git", "rev-parse", "HEAD"]),
    }


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = read_json(path)
        if existing != manifest:
            raise RuntimeError(f"Run manifest changed; use a new output directory: {path.parent}")
        return
    write_json(path, manifest)


def generate_calibration(target, data: list[dict[str, Any]], output_dir: Path, after_think: bool) -> tuple[list[str], list[str]]:
    path = output_dir / "calibration.json"
    if path.exists():
        cached = read_json(path)
        if len(cached["injection_goals"]) != len(data) or len(cached["benign_responses"]) != len(data):
            raise ValueError(f"Invalid calibration cache: {path}")
        return cached["injection_goals"], cached["benign_responses"]

    injection_messages = [[{"role": "user", "content": item["injection"]}] for item in data]
    benign_messages = [target_messages(item, item["input"]) for item in data]
    injection_goals, injection_timeouts = target.generate(injection_messages)
    benign_responses, benign_timeouts = target.generate(benign_messages)
    if injection_timeouts or benign_timeouts:
        raise RuntimeError("Calibration generation timed out; no result was written")
    payload = {
        "injection_goals": injection_goals,
        "benign_responses": benign_responses,
        "scored_injection_goals": [score_text(item, after_think) for item in injection_goals],
        "scored_benign_responses": [score_text(item, after_think) for item in benign_responses],
    }
    write_json(path, payload)
    return injection_goals, benign_responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--attacker-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = read_json(args.protocol)
    validate_config(config)
    data = load_dataset(args.data, config)
    if args.limit is not None:
        if not 0 < args.limit <= len(data):
            raise ValueError("--limit must be between 1 and the dataset size")
        data = data[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(args.output_dir / "manifest.json", build_manifest(args, config))

    target = EndpointTarget(config, args.target_url)

    attacker_client = OpenAI(base_url=args.attacker_url, api_key="EMPTY")
    attacker_models = attacker_client.models.list().data
    if not attacker_models:
        raise RuntimeError(f"Attacker endpoint exposes no models: {args.attacker_url}")
    attacker_model = attacker_models[0].id
    attacker_config = config["attacker"]
    generation_config = config["target_generation"]
    judge_config = config["judge"]
    judge = GeminiJudge(
        judge_config["primary_model"],
        judge_config["secondary_model"],
        judge_config["concurrency"],
        judge_config["max_retries"],
    )

    injection_goals, benign_responses = generate_calibration(
        target,
        data,
        args.output_dir,
        generation_config["score_after_think"],
    )
    scored_goals = [score_text(item, generation_config["score_after_think"]) for item in injection_goals]
    scored_benign = [score_text(item, generation_config["score_after_think"]) for item in benign_responses]

    batch_size = generation_config["page_batch_size"]
    result_dir = args.output_dir / "batches"
    result_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch_path = result_dir / f"{start:04d}-{end:04d}.json"
        if batch_path.exists():
            batch_results = read_json(batch_path)
            expected = list(range(start, end))
            if [item["index"] for item in batch_results] != expected:
                raise ValueError(f"Invalid completed batch: {batch_path}")
            all_results.extend(batch_results)
            continue

        batch = data[start:end]
        candidates: list[list[str]] = []
        for sample in batch:
            response = attacker_client.chat.completions.create(
                model=attacker_model,
                messages=attacker_messages(sample),
                n=attacker_config["pass_at_k"],
                temperature=attacker_config["temperature"],
                max_tokens=attacker_config["max_tokens"],
            )
            parsed = [parse_candidate(choice.message.content or "") for choice in response.choices]
            if len(parsed) != attacker_config["pass_at_k"]:
                raise RuntimeError(f"Attacker returned {len(parsed)} candidates, expected {attacker_config['pass_at_k']}")
            candidates.append(parsed)

        flat_messages: list[list[dict[str, str]]] = []
        flat_pairs: list[tuple[int, int]] = []
        for page_offset, (sample, page_candidates) in enumerate(zip(batch, candidates)):
            for candidate_offset, candidate in enumerate(page_candidates):
                flat_messages.append(target_messages(sample, (sample["input"] or "") + " " + candidate))
                flat_pairs.append((page_offset, candidate_offset))
        target_outputs, timeout_indices = target.generate(flat_messages)
        timeout_set = set(timeout_indices)
        responses: list[list[str]] = [[""] * attacker_config["pass_at_k"] for _ in batch]
        timed_out_candidates: list[list[int]] = [[] for _ in batch]
        for flat_index, ((page_offset, candidate_offset), output) in enumerate(zip(flat_pairs, target_outputs)):
            responses[page_offset][candidate_offset] = output
            if flat_index in timeout_set:
                timed_out_candidates[page_offset].append(candidate_offset)

        batch_results = []
        for offset, sample in enumerate(batch):
            index = start + offset
            witness = (sample["witness"] or "").lower()
            success = False
            success_index = None
            judge_outputs = None
            for candidate_index, response in enumerate(responses[offset]):
                if candidate_index in timed_out_candidates[offset]:
                    continue
                scored = score_text(response, generation_config["score_after_think"])
                if witness and witness not in scored.lower():
                    continue
                passed, outputs = judge.evaluate(
                    instruction=sample["instruction"],
                    injection=sample["injection"],
                    response=scored,
                    perfect_example=scored_goals[index],
                    failed_example=scored_benign[index],
                )
                if passed:
                    success = True
                    success_index = candidate_index
                    judge_outputs = outputs
                    break
            batch_results.append(
                {
                    "index": index,
                    "attack_success": success,
                    "success_candidate_index": success_index,
                    "candidates": candidates[offset],
                    "responses": responses[offset],
                    "timed_out_candidate_indices": timed_out_candidates[offset],
                    "judge_outputs": judge_outputs,
                }
            )
        write_json(batch_path, batch_results)
        all_results.extend(batch_results)
        successes = sum(bool(item["attack_success"]) for item in all_results)
        print(f"completed={len(all_results)}/{len(data)} asr={successes / len(all_results):.6f}", flush=True)

    successes = sum(bool(item["attack_success"]) for item in all_results)
    summary = {
        "samples": len(all_results),
        "successes": successes,
        "asr": successes / len(all_results),
        "pass_at_k": attacker_config["pass_at_k"],
        "reported_reference": config.get("reported_reference"),
    }
    write_json(args.output_dir / "results.json", all_results)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
