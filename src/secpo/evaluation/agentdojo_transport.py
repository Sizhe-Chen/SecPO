"""Lossless AgentDojo fixture transport and archived trajectory representation.

Ported from the paper-80 implementation. AgentDojo remains an external pinned
optional dependency; imports needed by its fixture loader happen only on use.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import MethodType
from typing import Any, Sequence

TRANSPORT_VERSION = "agentdojo-lossless-ascii-placeholder-v1"
_TRANSPORT_BINDING_ATTRIBUTE = "_adaptive_injection_transport_version"

def read_suite_file(*args: Any) -> str:
    from agentdojo.task_suite.task_suite import read_suite_file as read
    return read(*args)

def validate_injections(*args: Any) -> Any:
    from agentdojo.task_suite.task_suite import validate_injections as validate
    return validate(*args)

def get_text_content_as_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(item["content"] for item in content if item["content"] is not None)

class AgentDojoInjectionTransportError(RuntimeError):
    """The environment fixture could not losslessly carry an injection.

    This is an evaluator-infrastructure error, not a victim-model outcome.  In
    particular, callers must not turn it into a paid zero-fitness candidate or
    send it to the trajectory judge.
    """


def _collision_checked_placeholders(
    environment_text: str, values: dict[str, str]
) -> dict[str, str]:
    """Create one YAML-safe ASCII sentinel per injection vector.

    Tokens are deterministic for reproducibility, but every candidate set is
    still checked against both the fixture source and every exact replacement
    string.  The retry suffix makes even a deliberately constructed collision
    harmless.
    """

    canonical = json.dumps(
        [environment_text, sorted(values.items())],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()[:24].upper()
    haystacks = (environment_text, *values.values())
    vector_ids = sorted(values)
    attempt = 0
    while True:
        placeholders = {
            vector_id: (
                f"ADJLOSSLESS_{digest}_{attempt:04X}_{index:04X}_END"
            )
            for index, vector_id in enumerate(vector_ids)
        }
        tokens = tuple(placeholders.values())
        collides_with_source = any(
            token in haystack for token in tokens for haystack in haystacks
        )
        tokens_overlap = any(
            left in right
            for left_index, left in enumerate(tokens)
            for right_index, right in enumerate(tokens)
            if left_index != right_index
        )
        if not collides_with_source and not tokens_overlap:
            return placeholders
        attempt += 1


def _restore_placeholder_substrings(
    value: Any,
    replacements: dict[str, str],
    pattern: re.Pattern[str] | None = None,
) -> Any:
    """Recursively restore sentinels, including embedded prefix/suffix use."""

    if pattern is None and replacements:
        pattern = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(replacements, key=len, reverse=True)
            )
        )
    if isinstance(value, str):
        if pattern is None:
            return value
        return pattern.sub(lambda match: replacements[match.group(0)], value)
    if isinstance(value, list):
        return [
            _restore_placeholder_substrings(item, replacements, pattern)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _restore_placeholder_substrings(item, replacements, pattern)
            for item in value
        )
    if isinstance(value, set):
        restored = {
            _restore_placeholder_substrings(item, replacements, pattern)
            for item in value
        }
        if len(restored) != len(value):
            raise ValueError("placeholder restoration produced duplicate set values")
        return restored
    if isinstance(value, dict):
        restored_mapping: dict[Any, Any] = {}
        for key, item in value.items():
            restored_key = _restore_placeholder_substrings(
                key, replacements, pattern
            )
            if restored_key in restored_mapping:
                raise ValueError(
                    "placeholder restoration produced duplicate mapping keys"
                )
            restored_mapping[restored_key] = _restore_placeholder_substrings(
                item, replacements, pattern
            )
        return restored_mapping
    return value


def _lossless_load_and_inject_default_environment(
    self: Any, injections: dict[str, str]
) -> Any:
    """TaskSuite loader that keeps attacker text out of the YAML language."""

    import yaml

    stage = "read environment fixture"
    try:
        environment_text = read_suite_file(
            self.name, "environment.yaml", self.data_path
        )
        stage = "read injection-vector defaults"
        defaults = self.get_injection_vector_defaults()
        stage = "validate injection keys and values"
        validate_injections(injections, defaults)
        values = {**defaults, **injections}
        if not all(isinstance(key, str) for key in values):
            raise TypeError("injection-vector IDs must be strings")
        if not all(isinstance(value, str) for value in values.values()):
            raise TypeError("injection-vector values must be strings")

        stage = "construct collision-free placeholders"
        placeholders = _collision_checked_placeholders(environment_text, values)
        stage = "resolve fixture placeholders"
        resolved_text = environment_text.format(**placeholders)
        stage = "parse placeholder-only YAML"
        parsed_environment = yaml.safe_load(resolved_text)
        stage = "restore exact injection strings"
        restored_environment = _restore_placeholder_substrings(
            parsed_environment,
            {
                placeholders[vector_id]: exact_value
                for vector_id, exact_value in values.items()
            },
        )
        stage = "validate restored environment"
        return self.environment_type.model_validate(restored_environment)
    except AgentDojoInjectionTransportError:
        raise
    except Exception as error:
        raise AgentDojoInjectionTransportError(
            "Lossless AgentDojo injection transport failed for suite "
            f"{self.name!r} while attempting to {stage} "
            f"({type(error).__name__})"
        ) from error


def bind_lossless_injection_transport(task_suite: Any) -> Any:
    """Bind lossless fixture loading to exactly this benchmark suite instance."""

    installed_version = getattr(task_suite, _TRANSPORT_BINDING_ATTRIBUTE, None)
    if installed_version is not None and installed_version != TRANSPORT_VERSION:
        raise AgentDojoInjectionTransportError(
            "TaskSuite already has an incompatible adaptive injection transport: "
            f"{installed_version!r}"
        )
    if installed_version == TRANSPORT_VERSION:
        loader = getattr(task_suite, "load_and_inject_default_environment", None)
        if (
            getattr(loader, "__func__", None)
            is not _lossless_load_and_inject_default_environment
        ):
            raise AgentDojoInjectionTransportError(
                "TaskSuite transport version marker does not match its bound loader"
            )
        return task_suite

    task_suite.load_and_inject_default_environment = MethodType(
        _lossless_load_and_inject_default_environment, task_suite
    )
    setattr(task_suite, _TRANSPORT_BINDING_ATTRIBUTE, TRANSPORT_VERSION)
    return task_suite


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return get_text_content_as_str(content)
    except (KeyError, TypeError):
        return str(content)


def _clip(text: str, limit: int, marker: str = "\n...[truncated]...") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


def _clip_head_tail(text: str, limit: int) -> str:
    """Bound a trajectory while preserving both discovery and final actions."""
    marker = "\n...[middle of trajectory truncated]...\n"
    if len(text) <= limit:
        return text
    available = max(0, limit - len(marker))
    head = available // 3
    tail = available - head
    return text[:head] + marker + text[-tail:]


def serialize_trajectory(
    messages: Sequence[dict[str, Any]],
    *,
    transcript_max_chars: int,
    tool_result_max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Render visible agent/tool evidence while excluding system/reasoning data."""
    events: list[dict[str, Any]] = []
    actual_calls: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role == "assistant":
            visible = _content_text(message.get("content"))
            if visible:
                events.append({"role": "assistant", "content": visible})
            for call in message.get("tool_calls") or []:
                item = {
                    "id": call.id,
                    "function": call.function,
                    "args": _jsonable(call.args),
                    "result_preview": None,
                    "error": None,
                }
                actual_calls.append(item)
                events.append(
                    {
                        "role": "assistant_tool_call",
                        "id": item["id"],
                        "function": item["function"],
                        "args": item["args"],
                    }
                )
            continue
        if role == "tool":
            call = message.get("tool_call")
            result_preview = _clip(
                _content_text(message.get("content")), tool_result_max_chars
            )
            call_error = message.get("error")
            result_call_id = message.get("tool_call_id") or (
                call.id if call is not None else None
            )
            if call is not None:
                for actual_call in reversed(actual_calls):
                    if (
                        actual_call["function"] == call.function
                        and (
                            result_call_id is None
                            or actual_call["id"] == result_call_id
                        )
                        and actual_call["result_preview"] is None
                    ):
                        actual_call["result_preview"] = result_preview
                        actual_call["error"] = call_error
                        break
            events.append(
                {
                    "role": "tool_result",
                    "function": call.function if call is not None else None,
                    "content": result_preview,
                    "error": call_error,
                }
            )
            continue
        if role == "user":
            # The original task is supplied separately to the critic. Repeated
            # user turns are useful evidence for repeat_user_prompt defenses.
            events.append({"role": "user", "content": _content_text(message.get("content"))})

    transcript = json.dumps(events, ensure_ascii=False, indent=2)
    return _clip_head_tail(transcript, transcript_max_chars), actual_calls


def _expected_tool_calls(suite: Any, injection_task: Any) -> list[dict[str, Any]]:
    try:
        environment = suite.load_and_inject_default_environment({})
        calls = injection_task.ground_truth(environment)
    except Exception:
        return []
    result = []
    for call in calls:
        # Placeholder args describe the intended plan without resolving and
        # leaking environment-specific secrets to the attack generator.
        args = call.placeholder_args if call.placeholder_args is not None else {}
        result.append({"function": call.function, "args": _jsonable(args)})
    return result


def _tool_schemas(suite: Any) -> list[dict[str, Any]]:
    result = []
    for tool in suite.tools:
        result.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
            }
        )
    return result


def _walk_pipeline_elements(root: Any) -> list[Any]:
    """Return each pipeline component once, including nested tool loops."""
    result: list[Any] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        element = stack.pop()
        if id(element) in seen:
            continue
        seen.add(id(element))
        result.append(element)
        nested = getattr(element, "elements", None)
        if nested is not None and not isinstance(nested, (str, bytes)):
            try:
                stack.extend(reversed(list(nested)))
            except TypeError:
                pass
    return result


def _victim_pipeline_metadata(pipeline: Any, *, include_system_text: bool) -> dict[str, Any]:
    """Extract cache-safe victim identity without serializing API clients."""
    components = []
    for element in _walk_pipeline_elements(pipeline):
        component: dict[str, Any] = {
            "class": f"{type(element).__module__}.{type(element).__qualname__}"
        }
        if hasattr(element, "model"):
            for field_name in (
                "model",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "max_tokens",
                "seed",
                "tool_delimiter",
                "native_tool_calling",
                "enable_thinking",
                "preserve_thinking",
            ):
                if hasattr(element, field_name):
                    component[field_name] = _jsonable(getattr(element, field_name))
        system_message = getattr(element, "system_message", None)
        if isinstance(system_message, str):
            component["system_message_sha256"] = hashlib.sha256(
                system_message.encode("utf-8")
            ).hexdigest()
            if include_system_text:
                component["system_message"] = system_message
        if hasattr(element, "max_iters"):
            component["max_iters"] = getattr(element, "max_iters")
        components.append(component)
    return {"pipeline_name": getattr(pipeline, "name", None), "components": components}


def _injection_vector_context(suite: Any, vector_ids: Sequence[str]) -> list[dict[str, Any]]:
    import yaml

    defaults = suite.get_injection_vector_defaults()
    descriptions: dict[str, Any] = {}
    try:
        raw = read_suite_file(suite.name, "injection_vectors.yaml", suite.data_path)
        descriptions = yaml.safe_load(raw) or {}
    except Exception:
        pass
    result = []
    for vector_id in vector_ids:
        metadata = descriptions.get(vector_id, {})
        result.append(
            {
                "id": vector_id,
                "description": metadata.get("description", "") if isinstance(metadata, dict) else "",
                "benign_default_preview": _clip(str(defaults.get(vector_id, "")), 500),
            }
        )
    return result
