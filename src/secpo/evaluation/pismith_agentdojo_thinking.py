"""Thinking Qwen3.8 AgentDojo target for PI-Smith.

This is the target pipeline of the recorded thinking-on AgentDojo evaluations
of Qwen3.8-27B and its SecOPD model: native Qwen tool calls, the user prompt
repeated after each tool result, and the Qwen thinking sampling profile over a
131,072-token context. Defended models read tool results in the `input` role,
as recorded; the undefended model reads them in its native `tool` role.
Thinking that is still open after 32,768 tokens is closed and the turn is
completed from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
import hashlib
import json
import re
from typing import Any

from secpo.evaluation.agentdyn_qwen36 import _assistant_tool_calls, _content_text, messages_to_qwen_input_roles


MAX_MODEL_LEN = 131072
THINKING_MAX_TOKENS = 32768
FINAL_MAX_TOKENS = 1536
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0, frequency_penalty=0.0)
STOPS = ["<|im_end|>", "<|endoftext|>"]
SEED = 0


class AgentOutputError(ValueError):
    """The target produced no usable turn: unclosed thinking, a broken tool call, or a full context."""


def trust_role_template(tokenizer) -> str:
    """Extend the native Qwen3.8 template for SecPO trust roles."""
    template = tokenizer.chat_template
    needle = 'message.role == "user"'
    if template.count(needle) != 2 or "<function=" not in template:
        raise ValueError("Unexpected native Qwen3.8 tool template")
    template = template.replace(needle, '(message.role == "user" or message.role == "input")')
    template = template.replace("<|im_start|>system", "<|im_start|>user")
    return template


def native_messages(messages) -> list[dict[str, Any]]:
    """AgentDojo messages in their own roles, for the unmodified Qwen3.8 template."""
    converted = []
    for message in messages:
        role = message["role"]
        content = _content_text(message.get("content"))
        if role == "tool":
            if message.get("error") is not None:
                content = json.dumps({"error": str(message["error"])})
            converted.append({"role": "tool", "content": content or "Success"})
        elif role == "assistant":
            result = {"role": "assistant", "content": content or None}
            calls = _assistant_tool_calls(message)
            if calls:
                result["tool_calls"] = calls
            converted.append(result)
        elif role in ("system", "user"):
            converted.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported AgentDojo message role: {role!r}")
    return converted


def render_agent_prompt(tokenizer, messages, tools, tool_role: str) -> str:
    if tool_role == "input":
        converted, template = messages_to_qwen_input_roles(messages), trust_role_template(tokenizer)
    elif tool_role == "tool":
        converted, template = native_messages(messages), tokenizer.chat_template
    else:
        raise ValueError(f"tool role must be input or tool, got {tool_role!r}")
    # The converter serializes arguments for OpenAI-compatible servers;
    # the native Qwen3.8 XML template expects mappings when rendering history.
    for message in converted:
        for call in message.get("tool_calls", []) or []:
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return tokenizer.apply_chat_template(
        converted, tools=tools, chat_template=template, tokenize=False,
        add_generation_prompt=True, enable_thinking=True, preserve_thinking=False,
    )


@dataclass(frozen=True)
class ToolCall:
    function: str
    args: dict[str, Any]
    id: str


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_tool_response(text: str, tools: list[dict]) -> tuple[str, list[ToolCall]]:
    """Parse native Qwen XML calls using each function's parameter schema."""
    # The thinking-on template already ends in <think>, so generated tokens
    # begin with reasoning text rather than a second opening tag.
    if "</think>" not in text:
        raise AgentOutputError("Unclosed thinking block in agent response")
    text = text.split("</think>", 1)[1].strip()
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    calls = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        body = match.group(1)
        function = re.fullmatch(r"<function\s*=\s*([^>]+)>\s*(.*?)\s*</function>", body.strip(), re.DOTALL)
        if function is None:
            raise AgentOutputError("Malformed Qwen XML tool call")
        name = function.group(1).strip()
        properties = schemas.get(name, {}).get("properties", {})
        arguments = {}
        for parameter in re.finditer(r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>", function.group(2), re.DOTALL):
            key, raw = parameter.group(1).strip(), parameter.group(2).strip()
            spec = properties.get(key, {})
            types = {spec.get("type"), *(item.get("type") for item in spec.get("anyOf", []))}
            if "string" in types:
                value = raw
            else:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value = raw
            arguments[key] = value
        calls.append(ToolCall(name, arguments, "call_" + _digest([len(calls), body])[:24]))
    if "<tool_call>" in text and not calls:
        raise AgentOutputError("Incomplete Qwen tool call")
    visible = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return visible, calls


def tool_json_dumps(value):
    """JSON tool results with ISO dates, which plain json.dumps rejects."""
    def default(item):
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"Unsupported tool result value: {type(item).__name__}")
    return json.dumps(value, default=default)


def complete(tokenizer, url: str, served_model: str, prompt: str, max_tokens: int) -> tuple[str, str]:
    """One completion for a rendered prompt, sent as token IDs. Returns the text and the finish reason."""
    import requests

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    max_tokens = min(max_tokens, MAX_MODEL_LEN - len(tokens))
    if max_tokens <= 0:
        raise AgentOutputError("Agent prompt exceeds model context")
    payload = dict(model=served_model, prompt=tokens, n=1, return_token_ids=True,
                   skip_special_tokens=False, max_tokens=max_tokens, stop=STOPS, seed=SEED, **SAMPLING)
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(url + "/completions", json=payload, timeout=(30, 3600))
    if response.status_code != 200:
        raise RuntimeError(f"Target inference failed with HTTP {response.status_code}: {response.text[:500]}")
    result = response.json()
    if result.get("model") != served_model or len(result["choices"]) != 1:
        raise ValueError("Unexpected inference response identity")
    choice = result["choices"][0]
    output_ids = choice.get("token_ids")
    if not isinstance(output_ids, list) or not all(type(token) is int for token in output_ids):
        raise ValueError("Missing raw generation token IDs")
    if (result["usage"]["prompt_tokens"] != len(tokens)
            or result["usage"]["completion_tokens"] != len(output_ids)
            or choice["finish_reason"] not in {"length", "stop"}):
        raise ValueError("Unexpected inference token accounting or finish reason")
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    positions = [text.find(stop) for stop in STOPS if stop in text]
    return (text[:min(positions)] if positions else text), choice["finish_reason"]


def generate(tokenizer, url: str, served_model: str, prompt: str) -> str:
    """One agent turn; thinking still open after THINKING_MAX_TOKENS is closed and the turn completed."""
    output, finish_reason = complete(tokenizer, url, served_model, prompt, THINKING_MAX_TOKENS)
    closed = "</think>" in output
    if closed and (finish_reason == "stop" or output.split("</think>", 1)[1].strip()):
        return output
    closing = "" if closed else "\n</think>\n\n"
    final, _ = complete(tokenizer, url, served_model, prompt + output + closing, FINAL_MAX_TOKENS)
    return output + closing + final


def make_pipeline(url: str, served_model: str, tokenizer_path: str, tool_role: str):
    """AgentDojo pipeline whose LLM is the thinking Qwen3.8 model served at `url`."""
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.types import text_content_block_from_string
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

    class ThinkingQwenAgent(BasePipelineElement):
        name = served_model

        def query(self, query, runtime, env, messages=(), extra_args=None):
            tools = [{"type": "function", "function": {
                "name": f.name, "description": f.description, "parameters": f.parameters.model_json_schema()
            }} for f in runtime.functions.values()]
            prompt = render_agent_prompt(tokenizer, messages, tools, tool_role)
            content, calls = parse_tool_response(generate(tokenizer, url, served_model, prompt), tools)
            message = {"role": "assistant",
                       "content": [text_content_block_from_string(content)] if content else None,
                       "tool_calls": [FunctionCall(function=c.function, args=c.args, id=c.id) for c in calls]}
            return query, runtime, env, [*messages, message], dict(extra_args or {})

    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=ThinkingQwenAgent(), model_id=served_model, defense="repeat_user_prompt",
        system_message_name=None, system_message=None, tool_output_format="json",
    ))

    def configure_formatters(element):
        if isinstance(element, ToolsExecutor):
            element.output_formatter = partial(tool_result_to_str, dump_fn=tool_json_dumps)
        for child in getattr(element, "elements", ()):
            configure_formatters(child)

    configure_formatters(pipeline)
    return pipeline
