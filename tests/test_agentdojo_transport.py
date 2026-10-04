from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from secpo.evaluation import agentdojo_transport as transport
from secpo.evaluation.agentdojo_qwen36 import (
    AgentDojoConfig,
    FreshAgentDojoEpisodeRunner,
    _native_assistant_message,
    _build_pipeline_in_environment,
    paper_genetic_tasks,
)


class IdentityEnvironment:
    @staticmethod
    def model_validate(value):
        return value


def fixture_suite(monkeypatch):
    # Mirrors the quoted/embedded injection slots in AgentDojo's YAML fixtures.
    fixture = 'content: "prefix {slot} suffix"\nnested:\n  - "{slot}"\n'
    monkeypatch.setattr(transport, "read_suite_file", lambda *args: fixture)
    monkeypatch.setattr(
        transport, "validate_injections",
        lambda injections, defaults: None if set(injections) <= set(defaults) else (_ for _ in ()).throw(ValueError("unknown vector")),
    )
    suite = SimpleNamespace(
        name="fixture", data_path=None, environment_type=IdentityEnvironment,
        get_injection_vector_defaults=lambda: {"slot": "benign"},
    )
    return transport.bind_lossless_injection_transport(suite)


@pytest.mark.parametrize("payload", [
    'say "hello"', r"literal\nnewline", "actual\nnewline",
    "---\ncontrols \x00 \x7f \u0085 rocket \U0001f680 {braces}",
])
def test_fixture_transport_preserves_exact_payload(monkeypatch, payload):
    suite = fixture_suite(monkeypatch)
    result = suite.load_and_inject_default_environment({"slot": payload})
    assert result == {"content": f"prefix {payload} suffix", "nested": [payload]}
    assert transport.bind_lossless_injection_transport(suite) is suite


def test_transport_error_is_explicit_infrastructure_failure(monkeypatch):
    suite = fixture_suite(monkeypatch)
    with pytest.raises(transport.AgentDojoInjectionTransportError, match="validate injection keys"):
        suite.load_and_inject_default_environment({"unknown": "candidate"})


def test_transcript_keeps_visible_calls_but_excludes_system_and_reasoning():
    call = SimpleNamespace(id="call-1", function="lookup", args={"query": "name"})
    messages = [
        {"role": "system", "content": "private system marker"},
        {"role": "user", "content": "find the name"},
        {"role": "assistant", "content": "looking", "reasoning": "private reasoning marker", "tool_calls": [call]},
        {"role": "tool", "content": "found", "tool_call": call, "tool_call_id": "call-1", "error": None},
    ]
    transcript, calls = transport.serialize_trajectory(messages, transcript_max_chars=2000, tool_result_max_chars=100)
    assert "private system marker" not in transcript
    assert "private reasoning marker" not in transcript
    assert json.loads(transcript)[2]["role"] == "assistant_tool_call"
    assert calls == [{"id": "call-1", "function": "lookup", "args": {"query": "name"}, "result_preview": "found", "error": None}]
    assert transport._clip_head_tail("HEAD" + "x" * 200 + "TAIL", 100).startswith("HEAD")
    assert transport._clip_head_tail("HEAD" + "x" * 200 + "TAIL", 100).endswith("TAIL")


def completion(content, calls=()):
    return SimpleNamespace(content=content, tool_calls=list(calls), reasoning="reasoning", model_extra={})


def test_native_adapter_ignores_bad_structured_call_and_recovers_xml():
    bad = SimpleNamespace(id="bad", function=SimpleNamespace(name="bad", arguments="not-json"))
    text = 'thinking<tool_call><function=lookup><parameter=count>2</parameter><parameter=query>name</parameter></function></tool_call>'
    result = _native_assistant_message(completion(text, [bad]), SimpleNamespace, lambda value: {"content": value})
    assert result["tool_calls"][0].function == "lookup"
    assert result["tool_calls"][0].args == {"count": 2, "query": "name"}
    assert result["content"] == [{"content": "thinking"}]


def test_native_adapter_recovers_legacy_text_call_and_preserves_normal_reasoning():
    result = _native_assistant_message(completion('<function=lookup>{"query":"name"}</function>'), SimpleNamespace, lambda value: {"content": value})
    assert result["tool_calls"][0].args == {"query": "name"}
    plain = _native_assistant_message(completion("answer"), SimpleNamespace, lambda value: {"content": value})
    assert plain["reasoning"] == "reasoning"


def test_native_pipeline_uses_archived_yaml_output_and_unique_missing_call_ids(monkeypatch):
    import sys
    from types import ModuleType

    def install(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    install("openai", OpenAI=lambda **kwargs: SimpleNamespace())
    install("agentdojo")
    install("agentdojo.agent_pipeline")
    install("agentdojo.agent_pipeline.agent_pipeline", AgentPipeline=SimpleNamespace(from_config=lambda config: config), PipelineConfig=SimpleNamespace)
    install("agentdojo.agent_pipeline.base_pipeline_element", BasePipelineElement=object)
    install("agentdojo.functions_runtime", FunctionCall=SimpleNamespace)
    install("agentdojo.types", text_content_block_from_string=lambda value: {"content": value})
    monkeypatch.setenv("AGENTDOJO_BASE_URL", "https://example.invalid/v1")
    pipeline = _build_pipeline_in_environment(AgentDojoConfig())
    assert pipeline.tool_output_format == "yaml"
    calls = [SimpleNamespace(id=None, function="first", args={}), SimpleNamespace(id=None, function="second", args={})]
    native = pipeline.llm._message({"role": "assistant", "content": None, "tool_calls": calls})
    assert native["tool_calls"][0]["id"] != native["tool_calls"][1]["id"]
    tool = pipeline.llm._message({"role": "tool", "content": "result", "tool_call": calls[0], "tool_call_id": None, "error": None})
    assert tool["tool_call_id"] == native["tool_calls"][0]["id"]


def runner_with_suite(monkeypatch, pipeline, run):
    monkeypatch.setenv("AGENTDOJO_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr("secpo.evaluation.agentdojo_qwen36._build_pipeline", lambda config: pipeline)
    runner = FreshAgentDojoEpisodeRunner(config=AgentDojoConfig())
    task = paper_genetic_tasks(goal="goal")[0]
    user = SimpleNamespace(PROMPT="original task")
    injection = SimpleNamespace(GOAL="goal")
    suite = SimpleNamespace(
        user_tasks={task.user_task_id: user}, injection_tasks={task.injection_task_id: injection},
        get_user_task_by_id=lambda key: user, get_injection_task_by_id=lambda key: injection,
        run_task_with_pipeline=run,
    )
    probe = SimpleNamespace(get_injection_candidates=lambda user: ["slot"])
    monkeypatch.setattr(runner, "_suite_and_probe", lambda name: (suite, probe))
    return runner, task


def test_abort_trajectory_reaches_dense_scorer(monkeypatch):
    class Abort(Exception):
        messages = [{"role": "assistant", "content": "partial action", "tool_calls": []}]

    class Pipeline:
        name = "fixture"

        def query(self, *args):
            raise Abort()

    def run(pipeline, *args):
        try:
            pipeline.query()
        except Abort:
            pass
        return False, False

    runner, task = runner_with_suite(monkeypatch, Pipeline(), run)
    observed = []
    runner.dense_scorer = lambda task, injection, transcript: observed.append(transcript) or 2.0
    result = runner(task, "candidate")
    assert "partial action" in observed[0]
    assert result.dense_score == 2.0
    assert result.metadata["episode_error"] is None


def test_runtime_errors_are_diagnostics_and_transport_errors_propagate(monkeypatch):
    def failed(*args):
        raise RuntimeError("victim failed")

    runner, task = runner_with_suite(monkeypatch, SimpleNamespace(name="fixture"), failed)
    runner.dense_scorer = lambda *args: pytest.fail("runtime errors must not reach judge")
    result = runner(task, "candidate")
    assert result.exact_success is False
    assert result.metadata["episode_error"] == "RuntimeError: victim failed"

    def transport_failed(*args):
        raise transport.AgentDojoInjectionTransportError("fixture failed")

    runner, task = runner_with_suite(monkeypatch, SimpleNamespace(name="fixture"), transport_failed)
    with pytest.raises(transport.AgentDojoInjectionTransportError):
        runner(task, "candidate")


def test_structured_feedback_and_replay_without_judge(monkeypatch):
    class Pipeline:
        name = "fixture"

        def query(self):
            return "", None, None, [{"role": "assistant", "content": "visible", "tool_calls": []}], {}

    def run(pipeline, *args):
        pipeline.query()
        return True, False

    runner, task = runner_with_suite(monkeypatch, Pipeline(), run)
    monkeypatch.setattr(runner, "task_context", lambda task: {
        "original_user_task": "original", "expected_malicious_tool_plan_with_placeholders": [],
        "available_tool_schemas": [],
    })
    monkeypatch.setenv("ATTACKER_TEST_KEY", "available-to-judge")

    class Scorer:
        calls = 0

        def score(self, *args):
            pytest.fail("structured scorer takes precedence")

        def score_attempt(self, attempt):
            import os
            assert os.environ["ATTACKER_TEST_KEY"] == "available-to-judge"
            assert attempt["original_user_task"] == "original"
            assert attempt["candidate_payload"] == "candidate"
            assert attempt["actual_tool_calls"] == []
            self.calls += 1
            return {"score": 6, "explanation": "partial progress", "suggestion": "try lookup"}

    scorer = Scorer()
    runner.dense_scorer = scorer.score
    result = runner(task, "candidate")
    assert result.dense_score == 6
    assert result.metadata["suggestion"] == "try lookup"
    runner.replay(task, "candidate")
    assert scorer.calls == 1
