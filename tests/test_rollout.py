import json
from dataclasses import replace

import pytest
from test_protocol import PREFIX, xml_call

from coding_rl import make_environment
from coding_rl.rollout.episode import bounded_observation, run_model_episode
from coding_rl.rollout.evaluate import aggregate, evaluate
from config import ModelConfig
from environments.execution.sandbox import Sandbox
from environments.patches import make_diff
from models.backend import Generation, GenerationTimeout
from utils.errors import InfrastructureError
from utils.seeding import derive_seed


class FakeBackend:
    identity = {"kind": "test", "backend": "fake"}
    startups = []

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []
        self.messages = []
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True

    def count_text(self, text):
        return (len(text) + 3) // 4

    def render(self, messages):
        self.messages.append(json.loads(json.dumps(messages)))
        return [1] * self.count_text(json.dumps(messages)), PREFIX

    def generate(self, ids, **kwargs):
        self.requests.append({"input_ids": ids, **kwargs})
        output = next(self.outputs, "Done.")
        if isinstance(output, BaseException):
            raise output
        if isinstance(output, tuple):
            text, stop = output
        else:
            text, stop = output + "<|im_end|>", "eos"
        return Generation(
            "fake-request", [1] * min(64, kwargs["max_new_tokens"]), text, stop, 0.001
        )


@pytest.fixture
def model_env(config):
    # Only authored fixtures execute locally in these test-labelled fake runs.
    instance = make_environment(
        replace(config, model=ModelConfig(), sandbox=replace(config.sandbox, kind="docker"))
    )
    instance.sandbox = Sandbox(config.sandbox)
    try:
        yield instance
    finally:
        instance.cleanup()


def test_fake_read_patch_test_and_independent_grade(model_env):
    task = model_env.catalog.get("zero-timeout")
    patch = make_diff(task.starter, task.variant("correct"))
    backend = FakeBackend(
        [
            xml_call("read_file", path=task.module),
            xml_call("apply_patch", patch=patch),
            xml_call("run_tests"),
            "Done.",
        ]
    )
    outcome = run_model_episode(model_env, backend)
    assert outcome.completed and outcome.reward == 1.0
    assert outcome.driver_kind == "test"
    assert outcome.core_calls == outcome.proposed_calls == outcome.accepted_calls == 3
    assert outcome.accepted_edits == 1 and outcome.turns == 4
    assert outcome.finish_reason == "final_response" and outcome.cleanup_complete
    assert model_env.workspace is None
    generations = [
        json.loads(line)
        for line in (model_env.recorder.root / f"{outcome.episode_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sum(e["event"] == "generation" for e in generations) == 4
    assert all(e["policy"]["kind"] == "test" for e in generations)
    conversation = json.loads(
        (model_env.recorder.root / f"{outcome.episode_id}-conversation.json").read_text()
    )
    assert sum(m["role"] == "tool" for m in conversation) == 3
    assert all(
        "expected" not in json.dumps(m) and "references" not in json.dumps(m)
        for batch in backend.messages
        for m in batch
    )
    with pytest.raises(ValueError, match="genuine"):
        aggregate([outcome], planned=1)
    with pytest.raises(ValueError, match="actual model"):
        evaluate(model_env, backend)


def test_truncated_turn_never_applies_even_a_complete_patch(model_env):
    task = model_env.catalog.get("zero-timeout")
    patch = make_diff(task.starter, task.variant("correct"))
    backend = FakeBackend([(xml_call("apply_patch", patch=patch), "length")])
    outcome = run_model_episode(model_env, backend)
    assert outcome.reward == 0.0 and outcome.finish_reason == "token_limit"
    assert outcome.core_calls == outcome.accepted_edits == 0
    assert outcome.submission_hash == task.starter_hash


def test_correct_source_retains_reward_at_tool_horizon(model_env):
    model_env.config = replace(
        model_env.config, environment=replace(model_env.config.environment, max_tool_calls=1)
    )
    task = model_env.catalog.get("zero-timeout")
    backend = FakeBackend(
        [xml_call("apply_patch", patch=make_diff(task.starter, task.variant("correct")))]
    )
    outcome = run_model_episode(model_env, backend)
    assert outcome.reward == 1.0 and outcome.finish_reason == "tool_limit"


def test_invalid_proposals_are_charged_without_core_entry(model_env):
    backend = FakeBackend(
        [xml_call("cleanup"), xml_call("read_file", path="a", start_line="true"), "Done"]
    )
    outcome = run_model_episode(model_env, backend)
    assert outcome.proposed_calls == outcome.rejected_calls == 2
    assert outcome.core_calls == 0 and outcome.reward == 0.0


def test_parser_recovery_limit_and_multicall_atomic_rejection(model_env):
    task = model_env.catalog.get("zero-timeout")
    patch = make_diff(task.starter, task.variant("correct"))
    backend = FakeBackend(
        [xml_call("apply_patch", patch=patch) + xml_call("run_tests"), "<tool_call>"]
    )
    outcome = run_model_episode(model_env, backend)
    assert outcome.proposed_calls == outcome.rejected_calls == 3
    assert outcome.parser_errors == 2 and outcome.finish_reason == "parser_limit"
    assert outcome.accepted_edits == outcome.core_calls == 0


@pytest.mark.parametrize(
    "failure,reason,completed",
    [
        (GenerationTimeout("timeout"), "generation_timeout", True),
        (InfrastructureError("worker crash"), "infrastructure_failure", False),
        (KeyboardInterrupt(), "cancelled", False),
    ],
)
def test_generation_failure_and_cleanup(model_env, failure, reason, completed):
    outcome = run_model_episode(model_env, FakeBackend([failure]))
    assert outcome.finish_reason == reason and outcome.completed is completed
    assert outcome.reward == (0.0 if completed else None)
    assert outcome.cleanup_complete


def test_model_warms_before_episode_clock_and_fresh_history(model_env):
    backend = FakeBackend(["Done.", "Done."])

    def start():
        assert model_env.phase != "active"

    backend.start = start
    first = run_model_episode(model_env, backend)
    second = run_model_episode(model_env, backend, attempt=1)
    assert first.episode_id != second.episode_id and first.starter_hash == second.starter_hash
    assert first.seed != second.seed
    assert all(len(messages) == 2 for messages in backend.messages)


def test_initial_context_admission_and_observation_token_bounds(model_env):
    backend = FakeBackend(["Done."])
    text = bounded_observation({"ok": True, "stdout": "x" * 5000, "returncode": 0}, backend, 64)
    result = json.loads(text)
    assert result["ok"] and result["truncated"] and result["returncode"] == 0
    assert backend.count_text(text) <= 64
    backend.render = lambda messages: ([1] * 2048, PREFIX)
    with pytest.raises(ValueError, match="Initial public prompt"):
        run_model_episode(model_env, backend)
    assert model_env.workspace is None


def test_context_limit_stops_before_patch_mutation(model_env):
    task = model_env.catalog.get("zero-timeout")
    backend = FakeBackend(
        [xml_call("apply_patch", patch=make_diff(task.starter, task.variant("correct")))]
    )
    render = backend.render
    backend.render = lambda messages: (
        ([1] * 2000, PREFIX) if len(messages) > 2 else render(messages)
    )
    outcome = run_model_episode(model_env, backend)
    assert outcome.finish_reason == "context_limit" and outcome.accepted_edits == 0


def test_recorder_failure_releases_resources(model_env, monkeypatch):
    emit = model_env.recorder.emit

    def fail(event, **payload):
        if event == "generation":
            raise InfrastructureError("disk full")
        emit(event, **payload)

    monkeypatch.setattr(model_env.recorder, "emit", fail)
    outcome = run_model_episode(model_env, FakeBackend(["Done"]))
    assert outcome.reward is None and outcome.infrastructure_error
    assert outcome.cleanup_complete


def test_honest_metrics_include_failures_and_exclude_infrastructure(model_env):
    synthetic = run_model_episode(model_env, FakeBackend(["Done"]))
    failed = replace(synthetic, driver_kind="model", policy={"kind": "model"})
    correct = replace(
        failed, reward=1.0, verdict={**failed.verdict, "reward": 1.0, "status": "correct"}
    )
    incomplete = replace(failed, completed=False, reward=None, infrastructure_error="outage")
    metrics = aggregate([failed, correct, incomplete], planned=4)
    assert metrics["verified_pass_rate"] == metrics["mean_binary_reward"] == 0.5
    assert metrics["valid"] == 2 and metrics["incomplete"] == metrics["not_started"] == 1
    assert aggregate([incomplete], planned=1)["verified_pass_rate"] is None


def test_stable_seed_derivation():
    assert derive_seed(42, "zero-timeout", 0, 1) == derive_seed(42, "zero-timeout", 0, 1)
    assert len({derive_seed(42, "zero-timeout", 0, i) for i in range(4)}) == 4


def test_verifier_outage_preserves_final_stop_reason(model_env, monkeypatch):
    def fail(*args, **kwargs):
        raise InfrastructureError("grading unavailable")

    monkeypatch.setattr(model_env.sandbox, "visible_tests", fail)
    outcome = run_model_episode(model_env, FakeBackend(["Done"]))
    assert outcome.finish_reason == "final_response"
    assert not outcome.completed and outcome.reward is None
    assert outcome.infrastructure_error and outcome.cleanup_complete


def test_evaluator_saves_partial_coverage_after_startup_failure(model_env, monkeypatch, tmp_path):
    from coding_rl.rollout import evaluate as module

    backend = FakeBackend([])
    backend.identity = {"kind": "model", "backend": "synthetic-unit-test"}
    model_env.config = replace(
        model_env.config,
        evaluation=replace(
            model_env.config.evaluation,
            attempts_per_task=2,
            report_path=str(tmp_path / "report.json"),
        ),
    )
    selected = []

    def fail(environment, backend, task_id, **kwargs):
        selected.append(task_id)
        raise InfrastructureError("simulated model startup failure")

    monkeypatch.setattr(module, "run_model_episode", fail)
    report = module.evaluate(model_env, backend)
    assert selected == ["chunk-tail", "chunk-tail"]
    assert report["metrics"]["planned"] == report["metrics"]["incomplete"] == 2
    assert report["metrics"]["valid"] == 0
    assert report["metrics"]["verified_pass_rate"] is None
    assert json.loads((tmp_path / "report.json").read_text())["metrics"] == report["metrics"]
