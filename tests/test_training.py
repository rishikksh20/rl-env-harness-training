"""Training contracts use authored fixtures; these are not policy-learning evidence."""

import inspect
import json
from dataclasses import replace

import pytest

from coding_rl.training.adapter import NativeEnvironment, environment_factory, training_rows
from coding_rl.training.trainer import RunOwner, native_arguments, tool_methods
from coding_rl.training.worker import cleanup_resources
from config import ModelConfig, TrainingConfig, load_config
from environments.coding import TOOLS, tool_schemas
from environments.patches import make_diff
from tasks.catalog import TaskCatalog
from utils.errors import InfrastructureError


class CharacterPresenter:
    def count_text(self, text):
        return len(text)


@pytest.fixture
def native_setup(config, tmp_path):
    config = replace(config, model=ModelConfig(), training=TrainingConfig(observation_tokens=1024))
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "private").mkdir()
    owner = RunOwner(config, "a" * 32, directory)
    factory = environment_factory(config, TaskCatalog(), owner, CharacterPresenter())
    yield config, owner, factory
    owner.cleanup()


@pytest.mark.parametrize(
    "settings",
    [
        {"group_size": True},
        {"group_size": 2},
        {"max_groups": 9},
        {"learning_rate": float("nan")},
        {"mode": "cuda"},
        {"wall_seconds": 0},
        {"lora_targets": []},
        {"lora_targets": ["q_proj", "q_proj"]},
    ],
)
def test_training_settings_reject_unsupported_guarantees(settings):
    with pytest.raises(ValueError):
        TrainingConfig(**settings)


def test_distinct_training_settings_and_old_configs(tmp_path):
    assert load_config("configs/default.yaml").training is None
    assert load_config("configs/rollout-cpu.yaml").model.source == "hub"
    path = tmp_path / "invalid.yaml"
    path.write_text("training:\n  max_new_tokens: 32\n")
    with pytest.raises(ValueError, match="Unknown TrainingConfig"):
        load_config(path)
    assert (
        native_arguments(load_config("configs/train-cpu.yaml"), tmp_path)[
            "gradient_accumulation_steps"
        ]
        == 1
    )


def test_native_rows_have_one_public_prompt_and_no_private_data():
    rows = training_rows(TaskCatalog())
    assert {row["task_id"] for row in rows} == {"zero-timeout", "empty-default", "range-boundary"}
    for row in rows:
        assert set(row) == {"task_id", "prompt"}
        assert [message["role"] for message in row["prompt"]] == ["system", "user"]
        assert "Editable files:" in row["prompt"][1]["content"]
        assert "sequentially" in row["prompt"][0]["content"]
        assert not any(
            secret in json.dumps(row)
            for secret in (
                "private_cases",
                "references",
                "/tmp/",
                "expected",
                "chunk-tail",
            )
        )


def test_discovery_has_exactly_four_tools_and_core_argument_contract(native_setup):
    _, _, factory = native_setup
    env = factory()
    assert {method.__name__ for method in tool_methods(env)} == set(TOOLS)
    assert {
        name
        for name, _ in inspect.getmembers(NativeEnvironment, inspect.isfunction)
        if not name.startswith("_")
    } == {*TOOLS, "reset", "get_reward"}
    for name, schema in tool_schemas().items():
        signature = inspect.signature(getattr(env, name))
        assert set(signature.parameters) == set(schema["parameters"]["properties"])
        for key, parameter in signature.parameters.items():
            assert (
                parameter.annotation
                is {"string": str, "integer": int}[schema["parameters"]["properties"][key]["type"]]
            )


def test_native_reset_is_fresh_and_never_uses_held_out_tasks(native_setup):
    _, owner, factory = native_setup
    env = factory()
    with pytest.raises(ValueError, match="held-out"):
        env.reset("chunk-tail")
    assert env._core.workspace is None
    public = next(
        row["prompt"]
        for row in training_rows(env._core.catalog)
        if row["task_id"] == "zero-timeout"
    )
    with pytest.raises(ValueError, match="prompt"):
        env.reset("zero-timeout", prompt=[])
    assert env.reset("zero-timeout", prompt=public) is None
    episode, workspace = env._core.episode_id, env._core.workspace
    assert env.get_reward() == 0.0
    assert env._core.workspace is None and not workspace.exists()
    env.reset("zero-timeout")
    assert env._core.episode_id != episode
    assert len(owner.groups[0].attempts) == 2
    assert env.get_reward() == 0.0


def test_reset_unfinished_attempt_is_infrastructure_failure(native_setup):
    _, owner, factory = native_setup
    env = factory()
    env.reset("zero-timeout")
    with pytest.raises(InfrastructureError, match="unfinished"):
        env.reset("zero-timeout")
    assert owner.groups[0].status == "rejected"


def test_native_validation_rejects_boolean_before_core_and_bounds_json(native_setup):
    _, _, factory = native_setup
    env = factory()
    env.reset("zero-timeout")
    result = json.loads(env.read_file("timeout.py", start_line=True))
    assert not result["ok"] and env._core.calls == 0
    result = json.loads(env.read_file("timeout.py"))
    assert result["ok"] and env._core.calls == 1
    assert env._attempt.observations[-1]["content"] == json.dumps(result)


def test_swallowed_tool_infrastructure_failure_cannot_be_rehabilitated(native_setup, monkeypatch):
    _, owner, factory = native_setup
    env = factory()
    env.reset("zero-timeout")
    task = env._core.task
    env.apply_patch(make_diff(task.starter, task.variant("correct")))

    def outage(*args, **kwargs):
        raise InfrastructureError("simulated Docker outage")

    original = env._core.sandbox.visible_tests
    monkeypatch.setattr(env._core.sandbox, "visible_tests", outage)
    with pytest.raises(InfrastructureError):
        env.run_tests()
    monkeypatch.setattr(env._core.sandbox, "visible_tests", original)
    with pytest.raises(InfrastructureError, match="latched"):
        env.get_reward()
    assert owner.groups[0].attempts[0].reward is None


def test_correctness_with_failed_cleanup_remains_incomplete(native_setup, monkeypatch):
    _, owner, factory = native_setup
    env = factory()
    env.reset("zero-timeout")
    task = env._core.task
    env.apply_patch(make_diff(task.starter, task.variant("correct")))
    cleanup = env._core.cleanup

    def fail():
        raise InfrastructureError("simulated cleanup failure")

    monkeypatch.setattr(env._core, "cleanup", fail)
    with pytest.raises(InfrastructureError, match="cleanup"):
        env.get_reward()
    monkeypatch.setattr(env._core, "cleanup", cleanup)
    assert env._core.inspect()["verdict"]["reward"] == 1.0
    assert owner.groups[0].attempts[0].reward is None
    with pytest.raises(InfrastructureError, match="latched"):
        env.get_reward()


def test_sequential_native_calls_keep_the_core_budget(native_setup):
    _, _, factory = native_setup
    env = factory()
    env.reset("zero-timeout")
    for _ in range(6):
        assert json.loads(env.list_files())["ok"]
    assert not json.loads(env.read_file("timeout.py"))["ok"]
    assert env._core.calls == 6


def test_incomplete_group_never_reaches_reward_or_loss(native_setup):
    _, owner, factory = native_setup
    env = factory()
    env.reset("zero-timeout")
    with pytest.raises(InfrastructureError, match="complete four"):
        owner.admission_reward([], [], [], [env])
    assert owner.groups[0].status == "rejected"
    assert env._attempt.reward is None


def test_parent_cleanup_is_limited_to_pre_registered_resources(tmp_path, monkeypatch):
    run_id = "a" * 32
    run = tmp_path / "owned"
    private = run / "private"
    private.mkdir(parents=True)
    (private / "episode").write_text("writable")
    (private / "containers.jsonl").write_text(
        json.dumps("coding-rl-" + run_id + "-" + "b" * 32) + '\n"partial'
    )
    staging = run / "checkpoints" / ".checkpoint-partial"
    staging.mkdir(parents=True)
    (staging / "weights").write_text("incomplete")
    complete = run / "checkpoints" / "step-1"
    complete.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    commands = []

    def remove(command, **kwargs):
        commands.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("coding_rl.training.worker.subprocess.run", remove)
    cleanup_resources(run, run_id)
    assert commands == [["docker", "rm", "--force", "coding-rl-" + run_id + "-" + "b" * 32]]
    assert not private.exists() and not staging.exists()
    assert complete.exists() and unrelated.exists()


def test_parent_cleanup_refuses_foreign_container_names(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    for content in ('"unrelated-container"\n', "invalid-json\n"):
        (private / "containers.jsonl").write_text(content)
        with pytest.raises(InfrastructureError, match="registry"):
            cleanup_resources(tmp_path, "a" * 32)


def test_cleanup_continues_after_docker_control_failure_and_keeps_retry_registry(
    tmp_path, monkeypatch
):
    private = tmp_path / "private"
    private.mkdir()
    name = "coding-rl-" + "a" * 32 + "-" + "b" * 32
    journal = private / "containers.jsonl"
    journal.write_text(json.dumps(name) + "\n")
    (private / "episode").mkdir()
    (tmp_path / "checkpoints" / ".checkpoint-partial").mkdir(parents=True)
    monkeypatch.setattr(
        "coding_rl.training.worker.subprocess.run",
        lambda *a, **k: type(
            "Result", (), {"returncode": 1, "stderr": "Docker daemon unavailable"}
        )(),
    )
    with pytest.raises(InfrastructureError, match="cleanup incomplete"):
        cleanup_resources(tmp_path, "a" * 32)
    assert journal.exists()
    assert not (private / "episode").exists()
    assert not list((tmp_path / "checkpoints").glob(".checkpoint-*"))


def test_training_requires_real_baseline_and_development_evidence(native_setup, tmp_path):
    from coding_rl.training.trainer import validate_prerequisites

    config, _, _ = native_setup
    with pytest.raises(InfrastructureError, match="B-close"):
        validate_prerequisites(config, TaskCatalog(), "image", {})
    baseline, development = tmp_path / "baseline.json", tmp_path / "development.json"
    baseline.write_text(json.dumps({"policy": {"kind": "scripted"}}))
    development.write_text("{}")
    config = replace(
        config,
        training=replace(
            config.training, baseline_report=str(baseline), development_outcome=str(development)
        ),
    )
    with pytest.raises(InfrastructureError, match="pinned base policy"):
        validate_prerequisites(config, TaskCatalog(), "image", {})
