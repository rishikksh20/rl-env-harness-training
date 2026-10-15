import json
from dataclasses import replace
from pathlib import Path

import pytest

from coding_rl import make_environment, run_episode
from coding_rl.errors import InfrastructureError
from coding_rl.patches import make_diff


def repair(env, task_id="zero-timeout", variant="correct"):
    env.reset(task_id)
    task = env.catalog.get(task_id)
    patch = make_diff(task.starter, task.variant(variant))
    if patch:
        assert env.apply_patch(patch)["ok"]


def test_fresh_reset_and_independent_worlds(env, config):
    other = make_environment(config)
    try:
        env.reset("zero-timeout")
        other.reset("zero-timeout")
        original = other.inspect()["source_hash"]
        first_id = env.episode_id
        first_workspace = env.workspace
        task = env.task
        assert env.apply_patch(make_diff(task.starter, task.variant("correct")))["ok"]
        assert env.inspect()["source_hash"] != original
        assert other.inspect()["source_hash"] == original
        assert env.workspace != other.workspace
        env.reset("zero-timeout")
        assert env.episode_id != first_id
        assert env.inspect()["source_hash"] == original
        assert not first_workspace.exists()
    finally:
        other.cleanup()


def test_four_same_task_states_and_actual_rewards(config):
    environments = [make_environment(config) for _ in range(4)]
    try:
        for env in environments:
            env.reset("zero-timeout")
        assert len({env.episode_id for env in environments}) == 4
        assert len({env.workspace for env in environments}) == 4
        assert len({env.inspect()["source_hash"] for env in environments}) == 1
        for index in (0, 2):
            env = environments[index]
            assert env.apply_patch(make_diff(env.task.starter, env.task.variant("correct")))["ok"]
        assert [env.get_reward() for env in environments] == [1.0, 0.0, 1.0, 0.0]
        assert all(env.workspace is None for env in environments)
    finally:
        for env in environments:
            env.cleanup()


@pytest.mark.parametrize(
    "path",
    [
        "../private.json",
        "/etc/passwd",
        "timeout.py/../README.md",
        "a\\timeout.py",
        "./timeout.py",
        "private.json",
        "a//b",
        "a\x00b",
    ],
)
def test_path_bounds(env, path):
    env.reset("zero-timeout")
    result = env.read_file(path)
    assert not result["ok"]
    assert env.calls == 1


def test_symlinks_and_protected_edits_rejected(env, tmp_path):
    env.reset("zero-timeout")
    module = env.workspace / "timeout.py"
    module.unlink()
    secret = tmp_path / "secret"
    secret.write_text("never shown")
    module.symlink_to(secret)
    assert not env.read_file("timeout.py")["ok"]
    assert env.get_reward() == 0.0
    assert env.inspect()["verdict"]["status"] == "invalid_submission"
    env.reset("zero-timeout")
    protected = env.workspace / "test_visible.py"
    protected.write_text("raise RuntimeError('modified test')\n")
    assert env.get_reward() == 0.0
    assert env.workspace is None


def test_hardlink_and_executable_modes_rejected(env, tmp_path):
    env.reset("zero-timeout")
    module = env.workspace / "timeout.py"
    (tmp_path / "alias").hardlink_to(module)
    assert not env.read_file("timeout.py")["ok"]
    env.reset("zero-timeout")
    (env.workspace / "timeout.py").chmod(0o755)
    assert env.get_reward() == 0.0


def test_failed_patch_is_atomic_and_recoverable(env):
    env.reset("zero-timeout")
    task = env.task
    good = make_diff(task.starter, task.variant("correct"))
    before = env.inspect()["source_hash"]
    assert not env.apply_patch(good + good)["ok"]
    assert env.inspect()["source_hash"] == before
    assert env.apply_patch(good)["ok"]
    assert env.get_reward() == 1.0


def test_protected_test_patch_is_rejected(env):
    env.reset("zero-timeout")
    task = env.task
    changed = {**task.starter, "test_visible.py": "def test_forged():\n    assert True\n"}
    assert not env.apply_patch(make_diff(task.starter, changed))["ok"]
    assert env.inspect()["source_hash"] == task.starter_hash


def test_invalid_utf8_patch_is_a_recoverable_action(env):
    env.reset("zero-timeout")
    assert not env.apply_patch("\ud800\n")["ok"]
    assert env.phase == "active"
    task = env.task
    assert env.apply_patch(make_diff(task.starter, task.variant("correct")))["ok"]
    assert env.get_reward() == 1.0


def test_budget_counts_rejected_core_calls_and_freezes(config):
    config = replace(config, environment=replace(config.environment, max_tool_calls=2))
    env = make_environment(config)
    try:
        env.reset("zero-timeout")
        assert not env.read_file("../private")["ok"]
        assert env.read_file("timeout.py")["ok"]
        assert not env.apply_patch("bad")["ok"]
        assert env.calls == 2
        assert env.inspect()["termination_reason"] == "tool_limit"
        assert env.get_reward() == 0.0
    finally:
        env.cleanup()


def test_read_and_patch_caps_and_line_bounds(config):
    config = replace(
        config,
        sandbox=replace(config.sandbox, output_bytes=12),
        environment=replace(config.environment, max_patch_bytes=12),
    )
    env = make_environment(config)
    try:
        env.reset("zero-timeout")
        result = env.read_file("timeout.py")
        assert result["truncated"] and len(result["content"].encode()) <= 12
        assert not env.read_file("timeout.py", start_line=True)["ok"]
        assert not env.apply_patch("x" * 13)["ok"]
        assert env.inspect()["source_hash"] == env.task.starter_hash
    finally:
        env.cleanup()


def test_terminal_result_is_cached_and_no_mutation_after_freeze(env, monkeypatch):
    repair(env)
    frozen_path = env.workspace
    env.finish("scripted_finish")
    assert not env.apply_patch("anything")["ok"]
    assert env.get_reward() == 1.0
    monkeypatch.setattr(
        env.sandbox, "invoke", lambda *a, **k: pytest.fail("cached verdict reran worker")
    )
    assert env.get_reward() == 1.0
    assert env.phase == "closed" and env.workspace is None
    assert not frozen_path.exists()
    records = [
        json.loads(line)
        for line in (env.recorder.root / f"{env.episode_id}.jsonl").read_text().splitlines()
    ]
    assert sum(record["event"] == "reward" for record in records) == 1


def test_visible_pass_is_not_hidden_success(env):
    repair(env, "range-boundary", "visible_only")
    visible = env.run_tests()
    assert visible["returncode"] == 0
    assert env.get_reward() == 0.0
    assert env.inspect()["verdict"]["status"] == "incorrect"


def test_infrastructure_retry_same_frozen_snapshot_no_false_reward(env, monkeypatch):
    repair(env)
    original = env.sandbox.visible_tests
    calls = []

    def outage(task, snapshot, **kwargs):
        calls.append(dict(snapshot))
        raise InfrastructureError("simulated engine outage")

    monkeypatch.setattr(env.sandbox, "visible_tests", outage)
    with pytest.raises(InfrastructureError):
        env.get_reward()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert env.inspect()["verdict"] is None
    assert env.workspace is None
    monkeypatch.setattr(env.sandbox, "visible_tests", original)
    assert env.get_reward() == 1.0
    records = (env.recorder.root / f"{env.episode_id}.jsonl").read_text()
    assert sum(json.loads(line)["event"] == "reward" for line in records.splitlines()) == 1


def test_local_backend_refuses_untrusted_edits(env):
    env.reset("zero-timeout")
    changed = {**env.task.starter, "timeout.py": "raise RuntimeError('unknown source')\n"}
    assert env.apply_patch(make_diff(env.task.starter, changed))["ok"]
    with pytest.raises(InfrastructureError, match="refuses non-authored"):
        env.run_tests()
    with pytest.raises(InfrastructureError, match="refuses non-authored"):
        env.get_reward()
    assert env.workspace is None and env.inspect()["verdict"] is None


def test_deadline_and_unknown_task_lifecycle(env):
    env.reset("zero-timeout")
    original_path = env.workspace
    with pytest.raises(ValueError, match="Unknown task"):
        env.reset("missing")
    assert env.workspace == original_path
    env._started -= env.config.environment.episode_seconds + 1
    assert not env.list_files()["ok"]
    assert env.inspect()["termination_reason"] == "episode_timeout"
    assert env.get_reward() == 0.0


def test_partial_reset_failure_releases_workspace(env, monkeypatch):
    def fail(_):
        raise OSError("simulated file allocation error")

    monkeypatch.setattr(env, "_commit", fail)
    with pytest.raises(OSError):
        env.reset("zero-timeout")
    assert env._owned is None and env.workspace is None


def test_all_authored_fixtures_and_trace_schema(env):
    for task in env.catalog.tasks():
        for variant, expected in (("correct", 1.0), ("baseline", 0.0)):
            summary = run_episode(env, task.task_id, variant=variant)
            assert summary.reward == expected
            assert env.workspace is None
            records = [
                json.loads(line)
                for line in (Path(summary.trace_directory) / f"{summary.episode_id}.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [r["sequence"] for r in records] == list(range(len(records)))
            assert all(
                r["schema_version"] == 2 and r["policy"]["kind"] == "scripted" for r in records
            )
            assert records[-1]["event"] == "reward"
            assert next(r["sequence"] for r in records if r["event"] == "cleanup") < next(
                r["sequence"] for r in records if r["event"] == "reward"
            )
            assert sum(r["event"] == "reward" for r in records) == 1
            assert not any(
                "private.json" in json.dumps(r) or "expected" in json.dumps(r) for r in records
            )


def test_cached_verdict_retries_cleanup_before_publishing_reward(env, monkeypatch):
    repair(env)
    original = env.cleanup

    def fail():
        raise InfrastructureError("simulated cleanup failure")

    monkeypatch.setattr(env, "cleanup", fail)
    with pytest.raises(InfrastructureError, match="cleanup"):
        env.get_reward()
    assert env._verdict.reward == 1.0 and env.workspace.exists()
    records = (env.recorder.root / f"{env.episode_id}.jsonl").read_text()
    assert '"event": "reward"' not in records
    monkeypatch.setattr(env, "cleanup", original)
    monkeypatch.setattr(env.sandbox, "invoke", lambda *a, **k: pytest.fail("reran verification"))
    assert env.get_reward() == env.get_reward() == 1.0
    assert env.workspace is None and env.phase == "closed"
    events = [
        json.loads(line)["event"]
        for line in (env.recorder.root / f"{env.episode_id}.jsonl").read_text().splitlines()
    ]
    assert events.count("verify") == events.count("reward") == 1
    assert events.index("cleanup") < events.index("reward")


def test_trusted_commit_io_failure_records_whether_revision_committed(env, monkeypatch):
    env.reset("zero-timeout")
    original = env._commit

    def fail_after_commit(files):
        original(files)
        raise OSError("old revision cleanup failed")

    monkeypatch.setattr(env, "_commit", fail_after_commit)
    with pytest.raises(InfrastructureError, match="filesystem"):
        env.apply_patch(make_diff(env.task.starter, env.task.variant("correct")))
    assert env.inspect()["source_hash"] != env.task.starter_hash
    records = [
        json.loads(line)
        for line in (env.recorder.root / f"{env.episode_id}.jsonl").read_text().splitlines()
    ]
    assert records[-1]["payload"]["revision_committed"] is True
