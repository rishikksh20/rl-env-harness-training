import json
import os
import sys
import time
from dataclasses import replace

import pytest

from coding_rl import make_environment, run_episode
from coding_rl.errors import InfrastructureError
from coding_rl.patches import make_diff
from coding_rl.sandbox import ProcessResult, Sandbox, supervise
from coding_rl.verifier import _json_value, _same, verify


def test_supervisor_timeout_and_descendants(tmp_path):
    code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(p.pid,flush=True); time.sleep(30)"
    )
    result = supervise(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, timeout=0.3, output_bytes=4096
    )
    assert result.status == "timeout" and result.seconds < 3
    child = int(result.stdout.strip())
    status = tmp_path.__class__(f"/proc/{child}/stat")
    # A killed child can briefly remain as a zombie until init reaps it.
    try:
        state = status.read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        state = None
    assert state in {None, "Z"}


def test_supervisor_bounds_output(tmp_path):
    result = supervise(
        [sys.executable, "-I", "-c", "print('x' * 1000000)"],
        cwd=tmp_path,
        timeout=3,
        output_bytes=100,
    )
    assert result.status == "output_limit"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 100


def test_supervisor_keeps_deadline_when_candidate_closes_output(tmp_path):
    code = "import os,time; os.close(1); os.close(2); time.sleep(30)"
    result = supervise(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, timeout=0.2, output_bytes=100
    )
    assert result.status == "timeout" and result.seconds < 3


def test_missing_executable_is_infrastructure(tmp_path):
    with pytest.raises(InfrastructureError):
        supervise([str(tmp_path / "does-not-exist")], cwd=tmp_path, timeout=1, output_bytes=100)


def test_private_controller_uses_pristine_protected_snapshot(env, monkeypatch):
    task = env.catalog.get("zero-timeout")
    snapshot = task.variant("correct")
    observed = []

    def visible(received_task, files, **kwargs):
        assert files is not snapshot
        assert files["test_visible.py"] == task.starter["test_visible.py"]
        return ProcessResult("completed", 0, "3 passed", "", 0)

    def invoke(received_task, files, inputs, **kwargs):
        assert set(files) == set(task.starter) and "private.json" not in files
        observed.append(dict(inputs))
        value = inputs.get("timeout")
        value = inputs.get("default", 30) if value is None else value
        return ProcessResult("completed", 0, json.dumps({"status": "ok", "value": value}), "", 0)

    monkeypatch.setattr(env.sandbox, "visible_tests", visible)
    monkeypatch.setattr(env.sandbox, "invoke", invoke)
    verdict = verify(
        task, snapshot, env.sandbox, submission_hash="test", deadline=time.monotonic() + 10
    )
    assert verdict.reward == 1.0 and len(observed) == len(task.private_cases)


@pytest.mark.parametrize(
    "output",
    [
        "3 passed",
        '{"status":"ok","value":0}\n3 passed',
        '{"status":"ok","value":NaN}',
        '{"status":"ok"}',
        '{"status":"ok","value":false}',
        '{"status":"ok","value":999,"value":0}',
    ],
)
def test_forged_or_invalid_worker_output_cannot_pass(env, monkeypatch, output):
    task = env.catalog.get("zero-timeout")
    monkeypatch.setattr(
        env.sandbox, "visible_tests", lambda *a, **k: ProcessResult("completed", 0, "", "", 0)
    )
    monkeypatch.setattr(
        env.sandbox, "invoke", lambda *a, **k: ProcessResult("completed", 0, output, "", 0)
    )
    verdict = verify(
        task,
        task.variant("correct"),
        env.sandbox,
        submission_hash="test",
        deadline=time.monotonic() + 10,
    )
    assert verdict.reward == 0.0


def test_strict_json_values():
    assert not _same(False, 0)
    assert not _same([False], [0])
    assert not _json_value(float("nan"))
    assert _same({"a": [0, False, None]}, {"a": [0, False, None]})


def test_grading_deadline_timeout_is_not_candidate_failure(env, monkeypatch):
    task = env.catalog.get("zero-timeout")
    monkeypatch.setattr(
        env.sandbox, "visible_tests", lambda *a, **k: ProcessResult("timeout", -9, "", "", 0)
    )
    with pytest.raises(InfrastructureError, match="Grading deadline"):
        verify(
            task,
            task.variant("correct"),
            env.sandbox,
            submission_hash="test",
            deadline=time.monotonic() + 1,
        )


docker = pytest.mark.skipif(
    os.environ.get("CODING_RL_DOCKER_TESTS") != "1",
    reason="Enable real Docker checks with CODING_RL_DOCKER_TESTS=1",
)


@pytest.fixture
def docker_env(config):
    config = replace(config, sandbox=replace(config.sandbox, kind="docker"))
    environment = make_environment(config)
    try:
        yield environment
    finally:
        environment.cleanup()


@docker
@pytest.mark.docker
def test_real_docker_reward_and_no_surviving_workers(docker_env):
    sandbox = docker_env.sandbox
    before = sandbox._docker_control(
        ["ps", "-a", "--filter", "name=coding-rl-", "--format", "{{.Names}}"]
    )
    assert run_episode(docker_env).reward == 1.0
    assert sandbox.identity.startswith("sha256:")
    assert (
        sandbox._docker_control(
            ["ps", "-a", "--filter", "name=coding-rl-", "--format", "{{.Names}}"]
        )
        == before
    )


@docker
@pytest.mark.docker
def test_real_docker_readonly_no_network_and_private_paths(docker_env):
    task = docker_env.catalog.get("zero-timeout")
    source = """def normalize_timeout(timeout=None, default=30):
    import os, pathlib, socket
    blocked = False
    try:
        pathlib.Path("timeout.py").write_text("tampered")
    except OSError:
        blocked = True
    connection = socket.socket()
    connection.settimeout(0.5)
    network = False
    try:
        connection.connect(("1.1.1.1", 80))
        network = True
    except OSError:
        pass
    finally:
        connection.close()
    return {"uid": os.getuid(), "readonly": blocked, "network": network,
            "files": sorted(p.name for p in pathlib.Path.cwd().iterdir())}
"""
    files = {**task.starter, task.module: source}
    result = docker_env.sandbox.invoke(task, files, {})
    value = json.loads(result.stdout)["value"]
    assert value["uid"] != 0 and value["readonly"] and not value["network"]
    assert value["files"] == sorted(task.starter)


@docker
@pytest.mark.docker
@pytest.mark.parametrize(
    "body, expected",
    [
        ("while True: pass", "timeout"),
        ("print('x' * 1000000); return 0", "output_limit"),
        ("import os; os.write(1, b'3 passed'); return 999", "candidate_protocol_error"),
        ("return bytearray(1024 * 1024 * 1024)", "memory_limit"),
        ("import os; os._exit(125)", "candidate_error"),
    ],
)
def test_real_docker_candidate_failures(docker_env, body, expected):
    if expected == "timeout":
        # Include container/pytest startup in the command budget; one second
        # can time out the visible runner before reaching the private loop.
        docker_env.sandbox = Sandbox(replace(docker_env.config.sandbox, timeout_seconds=3))
    docker_env.reset("zero-timeout")
    task = docker_env.task
    source = (
        "def normalize_timeout(timeout=None, default=30):\n"
        "    if timeout == 0:\n" + f"        {body}\n"
        "    return timeout or default\n"
    )
    files = {**task.starter, task.module: source}
    assert docker_env.apply_patch(make_diff(task.starter, files))["ok"]
    assert docker_env.get_reward() == 0.0
    assert docker_env.inspect()["verdict"]["status"] == expected


@docker
@pytest.mark.docker
def test_candidate_skip_does_not_become_infrastructure_error(docker_env):
    docker_env.reset("zero-timeout")
    task = docker_env.task
    source = "import pytest\npytest.skip('submitted skip', allow_module_level=True)\n"
    files = {**task.starter, task.module: source}
    assert docker_env.apply_patch(make_diff(task.starter, files))["ok"]
    assert docker_env.get_reward() == 0.0
    assert docker_env.inspect()["verdict"]["status"] == "visible_failed"
