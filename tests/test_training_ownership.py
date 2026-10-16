"""Cancellation releases only resources created under the parent's private run root."""

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from coding_rl.training_worker import _stop, cleanup_resources


def ownership_child(root, stage):
    root = Path(root)
    private = root / "private"
    if stage == "checkpoint":
        partial = root / "checkpoints" / ".checkpoint-interrupted"
        partial.mkdir(parents=True)
        (partial / "model.safetensors").write_bytes(b"incomplete")
    else:
        episode = private / "coding-rl-episode-owned"
        episode.mkdir()
        (episode / "source.py").write_text("candidate")
        if stage in {"visible_tests", "grading"}:
            snapshot = private / "coding-rl-snapshot-owned"
            snapshot.mkdir()
            (snapshot / "source.py").write_text("frozen")
    (root / "ready").touch()
    time.sleep(30)


def supervised_child(connection, config, run_id, directory, probe_only):
    ownership_child(directory, "checkpoint")


@pytest.mark.parametrize("stage", ["generation", "visible_tests", "grading", "checkpoint"])
def test_forced_child_termination_releases_private_state(tmp_path, stage):
    pytest.importorskip("psutil", reason="Parent process supervision belongs to the training extra")
    (tmp_path / "private").mkdir()
    process = multiprocessing.get_context("spawn").Process(
        target=ownership_child, args=(str(tmp_path), stage)
    )
    process.start()
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        _stop(process)
        assert not process.is_alive()
        cleanup_resources(tmp_path, "a" * 32)
        assert not (tmp_path / "private").exists()
        assert not list((tmp_path / "checkpoints").glob(".checkpoint-*"))
    finally:
        if process.is_alive():
            _stop(process)
        process.close()


def test_supervisor_deadline_keeps_interrupted_checkpoint_unusable(config, tmp_path, monkeypatch):
    from dataclasses import replace

    from coding_rl import trainer as trainer_module
    from coding_rl import training_worker as module
    from coding_rl.config import ModelConfig, TrainingConfig

    pytest.importorskip("psutil")
    monkeypatch.setattr(trainer_module, "dependency_report", lambda mode: {})
    monkeypatch.setattr(module, "_worker_main", supervised_child)
    config = replace(
        config,
        model=ModelConfig(),
        sandbox=replace(config.sandbox, kind="docker"),
        training=TrainingConfig(output_dir=str(tmp_path / "runs"), wall_seconds=1),
    )
    result = module.train(config)
    assert result["status"] == "incomplete"
    assert "wall_seconds" in result["parent_error"]
    directory = Path(result["directory"])
    assert (directory / "ready").exists()
    assert not (directory / "private").exists()
    assert not list((directory / "checkpoints").glob(".checkpoint-*"))


def docker_child(root, run_id, image):
    from coding_rl.config import SandboxConfig
    from coding_rl.sandbox import Sandbox, materialize

    root = Path(root)
    sandbox = Sandbox(SandboxConfig(image=image), resource_root=root / "private", run_id=run_id)
    sandbox.identity = image
    with materialize({"source.py": "pass\n"}, parent=root / "private") as snapshot:
        sandbox._docker_run(snapshot, ["-I", "-c", "import time; time.sleep(30)"], timeout=40)


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("CODING_RL_DOCKER_TESTS") != "1",
    reason="Enable real Docker checks with CODING_RL_DOCKER_TESTS=1",
)
def test_forced_training_child_stop_removes_live_registered_container(tmp_path):
    import json
    import subprocess
    import uuid

    pytest.importorskip("psutil")
    run_id = uuid.uuid4().hex
    private = tmp_path / "private"
    private.mkdir()
    image = "coding-rl-sandbox:stage-a"
    process = multiprocessing.get_context("spawn").Process(
        target=docker_child, args=(str(tmp_path), run_id, image)
    )
    process.start()
    journal = private / "containers.jsonl"
    name = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if journal.exists():
                name = json.loads(journal.read_text().splitlines()[0])
                running = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if running.stdout.strip() == "true":
                    break
            time.sleep(0.05)
        else:
            pytest.fail("Owned Docker worker did not start")
        _stop(process)
        cleanup_resources(tmp_path, run_id)
        result = subprocess.run(["docker", "inspect", name], capture_output=True, check=False)
        assert result.returncode != 0
        assert not private.exists()
    finally:
        if process.is_alive():
            _stop(process)
        cleanup_resources(tmp_path, run_id)
        process.close()
