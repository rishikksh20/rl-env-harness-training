"""Process ownership checks use tiny workers, without importing model libraries."""

import importlib.metadata
import multiprocessing
import subprocess
import sys
import time

import pytest

from coding_rl.config import RolloutConfig
from coding_rl.errors import InfrastructureError
from coding_rl.inference_worker import receive_record, send_record
from coding_rl.models import GenerationTimeout, TransformersBackend, dependency_versions


def sleeping_worker(connection):
    receive_record(connection)
    time.sleep(30)


def mismatched_worker(connection):
    request = receive_record(connection)
    send_record(
        connection,
        {
            "kind": "generation",
            "id": "late-" + request["id"],
            "token_ids": [1],
            "stop_reason": "eos",
        },
    )
    connection.close()


def crashed_worker(connection):
    receive_record(connection)
    connection.close()


def make_worker(target):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.process = context.Process(target=target, args=(child,))
    backend.connection = parent
    backend.tokenizer = [None] * 8
    backend.process.start()
    child.close()
    return backend


@pytest.mark.parametrize(
    "target,error",
    [
        (sleeping_worker, GenerationTimeout),
        (mismatched_worker, InfrastructureError),
        (crashed_worker, InfrastructureError),
    ],
)
def test_hard_deadline_crash_and_late_response_release_worker(target, error):
    backend = make_worker(target)
    process = backend.process
    started = time.monotonic()
    try:
        with pytest.raises(error):
            backend.generate(
                [1],
                max_new_tokens=2,
                seed=42,
                timeout=0.5 if target == sleeping_worker else 5,
                sampling=RolloutConfig(),
            )
        assert time.monotonic() - started < 8
        assert backend.process is None and backend.connection is None
        assert process._closed
    finally:
        backend.close()


def test_base_import_does_not_load_inference_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, coding_rl; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_extra_has_actionable_diagnostic(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(InfrastructureError, match="uv sync --extra rollout"):
        dependency_versions()
