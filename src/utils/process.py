"""Bounded subprocess supervision and process resource measurements."""

import os
import resource
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from utils.errors import InfrastructureError


@dataclass(frozen=True)
class ProcessResult:
    status: str
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def supervise(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    output_bytes: int,
    input_bytes: bytes = b"",
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """Stream bounded output and kill the entire local process group on a limit."""
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise InfrastructureError("Cannot start sandbox process") from exc
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    status = "completed"
    selector = selectors.DefaultSelector()
    try:
        assert process.stdin and process.stdout and process.stderr
        # Only trusted worker requests are written; the worker reads before it
        # imports candidate code. Requests are deliberately smaller than a pipe.
        if len(input_bytes) > 4096:
            raise InfrastructureError("Authored worker input exceeds 4096 bytes")
        try:
            process.stdin.write(input_bytes)
            process.stdin.close()
        except BrokenPipeError:
            pass
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map() or process.poll() is None:
            if time.monotonic() - started >= timeout and status == "completed":
                status = "timeout"
                _kill_group(process.pid)
            for key, _ in selector.select(timeout=0.05):
                data = os.read(key.fileobj.fileno(), 4096)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                remaining = output_bytes - sum(len(value) for value in streams.values())
                streams[key.data].extend(data[: max(0, remaining)])
                if len(data) > remaining and status == "completed":
                    status = "output_limit"
                    _kill_group(process.pid)
        process.wait(timeout=5)
    finally:
        _kill_group(process.pid)
        if process.poll() is None:
            process.wait(timeout=5)
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()
    return ProcessResult(
        status,
        process.returncode,
        streams["stdout"].decode("utf-8", errors="replace"),
        streams["stderr"].decode("utf-8", errors="replace"),
        time.monotonic() - started,
    )


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024
