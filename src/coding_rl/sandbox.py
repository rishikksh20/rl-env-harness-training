"""Bounded subprocess supervision and Docker execution of immutable snapshots."""

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from coding_rl.config import SandboxConfig
from coding_rl.errors import InfrastructureError
from coding_rl.tasks import Task, relative_path

WORKER = Path(__file__).with_name("worker.py")
VISIBLE_ARGS = [
    "-I",
    "-m",
    "pytest",
    "-q",
    "-c",
    "/dev/null",
    "-p",
    "no:cacheprovider",
    "--import-mode=prepend",
    "test_visible.py",
]
CONTROLLED_ENV = {
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}


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


@contextmanager
def materialize(files: Mapping[str, str], *, parent: Path | None = None) -> Iterator[Path]:
    """An owned temporary snapshot: never mount the authoritative working copy."""
    with tempfile.TemporaryDirectory(prefix="coding-rl-snapshot-", dir=parent) as directory:
        root = Path(directory)
        root.chmod(0o755)  # The non-root container user needs to read the bind mount.
        for name, content in files.items():
            relative_path(name)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            path.chmod(0o444)
        yield root


class Sandbox:
    """Two real strategies; trusted-local refuses every non-authored snapshot."""

    def __init__(
        self,
        config: SandboxConfig,
        *,
        resource_root: Path | None = None,
        run_id: str | None = None,
    ):
        self.config = config
        self.resource_root = resource_root
        self.run_id = run_id
        if run_id is not None and (
            len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id)
        ):
            raise ValueError("Invalid sandbox run identity")
        self.identity: str | None = None

    def _docker_control(self, arguments: list[str]) -> str:
        try:
            result = subprocess.run(
                ["docker", *arguments], capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError("Docker control command is unavailable") from exc
        if result.returncode:
            raise InfrastructureError(
                "Docker unavailable, image missing, or control command failed"
            )
        return result.stdout.strip()

    def preflight(self) -> None:
        if self.identity:
            return
        if self.config.kind == "trusted_local":
            self.identity = (
                f"trusted-fixtures-only:python-{sys.version_info.major}.{sys.version_info.minor}"
            )
            return
        info = json.loads(self._docker_control(["info", "--format", "{{json .}} "]))
        if info.get("CgroupVersion") != "2" or not all(
            info.get(key) for key in ("MemoryLimit", "PidsLimit")
        ):
            raise InfrastructureError(
                "Stage A Docker requires cgroup v2 memory, CPU, and PID limits"
            )
        identity = self._docker_control(
            ["image", "inspect", "--format", "{{.Id}}", self.config.image]
        )
        self.identity = identity
        # Verify effective limits inside a worker, rather than trusting flags.
        probe = (
            "import os,pathlib,pytest; p=pathlib.Path('/sys/fs/cgroup'); "
            f"assert os.getuid()!=0; assert (p/'memory.max').read_text().strip()=='{self.config.memory_mb * 1048576}'; "
            f"assert (p/'pids.max').read_text().strip()=='{self.config.pids_limit}'; "
            f"q,period=map(int,(p/'cpu.max').read_text().split()); assert abs(q/period-{self.config.cpus})<0.001; "
            "assert pathlib.Path('/workspace/probe').read_text()=='ready'; "
            "assert pathlib.Path('/opt/worker.py').is_file(); print('ready')"
        )
        try:
            with materialize({"probe": "ready"}, parent=self.resource_root) as root:
                result = self._docker_run(root, ["-I", "-c", probe], timeout=20)
            if (
                result.status != "completed"
                or result.returncode
                or result.stdout.strip() != "ready"
            ):
                raise InfrastructureError("Docker effective-limit/worker preflight failed")
        except BaseException:
            self.identity = None
            raise

    def _docker_run(
        self, root: Path, arguments: list[str], *, timeout: float, input_bytes: bytes = b""
    ) -> ProcessResult:
        name = f"coding-rl-{self.run_id + '-' if self.run_id else ''}{uuid.uuid4().hex}"
        if self.resource_root is not None and self.run_id is not None:
            # Persist ownership before launch so a parent can clean after SIGKILL.
            with (self.resource_root / "containers.jsonl").open("a") as stream:
                stream.write(json.dumps(name) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        cfg = self.config
        command = [
            "docker",
            "run",
            "--name",
            name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--memory={cfg.memory_mb}m",
            f"--memory-swap={cfg.memory_mb}m",
            f"--cpus={cfg.cpus}",
            f"--pids-limit={cfg.pids_limit}",
            "--log-driver=none",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=16777216",
            "--interactive",
            "--mount",
            f"type=bind,src={root},dst=/workspace,readonly",
            "--workdir=/workspace",
            "--entrypoint=/opt/lab/.venv/bin/python",
        ]
        if self.run_id is not None:
            command.extend(["--label", f"coding-rl.run={self.run_id}"])
        for key, value in CONTROLLED_ENV.items():
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.identity or cfg.image, *arguments])
        try:
            result = supervise(
                command,
                cwd=root,
                timeout=timeout,
                output_bytes=cfg.output_bytes,
                input_bytes=input_bytes,
            )
            if result.status == "completed":
                state = json.loads(
                    self._docker_control(["inspect", "--format", "{{json .State}}", name])
                )
                # A submitted function can itself exit with Docker-reserved
                # codes. Inspect trusted container startup state, not the
                # candidate's chosen exit number, to classify infrastructure.
                if state.get("Error") or state.get("StartedAt", "").startswith("0001-"):
                    raise InfrastructureError("Docker failed to start the trusted worker")
                if state.get("OOMKilled"):
                    result = ProcessResult(
                        "memory_limit",
                        result.returncode,
                        result.stdout,
                        result.stderr,
                        result.seconds,
                    )
            return result
        finally:
            # Removing the named container kills candidate descendants even
            # when the Docker client itself timed out or was interrupted.
            self._docker_control(["rm", "--force", name])

    def _run(
        self,
        task: Task,
        files: Mapping[str, str],
        arguments: list[str],
        *,
        timeout: float | None = None,
        input_bytes: bytes = b"",
    ) -> ProcessResult:
        if self.config.kind == "trusted_local" and not task.is_authored_snapshot(files):
            raise InfrastructureError("trusted_local refuses non-authored source; use Docker")
        self.preflight()
        limit = (
            min(self.config.timeout_seconds, timeout)
            if timeout is not None
            else self.config.timeout_seconds
        )
        if limit <= 0:
            raise InfrastructureError("Verification deadline exhausted")
        with materialize(files, parent=self.resource_root) as root:
            if self.config.kind == "docker":
                return self._docker_run(root, arguments, timeout=limit, input_bytes=input_bytes)
            env = {"PATH": os.defpath, **CONTROLLED_ENV}
            return supervise(
                [sys.executable, *arguments],
                cwd=root,
                timeout=limit,
                output_bytes=self.config.output_bytes,
                input_bytes=input_bytes,
                env=env,
            )

    def visible_tests(
        self, task: Task, files: Mapping[str, str], *, timeout: float | None = None
    ) -> ProcessResult:
        result = self._run(task, files, VISIBLE_ARGS, timeout=timeout)
        # Submitted Python can provoke any pytest exit code (including skip,
        # internal-error, and usage codes). A healthy prepared worker returning
        # nonzero is an evaluated failure, not evidence of an engine outage.
        return result

    def invoke(
        self, task: Task, files: Mapping[str, str], kwargs: dict, *, timeout: float | None = None
    ) -> ProcessResult:
        request = json.dumps(
            {"module": task.module, "function": task.function, "kwargs": kwargs}, allow_nan=False
        ).encode()
        worker_path = "/opt/worker.py" if self.config.kind == "docker" else str(WORKER)
        return self._run(task, files, ["-I", worker_path], timeout=timeout, input_bytes=request)
