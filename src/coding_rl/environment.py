"""The lesson: reset a world, perform bounded tools, freeze it, verify, then clean up."""

import inspect
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType

from coding_rl.config import Config
from coding_rl.errors import ActionError, InfrastructureError
from coding_rl.patches import apply_diff, make_diff
from coding_rl.sandbox import Sandbox
from coding_rl.tasks import Task, TaskCatalog, file_digest, relative_path
from coding_rl.trace import Recorder
from coding_rl.verifier import VERIFIER_VERSION, Verdict, grading_fingerprint, verify

TOOLS = {
    "list_files": {"directory": "relative directory, default '.'"},
    "read_file": {
        "path": "relative text file",
        "start_line": "1-based, default 1",
        "max_lines": "positive integer, default 100",
    },
    "apply_patch": {"patch": "exact-context unified diff for existing allowed source"},
    "run_tests": {},
}
REASONS = {
    "final_response",
    "tool_limit",
    "episode_timeout",
    "scripted_finish",
    "cancelled",
    "turn_limit",
    "token_limit",
    "context_limit",
    "generation_timeout",
    "parser_limit",
    "infrastructure_failure",
}


class CodingEnvironment:
    """Framework-independent world; debugger/cleanup methods are not model tools.

    A future TRL adapter must expose only the four methods in TOOLS.
    """

    def __init__(
        self,
        config: Config,
        catalog: TaskCatalog | None = None,
        observer: Callable[[dict], None] | None = None,
    ):
        self.config = config
        self.catalog = catalog or TaskCatalog()
        self.sandbox = Sandbox(config.sandbox)
        self.recorder = Recorder(Path(config.logging.trace_dir), observer)
        self.recorder.artifact("config", asdict(config))
        self.task: Task | None = None
        self.phase = "empty"
        self.calls = 0
        self.episode_id: str | None = None
        self.workspace: Path | None = None
        self._owned: tempfile.TemporaryDirectory | None = None
        self._submission: dict[str, str] | None = None
        self._verdict: Verdict | None = None
        self._reward_published = False
        self._invalid = False
        self._termination: str | None = None
        self._started = 0.0
        self._revision = 0
        self.cleanup_seconds = 0.0

    def reset(self, task_id: str) -> None:
        """Start a fresh episode; return no duplicate dataset prompt."""
        task = self.catalog.get(task_id)  # Reject unknown IDs before allocating.
        self.cleanup()
        self.task = task
        self.episode_id = uuid.uuid4().hex
        self.calls = 0
        self._submission = None
        self._verdict = None
        self._reward_published = False
        self._invalid = False
        self._termination = None
        self._started = time.monotonic()
        self.cleanup_seconds = 0.0
        self.recorder.start(self.episode_id, task_id)
        self.phase = "empty"
        try:
            self._owned = tempfile.TemporaryDirectory(prefix="coding-rl-episode-")
            self._commit(task.starter)
            if file_digest(self._capture()) != task.starter_hash:
                raise InfrastructureError("Reset did not reproduce authored starter")
            self.phase = "active"
            self.recorder.emit(
                "reset",
                starter_hash=task.starter_hash,
                split=task.split,
                backend=self.config.sandbox.kind,
                max_tool_calls=self.config.environment.max_tool_calls,
            )
        except BaseException:
            self.cleanup()
            raise

    def _commit(self, files: dict | MappingProxyType) -> None:
        assert self._owned
        staged = Path(self._owned.name) / uuid.uuid4().hex
        staged.mkdir()
        try:
            for name, content in files.items():
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            old = self.workspace
            # One pointer swap commits the entire validated revision. Workers
            # only see disposable snapshots; no subprocess holds this path.
            self.workspace = staged
            self._revision += 1
            if old:
                shutil.rmtree(old)
        except BaseException:
            if staged != self.workspace:
                shutil.rmtree(staged)
            raise

    def _read(self, name: str) -> str:
        assert self.task and self.workspace
        try:
            relative_path(name)
        except ValueError as exc:
            raise ActionError(str(exc)) from exc
        if name not in self.task.starter:
            raise ActionError("Path is outside the task's public file allowlist")
        path = self.workspace
        try:
            for component in name.split("/"):
                path = path / component
                if path.is_symlink():
                    raise ActionError("Symbolic links are forbidden")
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ActionError("Only unlinked regular text files are supported")
            if metadata.st_mode & 0o111:
                raise ActionError("Executable-mode changes are forbidden")
            if metadata.st_size > self.config.environment.max_file_bytes:
                raise ActionError("File exceeds byte budget")
            return path.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ActionError("Cannot read allowed UTF-8 source safely") from exc

    def _capture(self) -> dict[str, str]:
        assert self.task and self.workspace
        if self.workspace.is_symlink():
            raise ActionError("Workspace must not be a symbolic link")
        files = {name: self._read(name) for name in self.task.starter}
        for path in self.workspace.rglob("*"):
            if path.is_symlink() or (
                not path.is_dir() and path.relative_to(self.workspace).as_posix() not in files
            ):
                raise ActionError("Unexpected files or links in source snapshot")
        if any(
            files[name] != value
            for name, value in self.task.starter.items()
            if name not in self.task.editable
        ):
            raise ActionError("Protected task files changed")
        return files

    def _tool(self, name: str, arguments: dict, operation: Callable[[], dict]) -> dict:
        if self.task is None:
            return {"ok": False, "error": "inactive", "message": "Reset a task first"}
        # Bound call text in the human trace as well as observations.
        logged = {
            key: value[: self.config.environment.max_patch_bytes]
            if isinstance(value, str)
            else value
            for key, value in arguments.items()
        }
        self.recorder.emit("action", tool=name, arguments=logged)
        revision_before = self._revision
        try:
            if self.phase != "active":
                raise ActionError("Episode is frozen or closed")
            if time.monotonic() - self._started >= self.config.environment.episode_seconds:
                self.finish("episode_timeout")
                raise ActionError("Episode deadline exhausted")
            if self.calls >= self.config.environment.max_tool_calls:
                self.finish("tool_limit")
                raise ActionError("Tool-call budget exhausted")
            self.calls += 1
            result = {"ok": True, **operation()}
        except ActionError as exc:
            result = {"ok": False, "error": "invalid_action", "message": str(exc)}
        except InfrastructureError:
            self.recorder.emit("infrastructure_error", operation=name)
            raise
        except OSError as exc:
            self.recorder.emit(
                "infrastructure_error",
                operation=name,
                revision_committed=self._revision != revision_before,
            )
            raise InfrastructureError(f"Trusted filesystem operation failed during {name}") from exc
        result["remaining_calls"] = max(0, self.config.environment.max_tool_calls - self.calls)
        self.recorder.emit("observation", tool=name, result=result)
        return result

    def list_files(self, directory: str = ".") -> dict:
        """List public repository files without modifying state.

        Args:
            directory: A relative directory, or '.' for the repository root.
        """

        def operation() -> dict:
            try:
                normalized = relative_path(directory, root=True)
            except ValueError as exc:
                raise ActionError(str(exc)) from exc
            files = self._capture()
            names = sorted(
                name for name in files if normalized == "." or name.startswith(normalized + "/")
            )
            cap = self.config.environment.max_list_entries
            return {"files": names[:cap], "truncated": len(names) > cap}

        return self._tool("list_files", {"directory": directory}, operation)

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 100) -> dict:
        """Read bounded public file contents.

        Args:
            path: A permitted relative UTF-8 file path.
            start_line: First line to read, numbered from one.
            max_lines: Maximum positive number of lines to return.
        """

        def operation() -> dict:
            if (
                type(start_line) is not int
                or type(max_lines) is not int
                or min(start_line, max_lines) <= 0
            ):
                raise ActionError("Line bounds must be positive integers")
            lines = self._read(path).splitlines(keepends=True)
            content = "".join(lines[start_line - 1 : start_line - 1 + max_lines])
            raw = content.encode("utf-8")
            cap = self.config.sandbox.output_bytes
            return {
                "content": raw[:cap].decode("utf-8", errors="ignore"),
                "truncated": len(raw) > cap or start_line - 1 + max_lines < len(lines),
            }

        return self._tool(
            "read_file", {"path": path, "start_line": start_line, "max_lines": max_lines}, operation
        )

    def apply_patch(self, patch: str) -> dict:
        """Atomically edit existing allowed source with an exact unified diff.

        Args:
            patch: Unified diff using a/path and b/path headers, without Git directives.
        """

        def operation() -> dict:
            assert self.task
            if not isinstance(patch, str):
                raise ActionError("Patch must be UTF-8 text")
            try:
                encoded = patch.encode("utf-8")
            except UnicodeError as exc:
                raise ActionError("Patch must be valid UTF-8 text") from exc
            if len(encoded) > self.config.environment.max_patch_bytes:
                raise ActionError("Patch exceeds byte budget or is not text")
            before = self._capture()
            after = apply_diff(before, self.task.editable, patch)
            if any(
                len(value.encode()) > self.config.environment.max_file_bytes
                for value in after.values()
            ):
                raise ActionError("Patched file exceeds byte budget")
            self._commit(after)
            return {
                "changed_paths": [name for name in after if before[name] != after[name]],
                "source_hash": file_digest(after),
            }

        return self._tool("apply_patch", {"patch": patch}, operation)

    def run_tests(self) -> dict:
        """Run the fixed visible suite in a fresh execution snapshot."""

        def operation() -> dict:
            assert self.task
            remaining = self.config.environment.episode_seconds - (time.monotonic() - self._started)
            if remaining <= 0:
                self.finish("episode_timeout")
                raise ActionError("Episode deadline exhausted")
            result = self.sandbox.visible_tests(self.task, self._capture(), timeout=remaining)
            return asdict(result)

        return self._tool("run_tests", {}, operation)

    def finish(self, reason: str = "final_response") -> None:
        """Developer/harness lifecycle operation; never expose it as a public tool."""
        if reason not in REASONS:
            raise ValueError("Unknown termination reason")
        if self.phase != "active":
            return
        assert self.task
        self._termination = reason
        self.phase = "frozen"
        self.recorder.emit("finish", reason=reason, tool_calls=self.calls)
        try:
            self._submission = self._capture()
        except ActionError:
            self._submission = {}
            self._invalid = True
        artifact = self.recorder.artifact(
            "submission",
            {
                "files": self._submission,
                "diff": "" if self._invalid else make_diff(self.task.starter, self._submission),
                "starter_hash": self.task.starter_hash,
                "submission_hash": file_digest(self._submission),
            },
        )
        self.recorder.emit(
            "submission",
            artifact=artifact,
            submission_hash=file_digest(self._submission),
            valid=not self._invalid,
        )

    def get_reward(self) -> float:
        """Freeze, independently judge once, cache the completed verdict, and release resources."""
        if self._verdict is not None:
            self.cleanup()
            self._publish_reward()
            return self._verdict.reward
        if self.task is None:
            raise ValueError("Reset before asking for reward")
        self.finish()
        if self._submission is None:
            raise ValueError("Episode closed without a frozen submission")
        deadline = time.monotonic() + self.config.environment.grading_seconds
        digest = file_digest(self._submission)
        try:
            if self._invalid:
                self._verdict = Verdict(
                    0.0, "invalid_submission", 0, len(self.task.private_cases), digest
                )
            else:
                for attempt in range(self.config.environment.verifier_retries + 1):
                    try:
                        self._verdict = verify(
                            self.task,
                            MappingProxyType(self._submission),
                            self.sandbox,
                            submission_hash=digest,
                            deadline=deadline,
                        )
                        break
                    except InfrastructureError:
                        self.recorder.emit(
                            "infrastructure_error", operation="verify", attempt=attempt + 1
                        )
                        if attempt == self.config.environment.verifier_retries:
                            raise
            assert self._verdict
            self.phase = "judged"
            self.recorder.emit(
                "verify",
                **asdict(self._verdict),
                verifier_version=VERIFIER_VERSION,
                grading_fingerprint=grading_fingerprint(self.task),
                worker_identity=self.sandbox.identity,
            )
        finally:
            self.cleanup()
        self._publish_reward()
        return self._verdict.reward

    def _publish_reward(self) -> None:
        if not self._reward_published:
            assert self._verdict is not None and self._owned is None
            self.recorder.emit("reward", value=self._verdict.reward)
            self._reward_published = True

    def inspect(self) -> dict:
        """Developer-only state view; excludes private cases and host workspace paths."""
        snapshot = self._submission
        if self.phase == "active":
            snapshot = self._capture()
        return {
            "episode_id": self.episode_id,
            "task_id": self.task.task_id if self.task else None,
            "phase": self.phase,
            "tool_calls": self.calls,
            "remaining_calls": max(0, self.config.environment.max_tool_calls - self.calls),
            "remaining_seconds": max(
                0, self.config.environment.episode_seconds - (time.monotonic() - self._started)
            ),
            "allowed_paths": list(self.task.editable) if self.task else [],
            "tools": tool_schemas(),
            "termination_reason": self._termination,
            "source_hash": file_digest(snapshot) if snapshot is not None else None,
            "starter_hash": self.task.starter_hash if self.task else None,
            "diff": make_diff(self.task.starter, snapshot)
            if self.task and snapshot and not self._invalid
            else "",
            "verdict": asdict(self._verdict) if self._verdict else None,
        }

    def cleanup(self) -> None:
        if self._owned is None:
            return
        started = time.monotonic()
        try:
            self._owned.cleanup()
        except OSError as exc:
            self.recorder.emit("infrastructure_error", operation="cleanup")
            raise InfrastructureError("Cannot release episode workspace") from exc
        finally:
            self.cleanup_seconds += time.monotonic() - started
        self._owned = None
        self.workspace = None
        self.phase = "closed"
        self.recorder.emit("cleanup", released=True)


def make_environment(
    config: Config | None = None,
    *,
    catalog: TaskCatalog | None = None,
    observer: Callable[[dict], None] | None = None,
) -> CodingEnvironment:
    return CodingEnvironment(config or Config(), catalog, observer)


def tool_schemas() -> dict:
    """Describe only allowed tools, rather than discovering all public core methods."""
    result = {}
    for name, descriptions in TOOLS.items():
        method = getattr(CodingEnvironment, name)
        properties, required = {}, []
        for parameter in inspect.signature(method).parameters.values():
            if parameter.name == "self":
                continue
            schema = {
                "type": {str: "string", int: "integer"}[parameter.annotation],
                "description": descriptions[parameter.name],
            }
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
            else:
                schema["default"] = parameter.default
            properties[parameter.name] = schema
        result[name] = {
            "description": inspect.getdoc(method).split("\n\n")[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
    return result
