"""Four-tool native adapter; the core remains the only source/reward authority."""

from collections.abc import Callable

from coding_rl.config import Config
from coding_rl.environment import CodingEnvironment
from coding_rl.errors import InfrastructureError
from coding_rl.protocol import ToolCall, dispatch
from coding_rl.rollout import bounded_observation, public_messages
from coding_rl.tasks import TaskCatalog


def training_rows(catalog: TaskCatalog) -> list[dict]:
    rows = []
    for task in catalog.tasks("train"):
        messages = public_messages(task.public())
        messages[0]["content"] = messages[0]["content"].replace(
            "Make one tool call per assistant turn.",
            "Calls in each turn execute sequentially. Every entered call uses the tool budget.",
        )
        rows.append({"task_id": task.task_id, "prompt": messages})
    return rows


class NativeEnvironment:
    """Only reset/get_reward are reserved; every other public method is a tool."""

    def __init__(self, core: CodingEnvironment, owner, presenter):
        self._core = core
        self._owner = owner
        self._presenter = presenter
        self._attempt = None

    def reset(self, task_id: str, **metadata) -> None:
        task = self._core.catalog.get(task_id)
        if task.split != "train":
            raise ValueError("Native training refuses held-out tasks")
        if "prompt" in metadata:
            expected = next(
                row["prompt"]
                for row in training_rows(self._core.catalog)
                if row["task_id"] == task_id
            )
            if metadata["prompt"] != expected:
                raise ValueError("Native public prompt does not match task identity")
        if self._attempt is not None and self._attempt.reward is None:
            self._owner.fail(self._attempt, "Attempt reset without a completed reward")
            raise InfrastructureError("Cannot reset an unfinished native attempt")
        self._core.reset(task_id)
        self._attempt = self._owner.register(self._core)
        return None

    def get_reward(self) -> float:
        if self._attempt is None:
            raise InfrastructureError("Reset before native reward")
        self._owner.check_failures()
        try:
            reward = self._core.get_reward()
            if self._core.workspace is not None:
                raise InfrastructureError("Native reward requires completed cleanup")
            self._attempt.reward = reward
            self._attempt.submission_hash = self._core.inspect()["source_hash"]
            return reward
        except BaseException as exc:
            self._owner.fail(self._attempt, f"Reward/cleanup failed: {type(exc).__name__}")
            raise

    def _call(self, name: str, arguments: dict) -> str:
        if self._attempt is None:
            raise InfrastructureError("Reset before native tools")
        self._owner.check_failures()
        try:
            result = dispatch(self._core, ToolCall(name, arguments, {}))
            observation = bounded_observation(
                result,
                self._presenter,
                self._core.config.training.observation_tokens,
            )
            self._attempt.observations.append({"name": name, "content": observation})
            return observation
        except BaseException as exc:
            # TRL catches ordinary tool exceptions; the trusted latch survives that catch.
            self._owner.fail(self._attempt, f"Tool failed: {type(exc).__name__}")
            raise InfrastructureError("Native tool infrastructure failed") from exc

    def list_files(self, directory: str = ".") -> str:
        """List public repository files.

        Args:
            directory: Relative directory, or '.' for the repository root.
        """
        return self._call("list_files", {"directory": directory})

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 100) -> str:
        """Read bounded public file contents.

        Args:
            path: Permitted relative UTF-8 file path.
            start_line: First line to read, numbered from one.
            max_lines: Maximum positive number of lines to return.
        """
        return self._call(
            "read_file",
            {
                "path": path,
                "start_line": start_line,
                "max_lines": max_lines,
            },
        )

    def apply_patch(self, patch: str) -> str:
        """Atomically edit existing allowed source using an exact unified diff.

        Args:
            patch: Unified diff with a/path and b/path headers and exact context.
        """
        return self._call("apply_patch", {"patch": patch})

    def run_tests(self) -> str:
        """Run the fixed visible tests in a fresh contained snapshot."""
        return self._call("run_tests", {})


def environment_factory(config: Config, catalog: TaskCatalog, owner, presenter) -> Callable:
    """A no-argument factory; ownership stays outside the discoverable adapter API."""

    def create() -> NativeEnvironment:
        core = CodingEnvironment(
            config,
            catalog,
            resource_root=owner.resource_root,
            run_id=owner.run_id,
        )
        owner.cores.append(core)
        return NativeEnvironment(core, owner, presenter)

    return create
