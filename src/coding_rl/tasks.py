"""Load authored fixtures as data. Never import their Python on the controller."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType


def file_digest(files: Mapping[str, str]) -> str:
    """Canonical digest includes filenames and exact UTF-8 contents."""
    encoded = json.dumps(dict(files), sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def relative_path(path: str, *, root: bool = False) -> str:
    if not isinstance(path, str) or "\x00" in path or "\\" in path:
        raise ValueError("Use a repository-relative POSIX path")
    if root and path in {"", "."}:
        return "."
    parts = path.split("/")
    if not path or PurePosixPath(path).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("Use a canonical relative path without parent segments")
    return path


@dataclass(frozen=True)
class Case:
    kwargs: dict
    expected: object


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    prompt: str
    editable: tuple[str, ...]
    starter_hash: str


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    split: str
    module: str
    function: str
    editable: tuple[str, ...]
    starter: Mapping[str, str]
    starter_hash: str
    private_cases: tuple[Case, ...]
    references: Mapping[str, Mapping[str, str]]

    def public(self) -> PublicTask:
        return PublicTask(self.task_id, self.prompt, self.editable, self.starter_hash)

    def variant(self, name: str) -> dict[str, str]:
        if name == "baseline":
            return dict(self.starter)
        return {**self.starter, **self.references[name]}

    def is_authored_snapshot(self, snapshot: Mapping[str, str]) -> bool:
        return any(dict(snapshot) == self.variant(name) for name in ("baseline", *self.references))


class TaskCatalog:
    """Immutable public starters and protected grading data bundled in the wheel."""

    def __init__(self, root: Path | None = None):
        root = root or Path(__file__).parent / "fixtures"
        self._tasks: dict[str, Task] = {}
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            manifest = json.loads((folder / "manifest.json").read_text())
            starter = {}
            for name, expected_hash in manifest["files"].items():
                relative_path(name)
                path = folder / "starter" / name
                # Installed wheels can legitimately hardlink authored package
                # assets to uv's cache. Their hashes establish integrity; the
                # runtime workspace still rejects hardlinked editable files.
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Fixture files must be regular files")
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected_hash:
                    raise ValueError(f"Fixture hash mismatch: {manifest['task_id']} / {name}")
                starter[name] = raw.decode("utf-8")
            editable = tuple(manifest["editable"])
            if not editable or any(name not in starter for name in editable):
                raise ValueError("Editable files must belong to the starter")
            references = json.loads((folder / "references.json").read_text())
            if any(set(edits) - set(editable) for edits in references.values()):
                raise ValueError("Reference fixtures may edit allowed source only")
            cases = tuple(
                Case(**case) for case in json.loads((folder / "private.json").read_text())
            )
            task = Task(
                task_id=manifest["task_id"],
                prompt=manifest["prompt"],
                split=manifest["split"],
                module=manifest["module"],
                function=manifest["function"],
                editable=editable,
                starter=MappingProxyType(starter),
                starter_hash=file_digest(starter),
                private_cases=cases,
                references=MappingProxyType(
                    {k: MappingProxyType(v) for k, v in references.items()}
                ),
            )
            if task.task_id in self._tasks or task.split not in {"train", "eval"}:
                raise ValueError("Task IDs must be unique and splits must be train/eval")
            if task.module not in editable or not task.function.isidentifier():
                raise ValueError("Invalid authored callable contract")
            self._tasks[task.task_id] = task
        if not self._tasks:
            raise ValueError("Task catalog is empty")

    def get(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"Unknown task ID: {task_id}") from exc

    def tasks(self, split: str | None = None) -> tuple[Task, ...]:
        return tuple(task for task in self._tasks.values() if split is None or task.split == split)

    def dataset_rows(self, split: str = "train") -> list[dict]:
        """Export only public prompt and opaque task ID, without grader metadata."""
        return [
            {"task_id": t.task_id, "prompt": [{"role": "user", "content": t.prompt}]}
            for t in self.tasks(split)
        ]
