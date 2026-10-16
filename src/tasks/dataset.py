"""Import local Python repair tasks into the protected functional-JSON task format."""

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from tasks.catalog import TaskCatalog
from utils.paths import relative_path


def import_tasks(source: Path, destination: Path) -> dict:
    """Stage and validate the entire catalog before publishing it atomically."""
    if destination.exists():
        raise ValueError("Dataset destination already exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=destination.parent))
    try:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            required = {
                "task_id",
                "split",
                "prompt",
                "module",
                "function",
                "editable",
                "files",
                "private_cases",
            }
            if set(item) != required or not re.fullmatch(
                r"[a-z0-9][a-z0-9_-]{0,79}", item["task_id"]
            ):
                raise ValueError(
                    "Task rows require exactly the documented fields and a safe task_id"
                )
            folder = staging / item["task_id"]
            folder.mkdir()
            hashes = {}
            for name, content in item["files"].items():
                relative_path(name)
                if not isinstance(content, str) or len(content.encode()) > 65536:
                    raise ValueError("Starter files must be bounded UTF-8 text")
                path = folder / "starter" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                hashes[name] = hashlib.sha256(content.encode()).hexdigest()
            manifest = {k: v for k, v in item.items() if k not in {"files", "private_cases"}}
            (folder / "manifest.json").write_text(
                json.dumps({**manifest, "files": hashes}, indent=2) + "\n"
            )
            (folder / "private.json").write_text(json.dumps(item["private_cases"], indent=2) + "\n")
        catalog = TaskCatalog(staging)
        if not catalog.tasks("train") or not catalog.tasks("eval"):
            raise ValueError("Imported datasets require disjoint train and eval tasks")
        staging.rename(destination)
        return {
            "directory": str(destination.resolve()),
            "train": len(catalog.tasks("train")),
            "eval": len(catalog.tasks("eval")),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)
