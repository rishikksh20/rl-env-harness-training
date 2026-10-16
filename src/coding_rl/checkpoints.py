"""Atomic, content-identified full/LoRA artifacts; incomplete saves are never policies."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from coding_rl.errors import InfrastructureError
from coding_rl.tasks import relative_path

MANIFEST = "checkpoint.json"


def digest_record(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint links are forbidden")
        if path.is_file() and path != root / MANIFEST:
            result[path.relative_to(root).as_posix()] = _digest_file(path)
        elif not path.is_file() and not path.is_dir():
            raise ValueError("Checkpoint files must be regular")
    return result


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    manifest: dict

    @property
    def identity(self) -> dict:
        return {
            "source": self.manifest["kind"],
            "content_sha256": self.manifest["content_sha256"],
            "base": self.manifest["base"],
            "global_step": self.manifest["global_step"],
        }


def _validate_metadata(value: dict) -> None:
    if value.get("schema_version") != 1 or value.get("kind") not in {"full", "lora"}:
        raise ValueError("Unsupported checkpoint manifest")
    base = value.get("base", {})
    if not isinstance(base, dict) or not all(
        isinstance(base.get(key), str) and base[key]
        for key in ("name", "revision", "tokenizer_revision")
    ):
        raise ValueError("Checkpoint requires the pinned base/tokenizer identity")
    if type(value.get("global_step")) is not int or value["global_step"] < 1:
        raise ValueError("Checkpoint requires a completed optimizer step")
    for key in ("provenance", "update_evidence", "reload"):
        if not isinstance(value.get(key), dict) or not value[key]:
            raise ValueError(f"Checkpoint requires {key} evidence")
    if value["reload"].get("passed") is not True:
        raise ValueError("Checkpoint reload validation did not pass")


def load_checkpoint(path: str | Path) -> Checkpoint:
    root = Path(path)
    if root.is_symlink() or not root.is_dir() or (root / MANIFEST).is_symlink():
        raise ValueError("Checkpoint must be a completed local directory without links")
    try:
        value = json.loads((root / MANIFEST).read_text())
        if type(value) is not dict:
            raise ValueError("Invalid checkpoint manifest")
        _validate_metadata(value)
        files = value.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Checkpoint file manifest is empty")
        for name in files:
            relative_path(name)
        if files != _files(root):
            raise ValueError("Checkpoint files differ from their content manifest")
        if not any(name.endswith(".safetensors") for name in files):
            raise ValueError("Checkpoint requires safetensors weights")
        if value["kind"] == "lora" and "adapter_config.json" not in files:
            raise ValueError("LoRA checkpoint requires adapter_config.json")
        content = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != digest_record(content):
            raise ValueError("Checkpoint content identity mismatch")
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read completed checkpoint manifest") from exc
    return Checkpoint(root.resolve(), value)


def publish_checkpoint(
    destination: str | Path,
    *,
    metadata: dict,
    save: Callable[[Path], None],
    validate_reload: Callable[[Path], dict],
) -> Checkpoint:
    """Stage beside the destination and publish only after a successful reload check."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ValueError("Checkpoint destination already exists")
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=target.parent))
    try:
        save(staging)
        reload_result = validate_reload(staging)
        value = {**metadata, "schema_version": 1, "reload": reload_result}
        _validate_metadata(value)
        value["files"] = _files(staging)
        value["content_sha256"] = digest_record(value)
        (staging / MANIFEST).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        load_checkpoint(staging)
        staging.rename(target)
        return load_checkpoint(target)
    except OSError as exc:
        raise InfrastructureError("Checkpoint could not be committed") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def policy_identity(config) -> dict:
    if config.source == "hub":
        return {"source": "hub", "checkpoint": config.name, "revision": config.revision}
    artifact = load_checkpoint(config.checkpoint)
    if artifact.manifest["kind"] != config.source:
        raise ValueError("Configured policy kind differs from checkpoint")
    expected = {key: getattr(config, key) for key in ("name", "revision", "tokenizer_revision")}
    if artifact.manifest["base"] != expected:
        raise ValueError("Checkpoint base/tokenizer differs from the configured pinned model")
    return artifact.identity
