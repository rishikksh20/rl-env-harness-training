"""Atomic, content-identified full/LoRA artifacts; incomplete saves are never policies."""

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from utils.errors import InfrastructureError
from utils.paths import relative_path
from utils.records import digest_record

MANIFEST = "checkpoint.json"


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
    if value.get("schema_version") != 2 or value.get("kind") not in {"full", "lora"}:
        raise ValueError(
            "Unsupported checkpoint manifest; schema 1 lacks validated update proof; regenerate with schema 2"
        )
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
    for key in ("revision", "tokenizer_revision"):
        if not re.fullmatch(r"[a-f0-9]{40}", base[key]):
            raise ValueError("Checkpoint base revisions must be immutable commits")
    provenance, update, reload = (value[k] for k in ("provenance", "update_evidence", "reload"))
    for key in (
        "config_sha256",
        "source_sha256",
        "native_template_sha256",
        "response_template_sha256",
        "evaluation_template_sha256",
    ):
        if not isinstance(provenance.get(key), str) or not re.fullmatch(
            r"[a-f0-9]{64}", provenance[key]
        ):
            raise ValueError(f"Checkpoint requires a valid provenance {key}")
    trainable = provenance.get("trainable_parameters", {})
    if (
        not isinstance(trainable, dict)
        or not trainable
        or not all(isinstance(n, str) and type(p) is int and p > 0 for n, p in trainable.items())
    ):
        raise ValueError("Checkpoint requires the intended trainable parameter names/counts")
    if (
        not provenance.get("dependencies")
        or not provenance.get("tasks")
        or not provenance.get("worker_identity")
    ):
        raise ValueError("Checkpoint requires dependency/task/worker provenance")
    prerequisites = provenance.get("prerequisites", {})
    if any(
        not re.fullmatch(r"[a-f0-9]{64}", str(prerequisites.get(key, "")))
        for key in ("baseline_sha256", "development_sha256")
    ):
        raise ValueError("Checkpoint requires baseline/development evidence references")
    if (
        update.get("global_step") != value["global_step"]
        or update.get("informative") is not True
        or update.get("frozen_unchanged") is not True
        or type(update.get("policy_version")) is not int
        or update["policy_version"] < 0
    ):
        raise ValueError("Checkpoint update step/policy/informative proof contradicts the manifest")
    for key in ("task_gradient_norm", "max_observed_parameter_delta", "delta_tolerance"):
        if (
            type(update.get(key)) not in (int, float)
            or not math.isfinite(update[key])
            or update[key] <= 0
        ):
            raise ValueError(f"Checkpoint requires finite positive update {key}")
    if update["max_observed_parameter_delta"] <= update["delta_tolerance"]:
        raise ValueError("Checkpoint requires a measured parameter change above tolerance")
    for key in ("before_sha256", "after_sha256", "frozen_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(update.get(key, ""))):
            raise ValueError("Checkpoint requires parameter fingerprints")
    collection = update.get("collection_id", "")
    if (
        not re.fullmatch(r"[a-f0-9]{32}", collection)
        or update.get("tensor_artifact") != f"{collection}-tensors.json"
    ):
        raise ValueError("Checkpoint collection/tensor references disagree")
    audits = provenance.get("audit_files", {})
    required_audits = {
        f"audit/{collection}-{suffix}.json" for suffix in ("native", "tensors", "probabilities")
    }
    if (
        not isinstance(audits, dict)
        or set(audits) != required_audits
        or any(not re.fullmatch(r"[a-f0-9]{64}", str(v)) for v in audits.values())
    ):
        raise ValueError("Checkpoint requires portable native/tensor/probability audit references")
    parent = value.get("parent", {})
    generated = update.get("updated_policy_generation", {})
    if (
        update["before_sha256"] == update["after_sha256"]
        or parent.get("trainable_sha256") != update["before_sha256"]
        or parent.get("policy_version") != update["policy_version"]
        or generated.get("trainable_sha256") != update["after_sha256"]
    ):
        raise ValueError("Checkpoint parameter/parent/generation fingerprints disagree")
    for key in ("logits_atol", "logits_rtol"):
        if (
            type(reload.get(key)) not in (float, int)
            or not math.isfinite(reload[key])
            or not 0 < reload[key] <= 1e-5
        ):
            raise ValueError("Checkpoint requires bounded finite reload tolerances")
    for key in ("input_ids", "output_ids"):
        ids = reload.get(key)
        if (
            not isinstance(ids, list)
            or not ids
            or not all(
                isinstance(row, list) and row and all(type(t) is int and t >= 0 for t in row)
                for row in ids
            )
            or ids != generated.get(key)
        ):
            raise ValueError("Checkpoint deterministic generation/reload evidence disagrees")


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
        if any(
            files.get(name) != digest for name, digest in value["provenance"]["audit_files"].items()
        ):
            raise ValueError("Checkpoint audit references differ from saved files")
        if not any(name.endswith(".safetensors") for name in files):
            raise ValueError("Checkpoint requires safetensors weights")
        if value["kind"] == "lora" and "adapter_config.json" not in files:
            raise ValueError("LoRA checkpoint requires adapter_config.json")
        if value["kind"] == "full" and "config.json" not in files:
            raise ValueError("Full checkpoint requires config.json")
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
        value = {**metadata, "schema_version": 2, "reload": reload_result}
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
