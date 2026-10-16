import hashlib
import json
from dataclasses import replace

import pytest

from coding_rl.rollout.evaluate import compare_evaluations, evaluation_profile
from coding_rl.training.checkpoints import load_checkpoint, policy_identity, publish_checkpoint
from config import ModelConfig
from tasks.catalog import TaskCatalog


def metadata(kind):
    config = ModelConfig()
    digest = "a" * 64
    collection = "c" * 32
    return {
        "kind": kind,
        "global_step": 1,
        "base": {key: getattr(config, key) for key in ("name", "revision", "tokenizer_revision")},
        "parent": {"policy_version": 0, "trainable_sha256": digest},
        "provenance": {
            "fixture": "unit-test-only; not real training evidence",
            **dict.fromkeys(
                (
                    "config_sha256",
                    "source_sha256",
                    "native_template_sha256",
                    "response_template_sha256",
                    "evaluation_template_sha256",
                ),
                digest,
            ),
            "trainable_parameters": {"fixture.weight": 1},
            "dependencies": {"fixture": "1"},
            "tasks": [{"task_id": "fixture"}],
            "worker_identity": "fixture",
            "prerequisites": {"baseline_sha256": digest, "development_sha256": digest},
            "audit_files": {
                f"audit/{collection}-{name}.json": hashlib.sha256(b"{}").hexdigest()
                for name in ("native", "tensors", "probabilities")
            },
        },
        "update_evidence": {
            "collection_id": collection,
            "policy_version": 0,
            "global_step": 1,
            "informative": True,
            "frozen_unchanged": True,
            "task_gradient_norm": 1.0,
            "max_observed_parameter_delta": 0.001,
            "delta_tolerance": 1e-12,
            "before_sha256": digest,
            "after_sha256": "b" * 64,
            "frozen_sha256": digest,
            "tensor_artifact": collection + "-tensors.json",
            "updated_policy_generation": {
                "trainable_sha256": "b" * 64,
                "input_ids": [[1]],
                "output_ids": [[1, 2]],
            },
        },
    }


def reload_evidence(path):
    return {
        "passed": True,
        "logits_atol": 1e-5,
        "logits_rtol": 1e-5,
        "input_ids": [[1]],
        "output_ids": [[1, 2]],
    }


def save_weights(path, kind="full"):
    (path / "model.safetensors").write_bytes(b"test fixture, not model weights")
    (path / ("adapter_config.json" if kind == "lora" else "config.json")).write_text("{}")
    (path / "audit").mkdir()
    for name in ("native", "tensors", "probabilities"):
        (path / "audit" / ("c" * 32 + f"-{name}.json")).write_text("{}")


@pytest.mark.parametrize("kind", ["full", "lora"])
def test_content_identity_roundtrip_and_policy_base_validation(tmp_path, kind):
    target = tmp_path / kind
    artifact = publish_checkpoint(
        target,
        metadata=metadata(kind),
        save=lambda path: save_weights(path, kind),
        validate_reload=reload_evidence,
    )
    config = ModelConfig(source=kind, checkpoint=str(target))
    assert load_checkpoint(target).identity == artifact.identity == policy_identity(config)
    assert len(artifact.identity["content_sha256"]) == 64
    with pytest.raises(ValueError, match="base/tokenizer"):
        policy_identity(replace(config, revision="b" * 40))
    with pytest.raises(ValueError, match="kind"):
        policy_identity(replace(config, source="full" if kind == "lora" else "lora"))
    (target / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content manifest"):
        load_checkpoint(target)


@pytest.mark.parametrize("failure", ["save", "reload", "reload_false"])
def test_interrupted_or_unverified_save_is_not_a_policy(tmp_path, failure):
    def save(path):
        save_weights(path)
        if failure == "save":
            raise KeyboardInterrupt()

    def reload(path):
        if failure == "reload":
            raise RuntimeError("reload mismatch")
        return {"passed": False}

    with pytest.raises((KeyboardInterrupt, RuntimeError, ValueError)):
        publish_checkpoint(
            tmp_path / "policy", metadata=metadata("full"), save=save, validate_reload=reload
        )
    assert not (tmp_path / "policy").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_extra_files_links_and_changed_manifest_are_rejected(tmp_path):
    artifact = publish_checkpoint(
        tmp_path / "policy",
        metadata=metadata("full"),
        save=save_weights,
        validate_reload=reload_evidence,
    )
    link = artifact.path / "extra"
    link.symlink_to(artifact.path / "model.safetensors")
    with pytest.raises(ValueError, match="links"):
        load_checkpoint(artifact.path)
    link.unlink()
    (artifact.path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="content manifest"):
        load_checkpoint(artifact.path)
    (artifact.path / "extra.json").unlink()
    path = artifact.path / "checkpoint.json"
    manifest = json.loads(path.read_text())
    manifest["global_step"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="step|identity mismatch"):
        load_checkpoint(artifact.path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_gradient_norm", float("nan")),
        ("global_step", 2),
        ("max_observed_parameter_delta", 1e-20),
        ("tensor_artifact", "foreign.json"),
        ("informative", False),
    ],
)
def test_contradictory_update_proof_cannot_be_published(tmp_path, field, value):
    proof = metadata("full")
    proof["update_evidence"][field] = value
    with pytest.raises(ValueError):
        publish_checkpoint(
            tmp_path / "invalid", metadata=proof, save=save_weights, validate_reload=reload_evidence
        )
    assert not (tmp_path / "invalid").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_evaluation_comparison_ignores_policy_and_output_paths(config):
    config = replace(config, model=ModelConfig())
    catalog = TaskCatalog()
    identity = {"template_sha256": "template", "revision": "old"}
    before = evaluation_profile(config, catalog, "image-sha256", identity)
    changed = replace(
        config,
        model=replace(config.model, source="full", checkpoint="local"),
        evaluation=replace(config.evaluation, report_path="different"),
    )
    assert before == evaluation_profile(
        changed, catalog, "image-sha256", {**identity, "revision": "new"}
    )
    after = evaluation_profile(replace(changed, seed=43), catalog, "image-sha256", identity)
    baseline = {
        "split": "eval",
        "worker_identity": "image-sha256",
        "evaluation_profile": before,
        "metrics": {"valid": 0, "verified_pass_rate": None},
    }
    trained = {**baseline, "metrics": {"valid": 4, "verified_pass_rate": 0.5}}
    with pytest.raises(ValueError, match="actual attempt records|complete held-out"):
        compare_evaluations(baseline, trained)
    with pytest.raises(ValueError, match="profiles differ"):
        compare_evaluations(baseline, {**trained, "evaluation_profile": after})
