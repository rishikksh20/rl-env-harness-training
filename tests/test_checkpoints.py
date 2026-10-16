import json
from dataclasses import replace

import pytest

from coding_rl.checkpoints import load_checkpoint, policy_identity, publish_checkpoint
from coding_rl.config import ModelConfig
from coding_rl.evaluate import compare_evaluations, evaluation_profile
from coding_rl.tasks import TaskCatalog


def metadata(kind):
    config = ModelConfig()
    return {
        "kind": kind,
        "global_step": 1,
        "base": {key: getattr(config, key) for key in ("name", "revision", "tokenizer_revision")},
        "provenance": {"fixture": "unit-test-only"},
        "update_evidence": {"fixture": "not real training evidence"},
    }


def save_weights(path, kind="full"):
    (path / "model.safetensors").write_bytes(b"test fixture, not model weights")
    (path / ("adapter_config.json" if kind == "lora" else "config.json")).write_text("{}")


@pytest.mark.parametrize("kind", ["full", "lora"])
def test_content_identity_roundtrip_and_policy_base_validation(tmp_path, kind):
    target = tmp_path / kind
    artifact = publish_checkpoint(
        target,
        metadata=metadata(kind),
        save=lambda path: save_weights(path, kind),
        validate_reload=lambda path: {"passed": True, "fixture": True},
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
        validate_reload=lambda path: {"passed": True},
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
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(artifact.path)


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
    baseline = {"evaluation_profile": before, "metrics": {"valid": 0, "verified_pass_rate": None}}
    trained = {"evaluation_profile": before, "metrics": {"valid": 4, "verified_pass_rate": 0.5}}
    result = compare_evaluations(baseline, trained)
    assert result["pass_rate_change"] is None and not result["equal_valid_coverage"]
    with pytest.raises(ValueError, match="profiles differ"):
        compare_evaluations(baseline, {**trained, "evaluation_profile": after})
