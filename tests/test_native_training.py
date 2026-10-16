"""Opt-in installed TRL contracts; scripted fixtures never certify target-model learning."""

import json
import os
from dataclasses import replace

import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.environ.get("CODING_RL_NATIVE_TESTS") != "1",
        reason="Enable installed TRL checks with CODING_RL_NATIVE_TESTS=1",
    ),
]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    pytest.importorskip("trl")
    from coding_rl.training.trainer import compatibility_probe
    from config import load_config

    config = load_config("configs/train-cpu.yaml")
    directory = tmp_path_factory.mktemp("compatibility")
    try:
        report, tokenizer = compatibility_probe(config, directory)
    except OSError as exc:
        pytest.skip(f"Prepare the pinned Qwen tokenizer first: {exc}")
    assert report["versions"]["trl"] == "1.16.0"
    assert report["runtime_certified"] is False
    tokenizer._evaluation_template_sha256 = report["original_template_sha256"]
    return tokenizer


def make_tiny_model(tokenizer):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    model = Qwen3_5ForCausalLM(config)
    model.config.use_cache = False
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def setup_native(config, tmp_path, tokenizer, *, groups=1, completion_tokens=512, mode="full"):
    from coding_rl.training.trainer import RunOwner, build_native_trainer
    from config import ModelConfig, TrainingConfig
    from tasks.catalog import TaskCatalog

    config = replace(
        config,
        model=ModelConfig(),
        training=TrainingConfig(
            mode=mode,
            max_groups=groups,
            completion_tokens=completion_tokens,
            learning_rate=1e-4,
        ),
    )
    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "private").mkdir()
    owner = RunOwner(config, "a" * 32, directory)
    model = make_tiny_model(tokenizer)
    trainer = build_native_trainer(model, tokenizer, owner, TaskCatalog())
    return owner, model, trainer


def scripted_generation(tokenizer, owner, *, mode="mixed", cutoff=False):
    import torch

    from environments.patches import make_diff

    calls = []

    def generate(input_ids, attention_mask=None, generation_config=None, **kwargs):
        calls.append(owner.policy_version)
        outputs = []
        if len(input_ids) == 4:
            task = owner.cores[0].task
            patch = make_diff(task.starter, task.variant("correct"))
            if cutoff:
                # More than the remaining suffix budget: effects must invalidate the group.
                prefix = "word " * 35
            else:
                prefix = ""
            if mode == "tests":
                first = "<tool_call>\n<function=run_tests>\n</function>\n</tool_call><|im_end|>"
            elif mode == "multi":
                call = "<tool_call>\n<function=list_files>\n</function>\n</tool_call>\n"
                first = call + call + "<|im_end|>"
            else:
                first = (
                    prefix
                    + "<tool_call>\n<function=apply_patch>\n<parameter=patch>\n"
                    + patch
                    + "\n</parameter>\n</function>\n</tool_call><|im_end|>"
                )
            if owner.config.model.profile == "lfm2.5":
                first = (
                    "<|tool_call_start|>[apply_patch(patch="
                    + repr(patch)
                    + ")]<|tool_call_end|><|im_end|>"
                )
            outputs = [first if index < 2 else "Done.<|im_end|>" for index in range(4)]
        else:
            outputs = ["Done.<|im_end|>" for _ in input_ids]
        ids = [tokenizer.encode(text, add_special_tokens=False) for text in outputs]
        if cutoff and len(input_ids) == 4:
            # A complete call fits generation but leaves no room for its result.
            assert max(map(len, ids)) <= owner.config.training.completion_tokens
        width = max(map(len, ids))
        completion = torch.tensor(
            [row + [tokenizer.pad_token_id] * (width - len(row)) for row in ids]
        )
        return torch.cat([input_ids, completion], dim=1)

    return generate, calls


def test_native_parser_and_actual_tensor_masks(tokenizer, config, tmp_path, monkeypatch):
    from coding_rl.training.trainer import parameter_digest

    owner, model, trainer = setup_native(config, tmp_path, tokenizer)
    generation, calls = scripted_generation(tokenizer, owner)
    monkeypatch.setattr(model, "generate", generation)
    before = parameter_digest(model, frozen=False)
    try:
        trainer.train()
        assert calls == [0, 0]
        assert owner.tensor_checked and owner.probability_checked
        assert owner.pending.informative
        assert sorted(a.reward for a in owner.pending.attempts) == [0.0, 0.0, 1.0, 1.0]
        assert parameter_digest(model, frozen=False) != before
        artifact = json.loads((owner.directory / owner.pending.tensor_artifact).read_text())
        assert len(artifact["completion_ids"]) == 4
        assert any(0 in mask for mask in artifact["tool_mask"])
    finally:
        owner.cleanup()


def test_native_callbacks_observe_real_optimizer_and_updated_collection(
    tokenizer, config, tmp_path, monkeypatch
):
    from coding_rl.training import trainer as module

    owner, model, trainer = setup_native(config, tmp_path, tokenizer, groups=2)
    generation, calls = scripted_generation(tokenizer, owner)
    monkeypatch.setattr(model, "generate", generation)
    # Checkpoint serialization/reload is tested separately; this test isolates callbacks.
    monkeypatch.setattr(
        module, "save_policy", lambda *args: {"fixture": "not target-model evidence"}
    )
    checkpoints = []
    trainer.add_callback(module.update_callback(trainer, owner, {}, {}, checkpoints))
    try:
        trainer.train()
        assert calls == [0, 0, 1, 1]
        assert owner.policy_version == trainer.state.global_step == 2
        assert len(checkpoints) == 2
        assert all(g.update["task_gradient_norm"] > 0 for g in owner.groups)
        assert all(g.update["max_observed_parameter_delta"] > 0 for g in owner.groups)
    finally:
        owner.cleanup()


def test_native_swallowed_exception_aborts_before_backward(
    tokenizer, config, tmp_path, monkeypatch
):
    from coding_rl.training.trainer import parameter_digest
    from utils.errors import InfrastructureError

    owner, model, trainer = setup_native(config, tmp_path, tokenizer)
    generation, _ = scripted_generation(tokenizer, owner, mode="tests")
    monkeypatch.setattr(model, "generate", generation)

    def outage(*args, **kwargs):
        raise InfrastructureError("simulated infrastructure fault")

    monkeypatch.setattr("environments.execution.sandbox.Sandbox.visible_tests", outage)
    before = parameter_digest(model, frozen=False)
    try:
        with pytest.raises(InfrastructureError, match="latched"):
            trainer.train()
        assert not owner.tensor_checked and trainer.state.global_step == 0
        assert parameter_digest(model, frozen=False) == before
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        owner.cleanup()


def test_boundary_patch_effect_without_observation_rejects_whole_group(
    tokenizer, config, tmp_path, monkeypatch
):
    from utils.errors import InfrastructureError

    owner, model, trainer = setup_native(config, tmp_path, tokenizer, completion_tokens=160)
    generation, _ = scripted_generation(tokenizer, owner, cutoff=True)
    monkeypatch.setattr(model, "generate", generation)
    try:
        with pytest.raises(InfrastructureError, match="cutoff|Partial"):
            trainer.train()
        assert any(core.inspect()["source_hash"] != core.task.starter_hash for core in owner.cores)
        assert trainer.state.global_step == 0
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        owner.cleanup()


def test_corrupt_actual_mask_cannot_enter_native_loss(tokenizer, config, tmp_path, monkeypatch):
    from utils.errors import InfrastructureError

    owner, model, trainer = setup_native(config, tmp_path, tokenizer)
    generation, _ = scripted_generation(tokenizer, owner)
    monkeypatch.setattr(model, "generate", generation)
    original = owner.audit_tensors

    def corrupt(inputs):
        inputs["tool_mask"].fill_(1)
        original(inputs)

    monkeypatch.setattr(owner, "audit_tensors", corrupt)
    try:
        with pytest.raises(InfrastructureError, match="loss mask"):
            trainer.train()
        assert trainer.state.global_step == 0
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        owner.cleanup()


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_native_update_checkpoint_and_inference_reload(
    tokenizer, config, tmp_path, monkeypatch, mode
):
    import hashlib
    from dataclasses import asdict

    import torch
    from transformers import Qwen3_5ForCausalLM

    from coding_rl.training import trainer as module
    from models.worker import load_policy

    owner, base_model, trainer = setup_native(config, tmp_path, tokenizer, mode=mode)
    actual_generate = base_model.generate
    fixture_generate, _ = scripted_generation(tokenizer, owner)
    monkeypatch.setattr(base_model, "generate", fixture_generate)
    # Reconstruct the identical tiny base for LoRA, without downloading real weights.
    base_from_pretrained = Qwen3_5ForCausalLM.from_pretrained

    def tiny_base(path, **kwargs):
        if str(path) == owner.config.model.name:
            loaded = make_tiny_model(tokenizer)
            if kwargs.get("output_loading_info"):
                return loaded, {"missing_keys": [], "unexpected_keys": []}
            return loaded
        return base_from_pretrained(path, **kwargs)

    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", tiny_base)
    save_policy = module.save_policy

    def save_with_real_generation(*args):
        base_model.generate = actual_generate
        try:
            return save_policy(*args)
        finally:
            base_model.generate = fixture_generate

    monkeypatch.setattr(module, "save_policy", save_with_real_generation)
    digest = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    report = {
        "versions": module.dependency_report(mode),
        "native_template_sha256": digest,
        "original_template_sha256": tokenizer._evaluation_template_sha256,
        "response_template_sha256": module.digest_record(tokenizer.response_template),
        "trainable_parameters": {
            name: p.numel() for name, p in trainer.model.named_parameters() if p.requires_grad
        },
    }
    checkpoints = []
    trainer.add_callback(
        module.update_callback(
            trainer,
            owner,
            report,
            {"baseline_sha256": "a" * 64, "development_sha256": "b" * 64},
            checkpoints,
        )
    )
    try:
        trainer.train()
        assert len(checkpoints) == 1
        assert owner.groups[0].status == "committed"
        saved = replace(owner.config.model, source=mode, checkpoint=checkpoints[0]["path"])
        loaded, _ = load_policy(asdict(saved))
        assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
        trainer.model.eval()
        with torch.no_grad():
            ids = torch.tensor([[1, 2, 3]])
            expected = trainer.model(input_ids=ids, use_cache=False).logits
            actual = loaded(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        if mode == "full":
            from models.backend import TransformersBackend

            backend = TransformersBackend(saved)
            try:
                backend.start()
                assert (
                    backend.identity["policy_source"]["content_sha256"]
                    == checkpoints[0]["content_sha256"]
                )
                assert backend.identity["template_sha256"] != report["native_template_sha256"]
            finally:
                backend.close()
    finally:
        owner.cleanup()


def test_recorded_genuine_read_parses_with_selected_native_template(tokenizer):
    from pathlib import Path

    from trl.chat_template_utils import parse_response

    from coding_rl.training.adapter import NativeEnvironment
    from coding_rl.training.trainer import tool_methods

    evidence = json.loads(Path("docs/evidence/phase2-cpu-read.json").read_text())
    methods = tool_methods(NativeEnvironment.__new__(NativeEnvironment))
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Read the source."}],
        tools=methods,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    parsed = parse_response(
        tokenizer, evidence["generation"]["token_ids"], prefix=prefix, tools=methods
    )
    assert parsed["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": {"path": "timeout.py"},
    }


def test_native_multicall_profile_executes_sequentially_and_reports_equal_rewards(
    tokenizer, config, tmp_path, monkeypatch
):
    from coding_rl.training.trainer import parameter_digest

    owner, model, trainer = setup_native(config, tmp_path, tokenizer)
    generation, _ = scripted_generation(tokenizer, owner, mode="multi")
    monkeypatch.setattr(model, "generate", generation)
    before = parameter_digest(model, frozen=False)
    try:
        trainer.train()
        assert [core.calls for core in owner.cores] == [2, 2, 0, 0]
        assert not owner.pending.informative
        assert all(attempt.reward == 0.0 for attempt in owner.pending.attempts)
        assert parameter_digest(model, frozen=False) == before
    finally:
        owner.cleanup()


def test_lfm_native_parser_masks_and_optimizer(config, tmp_path, monkeypatch):
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM

    from coding_rl.training.trainer import (
        RunOwner,
        build_native_trainer,
        compatibility_probe,
        parameter_digest,
    )
    from config import load_config
    from tasks.catalog import TaskCatalog

    profile = load_config("configs/train-lfm2.5-350m.yaml")
    try:
        _, tokens = compatibility_probe(profile, tmp_path)
    except OSError as exc:
        pytest.skip(f"Prepare pinned LFM tokenizer with model-fetch: {exc}")
    config = replace(
        config, model=profile.model, training=replace(profile.training, mode="full", max_groups=1)
    )
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = Lfm2ForCausalLM(
        Lfm2Config(
            vocab_size=len(tokens),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            layer_types=["conv", "full_attention"],
            block_auto_adjust_ff_dim=False,
            block_multiple_of=1,
        )
    )
    model.generation_config.eos_token_id = tokens.eos_token_id
    model.generation_config.pad_token_id = tokens.pad_token_id
    owner = RunOwner(config, "a" * 32, tmp_path)
    trainer = build_native_trainer(model, tokens, owner, TaskCatalog())
    generation, _ = scripted_generation(tokens, owner)
    monkeypatch.setattr(model, "generate", generation)
    before = parameter_digest(model, frozen=False)
    try:
        trainer.train()
        assert owner.tensor_checked and owner.probability_checked and owner.pending.informative
        assert parameter_digest(model, frozen=False) != before
    finally:
        owner.cleanup()


@pytest.mark.parametrize("name", ["missing_tool", "read_file"])
def test_native_predispatch_errors_are_healthy_observations(
    tokenizer, config, tmp_path, monkeypatch, name
):
    import torch

    owner, model, trainer = setup_native(config, tmp_path, tokenizer)

    def generate(input_ids, **kwargs):
        text = (
            f"<tool_call>\n<function={name}>\n</function>\n</tool_call><|im_end|>"
            if len(input_ids) == 4
            else "Done.<|im_end|>"
        )
        # All four positions call once. The second generation still has four rows.
        if getattr(generate, "called", False):
            text = "Done.<|im_end|>"
        generate.called = True
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.cat([input_ids, torch.tensor([ids] * len(input_ids))], dim=1)

    monkeypatch.setattr(model, "generate", generate)
    try:
        trainer.train()
        assert owner.tensor_checked and owner.probability_checked
        assert all(core.calls == 0 for core in owner.cores)
        assert all(a.reward == 0 for a in owner.pending.attempts)
    finally:
        owner.cleanup()


@pytest.mark.parametrize("stage", ["generation", "visible", "grading"])
def test_interrupt_actual_native_boundaries_before_optimizer(
    tokenizer, config, tmp_path, monkeypatch, stage
):
    from environments.execution.sandbox import Sandbox
    from utils.errors import InfrastructureError

    owner, model, trainer = setup_native(config, tmp_path, tokenizer)
    generation, _ = scripted_generation(
        tokenizer, owner, mode="tests" if stage == "visible" else "mixed"
    )

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(model, "generate", interrupt if stage == "generation" else generation)
    if stage in {"visible", "grading"}:
        monkeypatch.setattr(Sandbox, "visible_tests" if stage == "visible" else "invoke", interrupt)
    try:
        # The tool adapter latches interruptions before TRL catches tool errors.
        error = InfrastructureError if stage == "visible" else KeyboardInterrupt
        with pytest.raises(error):
            trainer.train()
        assert trainer.state.global_step == 0
        assert all(p.grad is None for p in model.parameters())
    finally:
        owner.cleanup()
    assert all(core.workspace is None for core in owner.cores)
    assert not list(owner.resource_root.glob("coding-rl-*"))


def test_interrupted_real_second_save_preserves_first_committed_checkpoint(
    tokenizer, config, tmp_path, monkeypatch
):
    from coding_rl.training import trainer as module
    from coding_rl.training.checkpoints import load_checkpoint

    owner, model, trainer = setup_native(config, tmp_path, tokenizer, groups=2)
    actual_generate = model.generate
    actual_save = model.save_pretrained
    generation, _ = scripted_generation(tokenizer, owner)
    monkeypatch.setattr(model, "generate", generation)
    original_policy_save = module.save_policy
    saves = 0

    def interrupted_save(*args, **kwargs):
        nonlocal saves
        saves += 1
        actual_save(*args, **kwargs)
        if saves == 2:
            raise KeyboardInterrupt()

    def real_generation_during_save(*args):
        model.generate = actual_generate
        try:
            return original_policy_save(*args)
        finally:
            model.generate = generation

    monkeypatch.setattr(model, "save_pretrained", interrupted_save)
    monkeypatch.setattr(module, "save_policy", real_generation_during_save)
    report = {
        "versions": module.dependency_report("full"),
        "native_template_sha256": module.digest_record(tokenizer.chat_template),
        "original_template_sha256": tokenizer._evaluation_template_sha256,
        "response_template_sha256": module.digest_record(tokenizer.response_template),
        "trainable_parameters": {
            n: p.numel() for n, p in model.named_parameters() if p.requires_grad
        },
    }
    checkpoints = []
    trainer.add_callback(
        module.update_callback(
            trainer,
            owner,
            report,
            {"baseline_sha256": "a" * 64, "development_sha256": "b" * 64},
            checkpoints,
        )
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            trainer.train()
        assert len(checkpoints) == 1 and saves == 2
        first = load_checkpoint(checkpoints[0]["path"])
        assert first.manifest["global_step"] == 1
        assert first.manifest["provenance"]["audit_files"]
        assert not list((owner.directory / "checkpoints").glob(".checkpoint-*"))
        assert not (owner.directory / "checkpoints" / "step-2").exists()
    finally:
        owner.cleanup()
