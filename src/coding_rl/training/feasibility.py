"""Disposable capacity measurements, separate from task rewards and learning proof."""

import platform
import shutil
import time
from dataclasses import asdict

from tasks.catalog import TaskCatalog
from utils.errors import InfrastructureError


def preflight_capacity(config) -> dict:
    import psutil
    from transformers import AutoConfig

    spec = AutoConfig.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        cache_dir=config.model.cache_dir,
        local_files_only=config.model.offline,
        trust_remote_code=False,
        token=False,
    ).get_text_config()
    # Architecture-derived conservative approximation, followed by exact measured counts.
    hidden, layers, vocab = spec.hidden_size, spec.num_hidden_layers, spec.vocab_size
    intermediate = getattr(spec, "intermediate_size", hidden * 4)
    estimated_parameters = vocab * hidden * (1 if spec.tie_word_embeddings else 2) + layers * (
        4 * hidden**2 + 3 * hidden * intermediate
    )
    minimum = estimated_parameters * (16 if config.training.mode == "full" else 4)
    available = min(psutil.virtual_memory().available, config.training.memory_limit_mb * 1048576)
    if minimum > available:
        raise InfrastructureError(
            "Conservative pre-allocation FP32/Adam estimate exceeds host/config RAM; use LoRA or a smaller model"
        )
    return {
        "estimated_parameters_upper_bound": estimated_parameters,
        "estimated_weight_optimizer_bytes": minimum,
        "available_budget_bytes": available,
        "host_total_ram_bytes": psutil.virtual_memory().total,
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "threads": config.model.threads,
    }


def measure_capacity(config, tokenizer, owner) -> dict:
    import torch

    from coding_rl.training.adapter import NativeEnvironment, training_rows
    from coding_rl.training.trainer import build_native_trainer, feasibility_probe, tool_methods
    from models.worker import load_policy
    from utils.process import peak_rss_bytes

    host = preflight_capacity(config)
    torch.set_num_threads(config.model.threads)
    torch.manual_seed(config.seed)
    started = time.monotonic()
    model, loading = load_policy(asdict(config.model), trainable=True)
    load_seconds = time.monotonic() - started
    trainer = build_native_trainer(model, tokenizer, owner, TaskCatalog(config.task_root))
    model = trainer.model
    smoke = feasibility_probe(model, tokenizer, config)
    methods = tool_methods(NativeEnvironment.__new__(NativeEnvironment))
    prompts = [
        tokenizer.apply_chat_template(
            row["prompt"],
            tools=methods,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in training_rows(TaskCatalog(config.task_root))
    ]
    longest = max(prompts, key=len)
    # This bounds every supported native collection: initial prompt + completion region.
    row = (
        longest
        + tokenizer.encode("Repair the source.\n", add_special_tokens=False)
        * config.training.completion_tokens
    )
    row = row[: len(longest) + config.training.completion_tokens]
    ids = torch.tensor([row] * config.training.group_size)
    labels = ids.clone()
    labels[:, : len(longest)] = -100
    before = time.monotonic()
    model.train()
    result = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=labels,
        fused_lm_head=True,
        use_cache=False,
    )
    loss = result.loss
    del result
    if not bool(torch.isfinite(loss)):
        raise InfrastructureError("Non-finite representative capacity loss")
    forward_seconds = time.monotonic() - before
    before = time.monotonic()
    loss.backward()
    backward_seconds = time.monotonic() - before
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if (
        not gradients
        or not all(bool(torch.isfinite(g).all()) for g in gradients)
        or not any(bool(g.any()) for g in gradients)
    ):
        raise InfrastructureError(
            "Representative gradients must all be finite and include a nonzero gradient"
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.training.learning_rate
    )
    before = time.monotonic()
    optimizer.step()
    optimizer_seconds = time.monotonic() - before
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise InfrastructureError("Non-finite disposable optimizer result")
    model.zero_grad(set_to_none=True)
    staging = owner.directory / "private" / "capacity-checkpoint"
    before = time.monotonic()
    try:
        model.save_pretrained(staging, safe_serialization=True)
        from models.profiles import model_class

        if config.training.mode == "full":
            loaded = model_class(config.model).from_pretrained(
                staging, local_files_only=True, dtype=torch.float32
            )
        else:
            from peft import PeftModel

            base, _ = load_policy(asdict(config.model))
            loaded = PeftModel.from_pretrained(base, staging, local_files_only=True)
        loaded.eval()
        model.eval()
        probe = ids[:1, :3]
        with torch.no_grad():
            torch.testing.assert_close(
                loaded(input_ids=probe, use_cache=False).logits,
                model(input_ids=probe, use_cache=False).logits,
                atol=1e-5,
                rtol=1e-5,
            )
        del loaded
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "kind": "disposable_representative_capacity",
        "passed": True,
        "host": host,
        "model_class": type(model).__name__,
        "loading": loading,
        "load_seconds": load_seconds,
        "smoke": smoke,
        "batch_size": len(ids),
        "sequence_tokens": ids.shape[1],
        "completion_tokens": config.training.completion_tokens,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "save_reload_seconds": time.monotonic() - before,
        "peak_rss_bytes": peak_rss_bytes(),
        "trainable_parameters": {
            n: p.numel() for n, p in model.named_parameters() if p.requires_grad
        },
        "policy_disposition": "Child exits; diagnostic update and optimizer are discarded; training reloads pinned base",
        "limitation": "Capacity measurement; no task correctness or GRPO improvement claim",
    }
