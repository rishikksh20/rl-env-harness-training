"""Trusted spawned process: model weights and generation, never candidate execution."""

import time
from multiprocessing.connection import Connection

from utils.ipc import receive_record, send_record
from utils.process import peak_rss_bytes


class TokenTiming:
    """Transformers' streamer hook observes emission after prefill and each decode."""

    def __init__(self, started: float):
        self.started = started
        self.prompt = True
        self.first = self.last = None

    def put(self, value) -> None:
        if self.prompt:
            self.prompt = False
            return
        self.last = time.monotonic()
        if self.first is None:
            self.first = self.last

    def end(self) -> None:
        pass


def load_policy(settings: dict, *, trainable: bool = False):
    """Load weights in-process; local adapters always reconstruct the pinned base."""
    import torch

    from coding_rl.training.checkpoints import load_checkpoint, policy_identity
    from config import ModelConfig
    from models.profiles import model_class

    config = ModelConfig(
        **{
            k: v
            for k, v in settings.items()
            if k
            not in {
                "eos_token_id",
                "pad_token_id",
            }
        }
    )
    policy_identity(config)  # Validate all local bytes again inside the loading process.
    source = config.name
    options = {
        "revision": config.revision,
        "cache_dir": config.cache_dir,
        "local_files_only": config.offline,
    }
    if config.source == "full":
        source = str(load_checkpoint(config.checkpoint).path)
        options = {"local_files_only": True}
    model, loading = model_class(config).from_pretrained(
        source,
        **options,
        trust_remote_code=False,
        dtype=torch.float32,
        output_loading_info=True,
        token=False,
    )
    loading = {
        key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()
    }
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Incomplete language weight loading: {loading}")
    if any(
        not any(part in key for part in ("visual.", "vision.", "mtp."))
        for key in loading.get("unexpected_keys", [])
    ):
        raise ValueError(f"Unexplained checkpoint weights: {loading}")
    if config.source == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(load_checkpoint(config.checkpoint).path),
            is_trainable=trainable,
            local_files_only=True,
        )
    model.to(config.device)
    model.requires_grad_(trainable)
    model.train(trainable)
    return model, loading


def worker_main(connection: Connection, settings: dict) -> None:
    """Spawn entry point; heavy libraries stay behind this model-only boundary."""
    try:
        import torch

        torch.set_num_threads(settings["threads"])
        started = time.monotonic()
        model, loading = load_policy(settings)
        model.generation_config.eos_token_id = settings["eos_token_id"]
        model.generation_config.pad_token_id = settings["pad_token_id"]
        load_seconds = time.monotonic() - started
        # A real finite warm forward/generation happens before any episode reset.
        warm_started = time.monotonic()
        with torch.inference_mode():
            forward = model(input_ids=torch.tensor([[1, 2]]), use_cache=False)
            if not torch.isfinite(forward.logits).all():
                raise ValueError("Non-finite CPU warm forward logits")
            del forward
            warm = model.generate(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.ones((1, 2), dtype=torch.long),
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )
        if warm.shape[-1] != 3:
            raise ValueError("CPU warm generation did not produce one token")
        send_record(
            connection,
            {
                "kind": "ready",
                "load_seconds": load_seconds,
                "warm_seconds": time.monotonic() - warm_started,
                "model_class": type(model).__name__,
                "loading": loading,
                "peak_rss_bytes": peak_rss_bytes(),
            },
        )
        while True:
            request = receive_record(connection)
            if request.get("kind") == "close":
                return
            if set(request) != {
                "id",
                "input_ids",
                "attention_mask",
                "max_new_tokens",
                "seed",
                "do_sample",
                "temperature",
                "top_p",
            }:
                raise ValueError("Unexpected generation request fields")
            torch.manual_seed(request["seed"])
            started = time.monotonic()
            inputs = torch.tensor([request["input_ids"]], dtype=torch.long)
            attention = torch.tensor([request["attention_mask"]], dtype=torch.long)
            options = {
                "max_new_tokens": request["max_new_tokens"],
                "do_sample": request["do_sample"],
                "use_cache": True,
            }
            if request["do_sample"]:
                options.update(temperature=request["temperature"], top_p=request["top_p"])
            timing = TokenTiming(started)
            # No past_key_values are retained or supplied across requests.
            with torch.inference_mode():
                output = model.generate(
                    input_ids=inputs, attention_mask=attention, streamer=timing, **options
                )
            ids = output[0, inputs.shape[-1] :].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            send_record(
                connection,
                {
                    "kind": "generation",
                    "id": request["id"],
                    "token_ids": ids,
                    "stop_reason": "eos" if ids and ids[-1] in eos else "length",
                    "seconds": time.monotonic() - started,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "first_token_seconds": timing.first - started
                    if timing.first is not None
                    else None,
                    "decode_seconds": timing.last - timing.first
                    if len(ids) > 1 and timing.first is not None
                    else None,
                },
            )
    except BaseException as exc:
        try:
            send_record(
                connection, {"kind": "error", "message": f"{type(exc).__name__}: {exc}"[:4000]}
            )
        except (OSError, ValueError):
            pass
    finally:
        connection.close()
