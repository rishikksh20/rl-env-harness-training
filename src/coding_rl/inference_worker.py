"""Trusted spawned process: model weights and generation, never candidate execution."""

import json
import resource
import sys
import time
from multiprocessing.connection import Connection

MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def send_record(connection: Connection, value: dict) -> None:
    data = json.dumps(value, allow_nan=False).encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Inference message exceeds byte limit")
    connection.send_bytes(data)


def receive_record(connection: Connection) -> dict:
    value = json.loads(connection.recv_bytes(MAX_MESSAGE_BYTES))
    if type(value) is not dict:
        raise ValueError("Expected inference record")
    return value


def worker_main(connection: Connection, settings: dict) -> None:
    """Spawn entry point; heavy libraries stay behind this model-only boundary."""
    try:
        import torch
        from transformers import Qwen3_5ForCausalLM

        torch.set_num_threads(settings["threads"])
        started = time.monotonic()
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            settings["name"],
            revision=settings["revision"],
            cache_dir=settings["cache_dir"],
            local_files_only=settings["offline"],
            trust_remote_code=False,
            dtype=torch.float32,
            output_loading_info=True,
            token=False,
        )
        loading = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in loading.items()
        }
        # The text loader may discard vision/MTP weights. No required language
        # parameters may be missing or randomly initialized.
        missing = loading.get("missing_keys", [])
        unexpected = loading.get("unexpected_keys", [])
        if missing or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"Incomplete language weight loading: {loading}")
        if any(
            not any(part in key for part in ("visual.", "vision.", "mtp.")) for key in unexpected
        ):
            raise ValueError(f"Unexplained checkpoint weights: {unexpected}")
        model.to("cpu").eval()
        model.requires_grad_(False)
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
                "peak_rss_bytes": _peak_rss(),
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
            # No past_key_values are retained or supplied across requests.
            with torch.inference_mode():
                output = model.generate(input_ids=inputs, attention_mask=attention, **options)
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
                    "peak_rss_bytes": _peak_rss(),
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


def _peak_rss() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024
