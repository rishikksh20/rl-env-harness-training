"""Lazy CPU Transformers backend with one cancellable, persistent model process."""

import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import time
import uuid
from dataclasses import asdict, dataclass

from coding_rl.checkpoints import policy_identity
from coding_rl.config import ModelConfig, RolloutConfig
from coding_rl.errors import InfrastructureError
from coding_rl.inference_worker import receive_record, send_record, worker_main
from coding_rl.protocol import PROTOCOL_VERSION, function_tools, parse_response

PINNED_PACKAGES = {"torch": "2.14.1", "transformers": "5.19.0", "huggingface-hub": "2.2.0"}


class GenerationTimeout(TimeoutError):
    """Healthy generation horizon; the worker has been terminated and joined."""


@dataclass(frozen=True)
class Generation:
    request_id: str
    token_ids: list[int]
    raw_text: str
    stop_reason: str
    seconds: float
    peak_rss_bytes: int = 0


def derive_seed(base: int, task_id: str, attempt: int, turn: int = 0) -> int:
    data = json.dumps([base, task_id, attempt, turn]).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def dependency_versions() -> dict:
    versions = {}
    for package, expected in PINNED_PACKAGES.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InfrastructureError(
                "Missing CPU inference packages; run uv sync --extra rollout"
            ) from exc
        if versions[package].split("+")[0] != expected:
            raise InfrastructureError(
                f"Unsupported {package} version; run uv sync --locked --extra rollout"
            )
    return versions


class TransformersBackend:
    """Controller owns tokenizer; a clean spawned child owns all model weights."""

    def __init__(self, config: ModelConfig):
        self.config = config
        source_identity = policy_identity(config)
        self.versions = dependency_versions()
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.name,
                revision=config.tokenizer_revision,
                cache_dir=config.cache_dir,
                local_files_only=config.offline,
                trust_remote_code=False,
                token=False,
            )
        except (ImportError, OSError, ValueError) as exc:
            raise InfrastructureError(f"Cannot load pinned tokenizer: {exc}") from exc
        template = self.tokenizer.chat_template
        if not isinstance(template, str) or "<function=" not in template:
            raise InfrastructureError(
                "Checkpoint does not provide the supported Qwen3.5 XML template"
            )
        self.identity = {
            "kind": "model",
            "backend": "transformers",
            "checkpoint": config.name,
            "revision": config.revision,
            "tokenizer_revision": config.tokenizer_revision,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "protocol": PROTOCOL_VERSION,
            "device": config.device,
            "dtype": config.dtype,
            "versions": self.versions,
            "hardware": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "threads": config.threads,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "policy_source": source_identity,
        }
        self.process = None
        self.connection = None
        self.startups: list[dict] = []
        probe = [{"role": "user", "content": "Inspect the repository."}]
        _, prefix = self.render(probe)
        parse_response("Ready.<|im_end|>", generation_prefix=prefix)

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, messages: list[dict]) -> tuple[list[int], str]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=function_tools(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Keep the actual generation prefix with its prefilled reasoning closure.
        prefix = text[text.rfind("<|im_start|>assistant\n") :]
        return ids, prefix

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.close()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(
            target=worker_main,
            args=(
                child,
                {
                    **asdict(self.config),
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                },
            ),
            name="coding-rl-inference",
            daemon=True,
        )
        started = time.monotonic()
        try:
            self.process.start()
            child.close()
            result = self._receive(self.config.startup_seconds, startup=True)
            if result.get("kind") != "ready" or not isinstance(result.get("model_class"), str):
                raise InfrastructureError("Invalid inference startup response")
            self.startups.append({**result, "startup_seconds": time.monotonic() - started})
            self.identity["model_class"] = result["model_class"]
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout: float, *, startup: bool = False) -> dict:
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self.connection.poll(min(0.05, max(0, deadline - time.monotonic()))):
                    result = receive_record(self.connection)
                    if result.get("kind") == "error":
                        raise InfrastructureError(
                            f"Inference worker failed: {result.get('message')}"
                        )
                    return result
                if not self.process.is_alive():
                    raise InfrastructureError("Inference worker exited without a response")
        except (OSError, EOFError, ValueError) as exc:
            self.close()
            raise InfrastructureError("Invalid or lost inference-worker response") from exc
        self.close()
        if startup:
            raise InfrastructureError("Model startup deadline expired")
        raise GenerationTimeout("Generation deadline expired; inference worker terminated")

    def generate(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        seed: int,
        timeout: float,
        sampling: RolloutConfig,
    ) -> Generation:
        if self.process is None or not self.process.is_alive():
            raise InfrastructureError(
                "Inference worker is not ready; start before resetting an episode"
            )
        request_id = uuid.uuid4().hex
        try:
            send_record(
                self.connection,
                {
                    "id": request_id,
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "do_sample": sampling.do_sample,
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                },
            )
            result = self._receive(timeout)
            ids = result.get("token_ids")
            if (
                result.get("kind") != "generation"
                or result.get("id") != request_id
                or type(ids) is not list
                or not 0 < len(ids) <= max_new_tokens
                or any(type(i) is not int or not 0 <= i < len(self.tokenizer) for i in ids)
                or result.get("stop_reason") not in {"eos", "length"}
            ):
                raise InfrastructureError("Mismatched or invalid inference response")
            return Generation(
                request_id,
                ids,
                self.tokenizer.decode(ids, skip_special_tokens=False),
                result["stop_reason"],
                result["seconds"],
                result["peak_rss_bytes"],
            )
        except GenerationTimeout:
            raise
        except (OSError, ValueError) as exc:
            self.close()
            raise InfrastructureError("Cannot exchange inference request") from exc
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.process is not None:
            if self.process.is_alive() and self.connection is not None:
                try:
                    send_record(self.connection, {"kind": "close"})
                except (OSError, ValueError):
                    pass
                self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
            else:
                self.process.join(timeout=0.2)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
