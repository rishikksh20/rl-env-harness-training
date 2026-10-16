"""Strict configuration records; optional model packages stay behind runtime boundaries."""

import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _positive(name: str, value: Any, *, integer: bool = False) -> None:
    expected = (int,) if integer else (int, float)
    if (
        type(value) not in expected
        or (type(value) is float and not math.isfinite(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


@dataclass(frozen=True)
class SandboxConfig:
    kind: str = "docker"
    image: str = "coding-rl-sandbox:stage-a"
    timeout_seconds: float = 20
    memory_mb: int = 512
    cpus: float = 1.0
    pids_limit: int = 64
    output_bytes: int = 16384

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in {"docker", "trusted_local"}:
            raise ValueError("sandbox.kind must be docker or trusted_local")
        if not isinstance(self.image, str) or not self.image or self.image.startswith("-"):
            raise ValueError("sandbox.image must be a nonempty image reference")
        for name in ("timeout_seconds", "cpus"):
            _positive(f"sandbox.{name}", getattr(self, name))
        for name in ("memory_mb", "pids_limit", "output_bytes"):
            _positive(f"sandbox.{name}", getattr(self, name), integer=True)


@dataclass(frozen=True)
class EnvironmentConfig:
    max_tool_calls: int = 6
    episode_seconds: float = 120
    grading_seconds: float = 120
    verifier_retries: int = 1
    max_file_bytes: int = 65536
    max_patch_bytes: int = 65536
    max_list_entries: int = 200

    def __post_init__(self) -> None:
        for name in ("max_tool_calls", "max_file_bytes", "max_patch_bytes", "max_list_entries"):
            _positive(f"environment.{name}", getattr(self, name), integer=True)
        for name in ("episode_seconds", "grading_seconds"):
            _positive(f"environment.{name}", getattr(self, name))
        if type(self.verifier_retries) is not int or not 0 <= self.verifier_retries <= 3:
            raise ValueError("environment.verifier_retries must be an integer from 0 to 3")


@dataclass(frozen=True)
class LoggingConfig:
    trace_dir: str = "outputs/trajectories"
    level: str = "INFO"

    def __post_init__(self) -> None:
        if not isinstance(self.trace_dir, str) or not self.trace_dir:
            raise ValueError("logging.trace_dir must be a nonempty path")
        if not isinstance(self.level, str) or self.level not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        }:
            raise ValueError("logging.level must be DEBUG, INFO, WARNING, or ERROR")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    tokenizer_revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    device: str = "cpu"
    dtype: str = "float32"
    cache_dir: str = "outputs/model-cache"
    offline: bool = True
    startup_seconds: float = 600
    threads: int = 4
    source: str = "hub"
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.startswith("-"):
            raise ValueError("model.name must be a checkpoint ID")
        for name in ("revision", "tokenizer_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
                raise ValueError(f"model.{name} must be an immutable 40-character commit")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("The Phase 2 profile supports CPU float32 only")
        if type(self.offline) is not bool:
            raise ValueError("model.offline must be boolean")
        if not isinstance(self.cache_dir, str) or not self.cache_dir:
            raise ValueError("model.cache_dir must be a nonempty path")
        _positive("model.startup_seconds", self.startup_seconds)
        _positive("model.threads", self.threads, integer=True)
        if not isinstance(self.source, str) or self.source not in {"hub", "full", "lora"}:
            raise ValueError("model.source must be hub, full, or lora")
        if self.source == "hub":
            if self.checkpoint is not None:
                raise ValueError("Hub policies cannot specify a local checkpoint")
        elif not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("Local policies require model.checkpoint")
        elif not self.offline:
            raise ValueError("Local policy evaluation requires offline base/tokenizer loading")


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = "lora"
    output_dir: str = "outputs/training"
    max_groups: int = 8
    group_size: int = 4
    completion_tokens: int = 768
    context_tokens: int = 2048
    observation_tokens: int = 256
    tool_iterations: int = 6
    learning_rate: float = 1e-6
    temperature: float = 0.7
    top_p: float = 0.8
    memory_limit_mb: int = 16384
    wall_seconds: float = 3600
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    baseline_report: str | None = None
    development_outcome: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"full", "lora"}:
            raise ValueError("training.mode must be full or lora")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("training.output_dir must be a nonempty path")
        for name in (
            "max_groups",
            "group_size",
            "completion_tokens",
            "context_tokens",
            "observation_tokens",
            "tool_iterations",
            "memory_limit_mb",
            "lora_rank",
            "lora_alpha",
        ):
            _positive(f"training.{name}", getattr(self, name), integer=True)
        for name in ("learning_rate", "temperature", "top_p", "wall_seconds"):
            _positive(f"training.{name}", getattr(self, name))
        if self.group_size != 4 or self.max_groups > 8:
            raise ValueError("The native pilot requires four attempts and at most eight groups")
        if self.top_p > 1 or self.observation_tokens < 128:
            raise ValueError("training requires top_p <= 1 and observation_tokens >= 128")
        if self.context_tokens <= self.completion_tokens + self.observation_tokens:
            raise ValueError("training.context_tokens must reserve prompt and observation space")
        if (
            not isinstance(self.lora_targets, (tuple, list))
            or not self.lora_targets
            or any(
                not isinstance(name, str) or not name.isidentifier() for name in self.lora_targets
            )
            or len(set(self.lora_targets)) != len(self.lora_targets)
        ):
            raise ValueError("training.lora_targets must contain unique module leaf names")
        object.__setattr__(self, "lora_targets", tuple(self.lora_targets))
        for name in ("baseline_report", "development_outcome"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"training.{name} must be a nonempty path")


@dataclass(frozen=True)
class RolloutConfig:
    max_turns: int = 8
    parser_recovery_limit: int = 2
    max_new_tokens: int = 256
    episode_tokens: int = 768
    context_tokens: int = 2048
    observation_tokens: int = 256
    generation_seconds: float = 90
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "max_turns",
            "parser_recovery_limit",
            "max_new_tokens",
            "episode_tokens",
            "context_tokens",
            "observation_tokens",
        ):
            _positive(f"rollout.{name}", getattr(self, name), integer=True)
        for name in ("generation_seconds", "temperature", "top_p"):
            _positive(f"rollout.{name}", getattr(self, name))
        if self.top_p > 1 or type(self.do_sample) is not bool:
            raise ValueError("rollout requires top_p <= 1 and boolean do_sample")
        if self.observation_tokens < 128:
            raise ValueError("rollout.observation_tokens must reserve at least 128 tokens")
        if self.context_tokens <= self.max_new_tokens + self.observation_tokens:
            raise ValueError(
                "rollout.context_tokens cannot fit generation and observation reserves"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "eval"
    attempts_per_task: int = 4
    report_path: str = "outputs/evaluation.json"

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or self.split not in {"train", "eval"}:
            raise ValueError("evaluation.split must be train or eval")
        _positive("evaluation.attempts_per_task", self.attempts_per_task, integer=True)
        if not isinstance(self.report_path, str) or not self.report_path:
            raise ValueError("evaluation.report_path must be a nonempty path")


@dataclass(frozen=True)
class Config:
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    model: ModelConfig | None = None
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sandbox, SandboxConfig)
            or not isinstance(self.environment, EnvironmentConfig)
            or not isinstance(self.logging, LoggingConfig)
            or (self.model is not None and not isinstance(self.model, ModelConfig))
            or not isinstance(self.rollout, RolloutConfig)
            or not isinstance(self.evaluation, EvaluationConfig)
            or (self.training is not None and not isinstance(self.training, TrainingConfig))
        ):
            raise ValueError("Configuration sections must use validated records")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _record(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = data.keys() - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> Config:
    """Load YAML, reject unknown fields, and validate settings before allocation."""
    if path is None:
        return Config()
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML configuration") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    sections = {
        "sandbox": SandboxConfig,
        "environment": EnvironmentConfig,
        "logging": LoggingConfig,
        "model": ModelConfig,
        "rollout": RolloutConfig,
        "evaluation": EvaluationConfig,
        "training": TrainingConfig,
    }
    resolved = {
        key: _record(cls, data[key])
        for key, cls in sections.items()
        if key in data and not (key in {"model", "training"} and data[key] is None)
    }
    return _record(Config, {**data, **resolved})


def validate_model_mode(config: Config) -> None:
    if config.model is None:
        raise ValueError("Model mode requires a model section; use configs/rollout-cpu.yaml")
    if config.sandbox.kind != "docker":
        raise ValueError(
            "Real model mode requires Docker; trusted_local accepts authored fixtures only"
        )


def validate_training_mode(config: Config) -> None:
    validate_model_mode(config)
    if config.training is None:
        raise ValueError("Training requires a training section; use configs/train-cpu.yaml")
    if config.model.source != "hub":
        raise ValueError(
            "The pilot starts from a pinned Hub policy; training resume is unsupported"
        )
