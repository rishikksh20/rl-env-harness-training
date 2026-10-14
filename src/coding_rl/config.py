"""Small, strict configuration records; no framework-specific training settings yet."""

import math
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
class Config:
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sandbox, SandboxConfig)
            or not isinstance(self.environment, EnvironmentConfig)
            or not isinstance(self.logging, LoggingConfig)
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
    }
    resolved = {key: _record(cls, data[key]) for key, cls in sections.items() if key in data}
    return _record(Config, {**data, **resolved})
