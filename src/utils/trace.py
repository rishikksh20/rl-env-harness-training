"""Ordered public-safe JSONL events; native learning tensors have separate audit artifacts."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from utils.errors import InfrastructureError

EVENTS = {
    "reset",
    "action",
    "observation",
    "finish",
    "submission",
    "verify",
    "reward",
    "infrastructure_error",
    "cleanup",
    "generation",
    "parse",
    "dispatch",
    "model_observation",
    "termination",
    "outcome",
}
LOGGER = logging.getLogger(__name__)


class Recorder:
    def __init__(self, root: Path, observer: Callable[[dict], None] | None = None):
        self.run_id = uuid.uuid4().hex
        self.root = root / self.run_id
        self.root.mkdir(parents=True)
        self.observer = observer
        self.episode_id: str | None = None
        self.task_id: str | None = None
        self.sequence = 0
        self.started = time.monotonic()
        self.policy = {"kind": "scripted", "version": "stage-a-v1"}

    def start(self, episode_id: str, task_id: str) -> None:
        self.episode_id, self.task_id = episode_id, task_id
        self.sequence = 0
        self.started = time.monotonic()

    def emit(self, event: str, **payload: object) -> None:
        if event not in EVENTS or self.episode_id is None:
            raise ValueError("Unknown trace event or no episode")
        record = {
            "schema_version": 2,
            "event": event,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "sequence": self.sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "policy": self.policy,
            "payload": payload,
        }
        try:
            with (self.root / f"{self.episode_id}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except OSError as exc:
            raise InfrastructureError("Cannot write episode trace") from exc
        self.sequence += 1
        LOGGER.debug("episode=%s event=%s sequence=%s", self.episode_id, event, record["sequence"])
        if self.observer:
            self.observer(record)

    def artifact(self, name: str, value: object) -> str:
        filename = f"{self.episode_id}-{name}.json" if self.episode_id else f"{name}.json"
        path = self.root / filename
        try:
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except OSError as exc:
            raise InfrastructureError("Cannot write episode artifact") from exc
        return path.name
