"""Private assertion controller: candidate execution always stays in the worker."""

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

from coding_rl.errors import InfrastructureError
from coding_rl.sandbox import Sandbox
from coding_rl.tasks import Task

VERIFIER_VERSION = "json-cases-v1"


def grading_fingerprint(task: Task) -> str:
    """Identify the protected plan without exporting its contents."""
    plan = {
        "version": VERIFIER_VERSION,
        "starter": task.starter_hash,
        "module": task.module,
        "function": task.function,
        "cases": [{"kwargs": c.kwargs, "expected": c.expected} for c in task.private_cases],
    }
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Verdict:
    reward: float
    status: str
    passed: int
    total: int
    submission_hash: str


def _json_value(value: object, depth: int = 0) -> bool:
    if depth > 20:
        return False
    if value is None or type(value) in {bool, int, str}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item, depth + 1) for key, item in value.items())
    return False


def _same(actual: object, expected: object) -> bool:
    """Strict JSON equality avoids Python's False == 0 false positive."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[k], expected[k]) for k in actual
        )
    return actual == expected


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def verify(
    task: Task,
    snapshot: Mapping[str, str],
    sandbox: Sandbox,
    *,
    submission_hash: str,
    deadline: float,
) -> Verdict:
    """Reconstruct independently and check real callable outputs against private cases."""
    if set(snapshot) != set(task.starter) or any(
        snapshot[name] != content
        for name, content in task.starter.items()
        if name not in task.editable
    ):
        return Verdict(0.0, "invalid_submission", 0, len(task.private_cases), submission_hash)
    candidate = {**task.starter, **{name: snapshot[name] for name in task.editable}}
    remaining = deadline - time.monotonic()
    visible = sandbox.visible_tests(task, candidate, timeout=remaining)
    if visible.status == "timeout" and remaining < sandbox.config.timeout_seconds:
        raise InfrastructureError("Grading deadline expired before the full command limit")
    if visible.status != "completed" or visible.returncode:
        return Verdict(
            0.0,
            f"visible_{visible.status}" if visible.status != "completed" else "visible_failed",
            0,
            len(task.private_cases),
            submission_hash,
        )
    passed = 0
    for case in task.private_cases:
        # One fresh process/container per private case: no state from visible tests.
        remaining = deadline - time.monotonic()
        result = sandbox.invoke(task, candidate, case.kwargs, timeout=remaining)
        if result.status == "timeout" and remaining < sandbox.config.timeout_seconds:
            raise InfrastructureError("Grading deadline expired before the full command limit")
        if result.status != "completed":
            return Verdict(0.0, result.status, passed, len(task.private_cases), submission_hash)
        if result.returncode:
            return Verdict(0.0, "candidate_error", passed, len(task.private_cases), submission_hash)
        try:
            data = json.loads(result.stdout, object_pairs_hook=_unique_object)
            valid = (
                type(data) is dict
                and data.get("status") == "ok"
                and set(data) == {"status", "value"}
                and _json_value(data["value"])
            )
        except (ValueError, RecursionError):
            valid = False
        if not valid:
            return Verdict(
                0.0, "candidate_protocol_error", passed, len(task.private_cases), submission_hash
            )
        if not _same(data["value"], case.expected):
            return Verdict(0.0, "incorrect", passed, len(task.private_cases), submission_hash)
        passed += 1
    if not task.private_cases:
        raise InfrastructureError("Trusted private case set is empty")
    return Verdict(1.0, "correct", passed, len(task.private_cases), submission_hash)
