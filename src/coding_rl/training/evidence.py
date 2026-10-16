"""Validate retained local evidence before admitting training or comparisons."""

import json
from pathlib import Path

from environments.verifier import grading_fingerprint
from tasks.catalog import file_digest
from utils.seeding import derive_seed


def validate_outcome(record: dict, task, policy: dict) -> None:
    verdict = record.get("verdict") or {}
    if not (
        record.get("driver_kind") == "model"
        and record.get("completed") is True
        and record.get("cleanup_complete") is True
        and record.get("infrastructure_error") is None
        and type(record.get("reward")) in (int, float)
        and record["reward"] in (0, 1)
        and verdict.get("reward") == record["reward"]
        and type(verdict.get("passed")) is int
        and type(verdict.get("total")) is int
        and 0 <= verdict["passed"] <= verdict["total"] == len(task.private_cases)
        and bool(verdict.get("status"))
        and (verdict["status"] == "correct") == (record["reward"] == 1)
        and (record["reward"] != 1 or verdict["passed"] == verdict["total"])
        and record.get("starter_hash") == task.starter_hash
        and record.get("grading_fingerprint") == grading_fingerprint(task)
        and isinstance(record.get("submission_hash"), str)
        and verdict.get("submission_hash") == record["submission_hash"]
        and record.get("policy", {}).get("kind") == "model"
        and all(record["policy"].get(k) == v for k, v in policy.items())
    ):
        raise ValueError("Incomplete or inconsistent model outcome/verdict/policy")


def validate_evaluation(report: dict, *, complete: bool = False) -> None:
    from coding_rl.rollout.episode import ModelOutcome
    from coding_rl.rollout.evaluate import aggregate

    profile = report.get("evaluation_profile") or {}
    if report.get("split") != profile.get("split"):
        raise ValueError("Evaluation split differs from its profile")
    if report.get("worker_identity") != profile.get("sandbox", {}).get("image"):
        raise ValueError("Evaluation worker identity differs from its profile")
    tasks = profile.get("tasks", [])
    count = profile.get("attempts_per_task", 0)
    if (
        type(count) is not int
        or count < 1
        or not tasks
        or len({t["task_id"] for t in tasks}) != len(tasks)
    ):
        raise ValueError("Evaluation requires a nonempty declared task/attempt schedule")
    schedule = {
        (t["task_id"], derive_seed(profile["seed"], t["task_id"], i)): t
        for t in tasks
        for i in range(count)
    }
    records = report.get("attempts", [])
    seen, episodes = set(), set()
    for record in records:
        key = (record["task_id"], record["seed"])
        if key not in schedule or key in seen:
            raise ValueError("Duplicate or unplanned evaluation attempt")
        seen.add(key)
        if record.get("episode_id"):
            if record["episode_id"] in episodes:
                raise ValueError("Duplicate evaluation episode")
            episodes.add(record["episode_id"])
        if any(record.get(k) != schedule[key][k] for k in ("starter_hash", "grading_fingerprint")):
            raise ValueError("Evaluation attempt differs from declared task identity")
        if not all(record.get("policy", {}).get(k) == v for k, v in report["policy"].items()):
            raise ValueError("Evaluation attempt differs from report policy")
    measured = aggregate([ModelOutcome(**r) for r in records], planned=len(schedule))
    if report.get("metrics") != measured:
        raise ValueError("Evaluation metrics differ from the actual attempt records")
    if complete and (len(seen) != len(schedule) or measured["valid"] != len(schedule)):
        raise ValueError("Evaluation requires complete held-out coverage")


def validate_development_trace(record: dict, task, worker: str) -> None:
    root, episode = Path(record["trace_directory"]), record["episode_id"]
    events = [json.loads(line) for line in (root / f"{episode}.jsonl").read_text().splitlines()]
    if not events or any(
        e["episode_id"] != episode
        or e["task_id"] != task.task_id
        or e["policy"] != record["policy"]
        or e["sequence"] != i
        for i, e in enumerate(events)
    ):
        raise ValueError("Development trace identity/order does not match its outcome")
    if not any(e["event"] == "outcome" and e["payload"] == record for e in events):
        raise ValueError("Development outcome is absent from its episode trace")
    observations = [
        json.loads(e["payload"]["content"])
        for e in events
        if e["event"] == "model_observation" and e["payload"].get("name") == "run_tests"
    ]
    if not any(
        o.get("ok") is True and o.get("returncode") in (0, 1) and not o.get("timed_out")
        for o in observations
    ):
        raise ValueError("Development requires model-visible healthy run_tests feedback")
    if not any(
        e["event"] == "dispatch"
        and e["payload"].get("tool") == "apply_patch"
        and e["payload"].get("accepted") is True
        for e in events
    ):
        raise ValueError("Development requires an accepted model edit")
    verifies = [e["payload"] for e in events if e["event"] == "verify"]
    if (
        len(verifies) != 1
        or verifies[0].get("worker_identity") != worker
        or any(verifies[0].get(k) != v for k, v in record["verdict"].items())
    ):
        raise ValueError("Development verifier evidence does not match")
    submission = json.loads((root / f"{episode}-submission.json").read_text())
    files = submission["files"]
    if (
        file_digest(files) != record["submission_hash"]
        or submission.get("starter_hash") != task.starter_hash
        or set(files) != set(task.starter)
        or any(files[k] != v for k, v in task.starter.items() if k not in task.editable)
    ):
        raise ValueError("Development frozen source differs from its task/verdict")
