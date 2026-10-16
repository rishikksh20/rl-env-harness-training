"""Sequential held-out scheduling and explicit valid-attempt denominators."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from coding_rl.rollout.episode import ModelOutcome, run_model_episode
from config import validate_model_mode
from environments.verifier import grading_fingerprint
from utils.errors import InfrastructureError
from utils.seeding import derive_seed


def evaluation_profile(config, catalog, worker_identity: str, policy: dict) -> dict:
    """Comparable evaluation behavior, independent of weights and artifact paths."""
    return {
        "schema_version": 1,
        "split": config.evaluation.split,
        "attempts_per_task": config.evaluation.attempts_per_task,
        "seed": config.seed,
        "sandbox": {**asdict(config.sandbox), "image": worker_identity},
        "environment": asdict(config.environment),
        "rollout": asdict(config.rollout),
        "tokenizer": {
            "name": config.model.name,
            "revision": config.model.tokenizer_revision,
            **{
                key: policy.get(key)
                for key in (
                    "template_sha256",
                    "protocol",
                    "eos_token_id",
                    "pad_token_id",
                    "device",
                    "dtype",
                    "versions",
                    "threads",
                )
            },
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "starter_hash": task.starter_hash,
                "grading_fingerprint": grading_fingerprint(task),
            }
            for task in catalog.tasks(config.evaluation.split)
        ],
        "harness": policy.get("harness", {"driver": "manual", "contract": "single-call-v1"}),
    }


def compare_evaluations(baseline: dict, trained: dict) -> dict:
    profile = baseline.get("evaluation_profile")
    if not profile or profile != trained.get("evaluation_profile"):
        raise ValueError("Before/after evaluation profiles differ")
    from coding_rl.training.evidence import validate_evaluation

    validate_evaluation(baseline, complete=True)
    validate_evaluation(trained)
    before, after = baseline["metrics"], trained["metrics"]
    first, second = before["verified_pass_rate"], after["verified_pass_rate"]
    return {
        "baseline": before,
        "trained": after,
        "pass_rate_change": second - first if first is not None and second is not None else None,
        "equal_valid_coverage": before["valid"] == after["valid"],
    }


def aggregate(outcomes: list[ModelOutcome], *, planned: int) -> dict:
    if len({o.driver_kind for o in outcomes}) > 1 or any(
        o.driver_kind not in {"model", "external_harness"} or o.policy.get("kind") != "model"
        for o in outcomes
    ):
        raise ValueError("Only genuine model outcomes may enter baseline metrics")
    valid = [
        o
        for o in outcomes
        if o.completed
        and o.reward in (0.0, 1.0)
        and o.cleanup_complete
        and o.verdict is not None
        and o.verdict["reward"] == o.reward
        and o.infrastructure_error is None
    ]
    successes = sum(o.reward == 1.0 for o in valid)
    per_task = {}
    for task_id in sorted({o.task_id for o in outcomes}):
        attempts = [o for o in valid if o.task_id == task_id]
        per_task[task_id] = {
            "started": sum(o.task_id == task_id for o in outcomes),
            "valid": len(attempts),
            "successes": sum(o.reward == 1.0 for o in attempts),
            "verified_pass_rate": sum(o.reward for o in attempts) / len(attempts)
            if attempts
            else None,
            "verdicts": dict(Counter(o.verdict["status"] for o in attempts)),
            "mean_episode_seconds": sum(
                o.timings.get("episode_seconds", 0) for o in outcomes if o.task_id == task_id
            )
            / sum(o.task_id == task_id for o in outcomes),
            "generated_tokens": sum(o.generated_tokens for o in outcomes if o.task_id == task_id),
        }
    return {
        "planned": planned,
        "started": len(outcomes),
        "valid": len(valid),
        "incomplete": len(outcomes) - len(valid),
        "not_started": planned - len(outcomes),
        "retried": 0,
        "infrastructure_exclusions": len(outcomes) - len(valid),
        "successes": successes,
        "verified_pass_rate": successes / len(valid) if valid else None,
        "mean_binary_reward": successes / len(valid) if valid else None,
        "finish_reasons": dict(Counter(o.finish_reason for o in outcomes)),
        "per_task": per_task,
        "parser_errors": sum(o.parser_errors for o in outcomes),
        "proposed_calls": sum(o.proposed_calls for o in outcomes),
        "core_calls": sum(o.core_calls for o in outcomes),
        "accepted_edits": sum(o.accepted_edits for o in outcomes),
        "generated_tokens": sum(o.generated_tokens for o in outcomes),
        "costs": {
            "seconds": {
                key: sum(o.timings.get(key, 0) for o in outcomes)
                for key in (
                    "generation_seconds",
                    "tool_seconds",
                    "grading_seconds",
                    "cleanup_seconds",
                    "episode_seconds",
                )
            },
            "peak_rss_bytes": max(
                (o.timings.get("peak_rss_bytes", 0) for o in outcomes), default=0
            ),
            "mean_episode_seconds": sum(o.timings.get("episode_seconds", 0) for o in outcomes)
            / len(outcomes)
            if outcomes
            else None,
            "mean_generated_tokens": sum(o.generated_tokens for o in outcomes) / len(outcomes)
            if outcomes
            else None,
        },
        "parser_error_rate": sum(o.parser_errors for o in outcomes) / sum(o.turns for o in outcomes)
        if sum(o.turns for o in outcomes)
        else None,
        "rejected_call_rate": sum(o.rejected_calls for o in outcomes)
        / sum(o.proposed_calls for o in outcomes)
        if sum(o.proposed_calls for o in outcomes)
        else None,
        "sampling_note": "Settings are recorded per policy; greedy repeats are not independent samples",
    }


def evaluate(environment, backend, *, baseline: dict | None = None, episode_runner=None) -> dict:
    config = environment.config
    validate_model_mode(config)
    if backend.identity.get("kind") != "model":
        raise ValueError("Evaluation requires an actual model backend")
    tasks = environment.catalog.tasks(config.evaluation.split)
    if not tasks:
        raise ValueError("Selected evaluation split is empty")
    planned = len(tasks) * config.evaluation.attempts_per_task
    environment.sandbox.preflight()
    profile = evaluation_profile(
        config,
        environment.catalog,
        environment.sandbox.identity,
        backend.identity,
    )
    if baseline is not None and baseline.get("evaluation_profile") != profile:
        raise ValueError("Before/after evaluation profiles differ; refusing incomparable sampling")
    if baseline is not None:
        from coding_rl.training.evidence import validate_evaluation

        validate_evaluation(baseline, complete=True)
    outcomes = []
    report_path = Path(config.evaluation.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = asdict(config)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    report = {}
    for task in tasks:
        for attempt in range(config.evaluation.attempts_per_task):
            try:
                outcome = (episode_runner or run_model_episode)(
                    environment, backend, task.task_id, attempt=attempt
                )
            except (InfrastructureError, OSError, KeyboardInterrupt) as exc:
                # Startup/reload can fail before reset allocates an episode.
                backend.close()
                environment.cleanup()
                outcome = ModelOutcome(
                    task_id=task.task_id,
                    episode_id="",
                    policy=backend.identity,
                    seed=derive_seed(config.seed, task.task_id, attempt),
                    finish_reason="cancelled"
                    if isinstance(exc, KeyboardInterrupt)
                    else "infrastructure_failure",
                    completed=False,
                    reward=None,
                    verdict=None,
                    starter_hash=task.starter_hash,
                    submission_hash=None,
                    grading_fingerprint=grading_fingerprint(task),
                    proposed_calls=0,
                    core_calls=0,
                    accepted_calls=0,
                    accepted_edits=0,
                    turns=0,
                    generated_tokens=0,
                    parser_errors=0,
                    rejected_calls=0,
                    cleanup_complete=environment.workspace is None,
                    trace_directory=str(environment.recorder.root),
                    final_response="",
                    infrastructure_error=f"Before episode completion: {exc}",
                    driver_kind="external_harness" if episode_runner else "model",
                )
            outcomes.append(outcome)
            report = {
                "schema_version": 2,
                "split": config.evaluation.split,
                "config": configuration,
                "config_sha256": fingerprint,
                "policy": backend.identity,
                "worker_identity": environment.sandbox.identity,
                "model_startups": backend.startups,
                "evaluation_profile": profile,
                "metrics": aggregate(outcomes, planned=planned),
                "attempts": [asdict(o) for o in outcomes],
                "limitation": "Four fixtures and one held-out family measure integration, not broad coding ability",
            }
            if baseline is not None:
                report["comparison"] = compare_evaluations(baseline, report)
            # Save after each attempt; cancellation never erases earlier outcomes.
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            try:
                temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                temporary.replace(report_path)
            except OSError as exc:
                raise InfrastructureError("Cannot save evaluation report") from exc
            if outcome.finish_reason == "cancelled" or not outcome.cleanup_complete:
                return report
    return report
