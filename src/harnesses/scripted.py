"""Deterministic Stage A driver. Reference edits are never model rollout data."""

from dataclasses import dataclass

from environments.coding import CodingEnvironment
from environments.patches import make_diff


@dataclass(frozen=True)
class EpisodeSummary:
    task_id: str
    episode_id: str
    variant: str
    reward: float
    status: str
    starter_hash: str
    submission_hash: str
    tool_calls: int
    trace_directory: str


def run_episode(
    environment: CodingEnvironment, task_id: str = "zero-timeout", *, variant: str = "correct"
) -> EpisodeSummary:
    """Script an authored attempt, independently grade it, and always release resources."""
    task = environment.catalog.get(task_id)
    target = task.variant(variant)
    environment.recorder.policy = {"kind": "scripted", "version": "stage-a-v1"}
    environment.reset(task_id)
    try:
        for result in (environment.list_files(), environment.read_file(task.module)):
            if not result["ok"]:
                raise RuntimeError("Scripted fixture action failed")
        patch = make_diff(task.starter, target)
        if patch and not environment.apply_patch(patch)["ok"]:
            raise RuntimeError("Authored patch rejected")
        environment.run_tests()
        environment.finish("scripted_finish")
        reward = environment.get_reward()
        state = environment.inspect()
        return EpisodeSummary(
            task.task_id,
            state["episode_id"],
            variant,
            reward,
            state["verdict"]["status"],
            task.starter_hash,
            state["verdict"]["submission_hash"],
            state["tool_calls"],
            str(environment.recorder.root),
        )
    finally:
        environment.cleanup()
