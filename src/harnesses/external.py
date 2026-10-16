"""Trusted subprocess harnesses compose inference and tools through a local bridge."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from coding_rl.rollout.episode import ModelOutcome, public_messages
from config import validate_model_mode
from environments.verifier import grading_fingerprint
from harnesses.bridge import HarnessBridge
from harnesses.protocol import SYSTEM_PROMPT
from utils.errors import InfrastructureError
from utils.process import supervise
from utils.seeding import derive_seed


def harness_identity(driver: str, command: list[str]) -> dict:
    if driver == "opencode":
        binary = shutil.which("opencode")
        if not binary:
            raise ValueError("OpenCode is missing from PATH; install it or select --driver custom")
        version = subprocess.run(
            [binary, "--version"], text=True, capture_output=True, timeout=10, check=True
        ).stdout.strip()
        return {
            "driver": driver,
            "version": version,
            "contract": "four-tool-bridge-v1",
            "training": False,
        }
    if not command:
        raise ValueError("Custom harness requires an argv command after --")
    return {
        "driver": driver,
        "command": command,
        "files": {
            str(Path(arg).resolve()): hashlib.sha256(Path(arg).read_bytes()).hexdigest()
            for arg in command
            if Path(arg).is_file()
        },
        "contract": "four-tool-bridge-v1",
        "training": False,
    }


def opencode_config(bridge, config) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["coding-rl"],
        "model": "coding-rl/policy",
        "small_model": "coding-rl/policy",
        "provider": {
            "coding-rl": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Coding RL",
                "options": {"baseURL": bridge.url + "/v1", "apiKey": bridge.token},
                "models": {
                    "policy": {
                        "name": config.model.name,
                        "tool_call": True,
                        "limit": {
                            "context": config.rollout.context_tokens,
                            "output": config.rollout.max_new_tokens,
                        },
                    }
                },
            }
        },
        "mcp": {
            "coding_rl": {
                "type": "remote",
                "url": bridge.url + "/mcp",
                "oauth": False,
                "headers": {"Authorization": "Bearer " + bridge.token},
                "timeout": 30000,
            }
        },
        "permission": {"*": "deny", "coding_rl_*": "allow"},
        "tools": {"*": False, "coding_rl_*": True},
        "agent": {
            "coding_rl": {
                "mode": "primary",
                "prompt": SYSTEM_PROMPT,
                "steps": config.rollout.max_turns,
                "permission": {"*": "deny", "coding_rl_*": "allow"},
            }
        },
        "compaction": {"auto": False, "prune": False},
        "share": "disabled",
        "autoupdate": False,
    }


def run_harness_episode(
    environment,
    backend,
    task_id: str,
    *,
    attempt: int = 0,
    driver: str = "opencode",
    command: list[str] | None = None,
    identity: dict | None = None,
) -> ModelOutcome:
    config = environment.config
    validate_model_mode(config)
    command = command or []
    identity = identity or harness_identity(driver, command)
    environment.sandbox.preflight()
    backend.start()
    policy = {
        **backend.identity,
        "harness": identity,
        "sampling": {
            "do_sample": config.rollout.do_sample,
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
        },
        "seed": derive_seed(config.seed, task_id, attempt),
    }
    environment.recorder.policy = policy
    environment.reset(task_id)
    recorder = environment.recorder
    bridge = HarnessBridge(environment, backend, task_id, attempt)
    started = time.monotonic()
    failure, reward, process = None, None, None
    grading_seconds = 0.0
    bridge.start()
    try:
        with tempfile.TemporaryDirectory(prefix="coding-rl-harness-") as temporary:
            root = Path(temporary)
            task_path = root / "task.json"
            task_path.write_text(
                json.dumps(
                    {
                        "messages": public_messages(environment.task.public()),
                        "max_turns": config.rollout.max_turns,
                        "task_id": task_id,
                    }
                )
            )
            child_env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}
            }
            child_env.update(
                {
                    "CODING_RL_BRIDGE_URL": bridge.url,
                    "CODING_RL_BRIDGE_TOKEN": bridge.token,
                    "CODING_RL_TASK": str(task_path),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            if driver == "opencode":
                child_env.update(
                    {
                        "XDG_CONFIG_HOME": str(root / "config"),
                        "XDG_DATA_HOME": str(root / "data"),
                        "XDG_STATE_HOME": str(root / "state"),
                        "XDG_CACHE_HOME": str(root / "cache"),
                        "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config(bridge, config)),
                        "OPENCODE_DISABLE_AUTOUPDATE": "true",
                        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                    }
                )
                prompt = (
                    environment.task.prompt
                    + "\nEditable files: "
                    + ", ".join(environment.task.editable)
                    + "\nUse only the coding_rl MCP tools. Inspect source, apply a unified diff, run tests, then finish."
                )
                command = [
                    shutil.which("opencode"),
                    "--pure",
                    "run",
                    "--format",
                    "json",
                    "--title",
                    "coding-rl",
                    "--model",
                    "coding-rl/policy",
                    "--agent",
                    "coding_rl",
                    prompt,
                ]
            process = supervise(
                command,
                cwd=root,
                env=child_env,
                timeout=max(0.01, bridge.deadline - time.monotonic()),
                output_bytes=1024 * 1024,
            )
            recorder.artifact("harness-process", asdict(process))
            if process.status == "timeout":
                bridge.reason = "episode_timeout"
                backend.close()
            elif process.status != "completed" or process.returncode:
                raise InfrastructureError(
                    f"External harness failed: status={process.status}, exit={process.returncode}; inspect harness-process artifact"
                )
            if bridge.failure:
                raise InfrastructureError(bridge.failure)
            if bridge.turns == 0:
                raise InfrastructureError(
                    "External harness produced no model generation; inspect harness-process artifact"
                )
        # Stop requests before freezing the submission. No external process gets reward access.
        bridge.close()
        with bridge.lock:
            environment.finish(bridge.reason)
            before = time.monotonic()
            reward = environment.get_reward()
            grading_seconds = time.monotonic() - before - environment.cleanup_seconds
    except KeyboardInterrupt:
        bridge.reason, failure = "cancelled", "External harness interrupted"
        backend.close()
    except (InfrastructureError, OSError, ValueError) as exc:
        bridge.reason, failure = "infrastructure_failure", f"{type(exc).__name__}: {exc}"
        backend.close()
    finally:
        bridge.close()
        with bridge.lock:
            try:
                try:
                    if environment.phase == "active":
                        environment.finish(bridge.reason)
                finally:
                    environment.cleanup()
            except (InfrastructureError, OSError) as exc:
                failure = f"Cleanup/freeze incomplete: {exc}"
                reward = None
    state = environment.inspect()
    completed = failure is None and reward is not None and environment.workspace is None
    outcome = ModelOutcome(
        task_id=task_id,
        episode_id=environment.episode_id,
        policy=policy,
        seed=policy["seed"],
        finish_reason=bridge.reason,
        completed=completed,
        reward=reward if completed else None,
        verdict=state["verdict"],
        starter_hash=environment.task.starter_hash,
        submission_hash=state["source_hash"],
        grading_fingerprint=grading_fingerprint(environment.task),
        proposed_calls=bridge.proposed,
        core_calls=environment.calls,
        accepted_calls=bridge.accepted,
        accepted_edits=bridge.edits,
        turns=bridge.turns,
        generated_tokens=bridge.tokens,
        parser_errors=bridge.parser_errors,
        rejected_calls=bridge.rejected,
        cleanup_complete=environment.workspace is None,
        trace_directory=str(recorder.root),
        final_response=bridge.final,
        infrastructure_error=failure,
        timings={
            **bridge.timings,
            "episode_seconds": time.monotonic() - started,
            "grading_seconds": grading_seconds,
            "cleanup_seconds": environment.cleanup_seconds,
        },
        driver_kind="external_harness",
    )
    recorder.artifact("outcome", asdict(outcome))
    recorder.emit("outcome", **asdict(outcome))
    return outcome
