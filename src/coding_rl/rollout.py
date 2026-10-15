"""Manual inference episodes. This path receives public task data, never repairs."""

import json
import time
from dataclasses import asdict, dataclass, field

from coding_rl.config import validate_model_mode
from coding_rl.environment import CodingEnvironment
from coding_rl.errors import InfrastructureError
from coding_rl.models import GenerationTimeout, derive_seed
from coding_rl.protocol import SYSTEM_PROMPT, ProtocolError, dispatch, parse_response
from coding_rl.tasks import PublicTask
from coding_rl.verifier import grading_fingerprint


@dataclass(frozen=True)
class ModelOutcome:
    task_id: str
    episode_id: str
    policy: dict
    seed: int
    finish_reason: str
    completed: bool
    reward: float | None
    verdict: dict | None
    starter_hash: str
    submission_hash: str | None
    grading_fingerprint: str
    proposed_calls: int
    core_calls: int
    accepted_calls: int
    accepted_edits: int
    turns: int
    generated_tokens: int
    parser_errors: int
    rejected_calls: int
    cleanup_complete: bool
    trace_directory: str
    final_response: str
    infrastructure_error: str | None
    timings: dict = field(default_factory=dict)
    driver_kind: str = "model"


def public_messages(task: PublicTask) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt + "\nEditable files: " + ", ".join(task.editable)},
    ]


def bounded_observation(result: dict, backend, token_limit: int) -> str:
    """Shrink payloads inside a valid JSON envelope, preserving result metadata."""
    value = {**result, "truncated": result.get("truncated", False)}
    while True:
        text = json.dumps(value, ensure_ascii=True, allow_nan=False)
        if backend.count_text(text) <= token_limit:
            return text
        candidates = [
            (len(v), k)
            for k, v in value.items()
            if isinstance(v, (str, list)) and k not in {"status", "error", "source_hash"} and v
        ]
        if not candidates:
            raise InfrastructureError("Observation metadata cannot fit its token reserve")
        _, key = max(candidates)
        value[key] = value[key][: len(value[key]) // 2]
        value["truncated"] = True


def run_model_episode(
    environment: CodingEnvironment, backend, task_id: str = "zero-timeout", *, attempt: int = 0
) -> ModelOutcome:
    """Warm before reset; preserve every accepted action; independently judge files.

    Fake generation backends are useful in tests but must identify themselves as
    such, and their outcomes cannot enter model baseline metrics.
    """
    config = environment.config
    validate_model_mode(config)
    public = environment.catalog.get(task_id).public()
    settings = config.rollout
    seed = derive_seed(config.seed, task_id, attempt)
    messages = public_messages(public)
    input_ids, _ = backend.render(messages)
    reserve = settings.observation_tokens + 64  # tool/template delimiters and feedback overhead
    if len(input_ids) + settings.max_new_tokens + reserve > settings.context_tokens:
        raise ValueError("Initial public prompt cannot fit context and required reserves")
    environment.sandbox.preflight()
    backend.start()
    policy = {
        **backend.identity,
        "sampling": {
            "do_sample": settings.do_sample,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        },
        "seed": seed,
    }
    environment.recorder.policy = policy
    environment.reset(task_id)
    started = time.monotonic()
    deadline = started + config.environment.episode_seconds
    turns = tokens = proposed = accepted = edits = errors = consecutive = rejected = 0
    final = ""
    reason = "turn_limit"
    infrastructure = None
    reward = None
    terminated = False
    timings = {
        "generation_seconds": 0.0,
        "tool_seconds": 0.0,
        "grading_seconds": 0.0,
        "cleanup_seconds": 0.0,
        "peak_rss_bytes": 0,
        "first_token_seconds": None,
        "decode_tokens_per_second": None,
    }
    recorder = environment.recorder
    try:
        recorder.artifact("public-task", asdict(public))
        for turn in range(settings.max_turns):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "episode_timeout"
                break
            if proposed >= config.environment.max_tool_calls:
                reason = "tool_limit"
                break
            available = settings.episode_tokens - tokens
            if available <= 0:
                reason = "token_limit"
                break
            input_ids, prefix = backend.render(messages)
            max_new = min(settings.max_new_tokens, available)
            if len(input_ids) + max_new + reserve > settings.context_tokens:
                reason = "context_limit"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "episode_timeout"
                break
            turns += 1
            generation_started = time.monotonic()
            try:
                generated = backend.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    seed=derive_seed(config.seed, task_id, attempt, turn + 1),
                    timeout=min(settings.generation_seconds, remaining),
                    sampling=settings,
                )
            except GenerationTimeout:
                backend.close()
                timings["generation_seconds"] += time.monotonic() - generation_started
                reason = "episode_timeout" if time.monotonic() >= deadline else "generation_timeout"
                recorder.emit("generation", turn=turn, stop_reason=reason, completed=False)
                break
            timings["generation_seconds"] += time.monotonic() - generation_started
            tokens += len(generated.token_ids)
            timings["peak_rss_bytes"] = max(timings["peak_rss_bytes"], generated.peak_rss_bytes)
            recorder.emit(
                "generation",
                turn=turn,
                input_tokens=len(input_ids),
                generation_prefix=prefix,
                seed=derive_seed(config.seed, task_id, attempt, turn + 1),
                **asdict(generated),
            )
            # A cutoff turn never dispatches, even if it contains a complete patch.
            if time.monotonic() >= deadline:
                reason = "episode_timeout"
                break
            if generated.stop_reason != "eos":
                reason = "token_limit"
                break
            if tokens > settings.episode_tokens or len(generated.token_ids) > max_new:
                raise InfrastructureError("Backend exceeded the declared generation budget")
            try:
                parsed = parse_response(generated.raw_text, generation_prefix=prefix)
            except ProtocolError as exc:
                errors += 1
                consecutive += 1
                charged = min(exc.proposals, config.environment.max_tool_calls - proposed)
                proposed += charged
                rejected += charged
                recorder.emit("parse", valid=False, diagnostic=str(exc), proposed_calls=proposed)
                feedback = bounded_observation(
                    {"ok": False, "error": "protocol_error", "message": str(exc)},
                    backend,
                    settings.observation_tokens,
                )
                # No trustworthy tool envelope exists. Preserve the raw assistant
                # text and use an ordinary user feedback message in this template.
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": generated.raw_text.removesuffix("<|im_end|>"),
                        },
                        {"role": "user", "content": "Harness feedback: " + feedback},
                    ]
                )
                recorder.emit(
                    "model_observation", role="user", content="Harness feedback: " + feedback
                )
                if consecutive >= settings.parser_recovery_limit:
                    reason = "parser_limit"
                    break
                continue
            consecutive = 0
            recorder.emit("parse", valid=True, parsed=asdict(parsed))
            if parsed.call is None:
                final, reason = parsed.content, "final_response"
                messages.append(parsed.message())
                break
            proposed += 1
            assistant = parsed.message()
            # Prove space for this action plus a full observation BEFORE mutation.
            pending = [
                *messages,
                assistant,
                {"role": "tool", "name": parsed.call.name, "content": ""},
            ]
            pending_ids, _ = backend.render(pending)
            if len(pending_ids) + reserve > settings.context_tokens:
                reason = "context_limit"
                break
            before_calls = environment.calls
            tool_started = time.monotonic()
            messages.append(assistant)
            result = dispatch(environment, parsed.call)
            timings["tool_seconds"] += time.monotonic() - tool_started
            entered = environment.calls > before_calls
            if result["ok"]:
                accepted += 1
                if parsed.call.name == "apply_patch" and result.get("changed_paths"):
                    edits += 1
            else:
                rejected += 1
            observation = bounded_observation(result, backend, settings.observation_tokens)
            messages.append({"role": "tool", "name": parsed.call.name, "content": observation})
            recorder.emit(
                "dispatch",
                tool=parsed.call.name,
                entered_core=entered,
                accepted=result["ok"],
                proposed_calls=proposed,
                core_calls=environment.calls,
                accepted_calls=accepted,
                accepted_edits=edits,
            )
            recorder.emit(
                "model_observation", role="tool", name=parsed.call.name, content=observation
            )
            if environment.phase != "active":
                reason = environment.inspect()["termination_reason"]
                break
        if proposed >= config.environment.max_tool_calls and reason == "turn_limit":
            reason = "tool_limit"
        elif tokens >= settings.episode_tokens and reason == "turn_limit":
            reason = "token_limit"
        terminated = True
        recorder.emit(
            "termination",
            reason=reason,
            turns=turns,
            generated_tokens=tokens,
            proposed_calls=proposed,
            final_response=final,
        )
        environment.finish(reason)
        grading_started = time.monotonic()
        try:
            reward = environment.get_reward()
        finally:
            timings["grading_seconds"] = (
                time.monotonic() - grading_started - environment.cleanup_seconds
            )
    except KeyboardInterrupt:
        reason, infrastructure = "cancelled", "Episode cancelled"
        backend.close()
    except (InfrastructureError, OSError, RuntimeError, ValueError) as exc:
        if not terminated:
            reason = "infrastructure_failure"
        infrastructure = f"{type(exc).__name__}: {exc}"
        backend.close()
    finally:
        try:
            if environment.phase == "active":
                environment.finish(reason)
        except (InfrastructureError, OSError) as exc:
            infrastructure = f"Cannot freeze submission: {exc}"
            reward = None
        finally:
            try:
                environment.cleanup()
            except (InfrastructureError, OSError) as exc:
                infrastructure = f"Cleanup incomplete: {exc}"
                reward = None
        timings["cleanup_seconds"] = environment.cleanup_seconds
    state = environment.inspect()
    verdict = state["verdict"]
    completed = infrastructure is None and reward is not None and environment.workspace is None
    if not completed:
        reward = None
    timings["episode_seconds"] = time.monotonic() - started
    timings["generation_tokens_per_second"] = (
        tokens / timings["generation_seconds"] if timings["generation_seconds"] else None
    )
    outcome = ModelOutcome(
        task_id,
        environment.episode_id,
        policy,
        seed,
        reason,
        completed,
        reward,
        verdict,
        public.starter_hash,
        state["source_hash"],
        grading_fingerprint(environment.catalog.get(task_id)),
        proposed,
        environment.calls,
        accepted,
        edits,
        turns,
        tokens,
        errors,
        rejected,
        environment.workspace is None,
        str(recorder.root),
        final,
        infrastructure,
        timings,
        "model" if policy["kind"] == "model" else "test",
    )
    recorder.artifact("conversation", messages)
    recorder.artifact("outcome", asdict(outcome))
    recorder.emit("outcome", **asdict(outcome))
    return outcome
