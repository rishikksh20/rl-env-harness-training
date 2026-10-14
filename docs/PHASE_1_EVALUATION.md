# Phase 1 evaluation and readiness for Phase 2

Review date: **2026-10-14**. Phase 1 means Stage A of the original plan: the model-free environment. This review changes documentation only. Existing tests and a temporary fault-injection diagnostic were executed; runtime code, project dependencies, and task fixtures were not edited.

## 1. Assessment

Phase 1 provides a working educational environment with real isolated execution and independent binary verification. It is a sound foundation for adding a model. It does not yet provide a complete model rollout contract, and one confirmed cleanup/cache defect should be fixed at the start of Phase 2.

The distinction matters: passing the existing tests establishes the behaviors those tests exercise. It does not establish that every failure path is safe or that a real model can generate valid actions. The Phase 2 design is in [PHASE_2_MODEL_ROLLOUT_DESIGN.md](PHASE_2_MODEL_ROLLOUT_DESIGN.md).

## 2. Evidence gathered in this review

| Check | Result | What it establishes |
|---|---|---|
| Existing complete suite, with real Docker enabled | **68 passed**, including eight Docker tests | Current environment, verifier, bounds, and tested containment behaviors |
| Ruff lint | Passed | Current configured static checks |
| Ruff formatting check | Passed | Current formatting policy |
| Offline uv lock check | Passed | `pyproject.toml` and `uv.lock` agree |
| Cleanup failure followed by repeated reward request | Confirmed defect | Cached reward can bypass incomplete cleanup |
| Runtime source inspection | Completed | Concrete integration changes required for model rollouts |

The test invocation enabled `CODING_RL_DOCKER_TESTS` and used locked offline dependencies. Its completed result was 68 passed in 33.86 seconds. This time describes this review run, not a promised performance figure.

Previous wheel/build/demo results remain documented in [PHASE_A_STATUS.md](PHASE_A_STATUS.md). Those packaging and CLI results were not all rerun for this review. No model was downloaded or loaded, and no optimizer step was performed.

## 3. What works well

| Area | Current implementation | Evaluation |
|---|---|---|
| Task identity | Authored public starters with file hashes and train/evaluation partitions | Reproducible tiny problems |
| State lifecycle | Fresh episode workspace, explicit phases, frozen submissions | State persists within an attempt and resets across attempts |
| Tools | Four named methods with bounds and schema descriptions | Small, understandable action surface |
| Edits | Restricted exact-context diffs; complete validation before committing a new revision | Invalid patches do not partially edit files |
| Execution | Non-root, no-network Docker workers, read-only snapshots, resource/output/time caps | Meaningful tested containment for the local teaching profile |
| Reward | Fresh candidate reconstruction and protected controller assertions | Visible test output and assistant claims cannot declare correctness |
| Local option | Only exact authored fixture/reference snapshots may execute locally | Explicit teaching path with no arbitrary generated-code fallback |
| Group isolation | Four separate same-task states tested with real rewards | Useful invariant for later GRPO integration |
| Packaging | Project metadata, entry point, dependency group, lock, and bundled fixtures | Appropriate uv foundation without a training stack |

Keep these decisions. Phase 2 should add decision-making around the existing world, rather than rebuilding the environment.

The JSON-callable verifier is deliberately narrower than arbitrary repository grading. Its private assertion controller remains outside submitted Python's interpreter. The candidate can still inspect its own worker and current inputs or attempt to spoof values. The existing tests demonstrate specific behaviors; they do not establish protection against every possible attack or broad coding competence.

## 4. Confirmed defect: cached verdict bypasses unfinished cleanup

Location: [environment.py](../src/coding_rl/environment.py), `get_reward` and `cleanup`.

The method stores a completed verdict and emits a reward before its final cleanup path. If cleanup then raises an infrastructure exception, the workspace can remain owned and live. A later call sees the cached verdict and returns immediately, without retrying cleanup.

The temporary diagnostic used an authored correct zero-timeout repair, then simulated a cleanup exception. After restoring the real cleanup method, a second reward call produced:

| Observed field | Value |
|---|---|
| Cached reward | 1.0 |
| Workspace still live | True |
| Phase | Judged |

The diagnostic explicitly cleaned its temporary workspace afterward. It did not change the project implementation.

This does not make the verified patch incorrect. It makes episode completion and reward delivery inconsistent with the promised resource lifecycle. A future evaluator could report a completed attempt while its resources remain live.

**Required design correction:** retain the trusted verdict, separately track completion/cleanup status, and retry pending cleanup before returning a completed episode result. A failed cleanup must not turn into reward zero. A repeated request must not rerun private tests or emit duplicate reward events. Reporting must distinguish known correctness from an infrastructure-incomplete attempt.

**Required future regression check:** fail cleanup after successful verification; confirm no normally completed result while resources remain live; restore cleanup; confirm one stable verdict, successful release, and one usable reward event published after successful cleanup.

## 5. Necessary integration changes

These are planned extensions, not failures of the original scripted scope.

| Priority | Current limitation | Change before real rollout acceptance |
|---|---|---|
| First | `harness.run_episode` selects a reference variant and constructs its patch | Keep scripted lessons separate; add a genuine model episode path that never reads reference repairs |
| First | Recorder labels every event as scripted | Accept actual policy/checkpoint identity and generation events before recording model runs |
| First | Core counts entered tool methods, not unknown/malformed proposals rejected before entry | Add a bounded dispatcher counter and typed argument validation; distinguish proposals, core invocations, and accepted effects |
| First | Configuration has no model, sampling, or token limits | Add optional model/rollout/evaluation records with strict mode-aware validation |
| First | Finish reasons cover only current scripted/core stops | Add precise turn/token/context/parser/generation reasons and preserve correctness separately |
| First | Environment deadline is checked at tool entry | Supervise inference itself; a long generation must not outlive the episode deadline |
| First | Tool schemas are a named inspection mapping | Translate the whitelist into the actual chat-template function schema format |
| Before baseline comparison | Verdict metadata has a constant verifier label but no digest of the private case plan | Record a private-safe grading-plan fingerprint and the resolved run/model/template identities |
| Before baseline comparison | Summary is tailored to authored variants | Introduce model outcome data with termination, parser failures, timings, tokens, and nullable infrastructure outcome |

The model input must contain only public task information and the four tool schemas. `Task` also holds protected cases and reference variants. Serializing the whole task record into a prompt would leak grading material; explicitly whitelist public fields.

## 6. Failure-path improvements to include

Trusted filesystem operations such as revision creation/removal can raise ordinary `OSError`, while the tool wrapper explicitly handles only action and infrastructure exceptions. Phase 2's owner should classify these as infrastructure failures and record whether the authoritative revision committed. They must not look like a model's malformed call or disappear into an unreported abort. This is a source-based observation; this review did not reproduce a real disk failure.

Also cover recorder-write failures, cancellation during generation, a crashed inference worker, and cleanup errors after a verdict exists. Separate correctness, generation stop reason, and run completion status. Preserve the last accepted source snapshot for diagnostics when possible.

A run with unknown correctness has no reward. A run with known correctness but failed cleanup retains its verdict and remains incomplete until the resource problem is resolved. Healthy limit-caused attempts can be verified and reported with their actual limit reason.

## 7. Scope and evidence limits

Four function-level fixtures and one held-out family are enough to demonstrate plumbing. They are insufficient for a generalization claim. The reference-driven smoke result is not a model pass rate, and the four-state test is not a GRPO update.

CPU model loading, Qwen3.5 tool parsing, chat-template behavior, optional inference dependencies, generation timing, and inference memory remain untested. Docker workers execute candidate code; they do not provide model inference. The Phase 1 wheel's small dependency set should remain usable without installing model packages.

Python 3.12 is the executed profile. Python 3.11 remains a declared minimum, not a newly verified profile. Additional operating systems or CPU architectures require their own dependency and Docker-limit checks.

## 8. Readiness decision

Proceed to Phase 2 with a short prerequisite milestone: fix cleanup/cache completion, establish truthful traces and outcome records, and define validated dispatch and stopping behavior. Then resolve the optional CPU model stack, prove actual generated tools, and produce a held-out baseline.

The environment, tools, Docker execution, task starters, and protected binary reward design should be reused. Their first expansion should be in the controller/harness boundary, not in a larger tool surface or a new RL framework.
