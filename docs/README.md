# Minimal coding-agent RL lab: implementation drafts

These documents turn [the build specification](../CODEX_AGENTIC_RL_ENV_SPEC.md) into a small, staged Python implementation plan. Stage A and the Phase 2 CPU rollout/evaluation implementation are present: see [the run guide](../README.md), [Stage A status](PHASE_A_STATUS.md), and [Phase 2 validation status](PHASE_2_STATUS.md). The original architecture drafts explain the design; training remains a future stage.

The [Phase 1 evaluation](PHASE_1_EVALUATION.md) identified a cleanup/cache edge case, now fixed with a regression test. The [Phase 2 rollout design](PHASE_2_MODEL_ROLLOUT_DESIGN.md) remains the original design record; [Phase 2 status](PHASE_2_STATUS.md) distinguishes implemented behavior from executed model/Docker gates.

The next implementation work is specified in [Phase 3: native TRL training](PHASE_3_NATIVE_TRAINING_DESIGN.md). First close the real Docker episode and declared pre-training baseline, then build the native adapter, complete-group/tensor admission, genuine update, checkpoint reload/evaluation, and separate full/LoRA proofs. [The step-by-step roadmap](06_STEP_BY_STEP_IMPLEMENTATION.md) lists this work in coding order. The new design changes documentation only; it adds no training command or dependency.

The central idea is simple: a model uses tools to change a private copy of a tiny repository; independent executable tests judge the final changes; TRL uses rewards from several attempts at the same task to update the model.

## Read in this order

| Document | Question it answers |
|---|---|
| [PHASE_2_STATUS.md](PHASE_2_STATUS.md) | What Phase 2 code exists, how do I run it, and which acceptance gates have actually run? |
| [PHASE_3_NATIVE_TRAINING_DESIGN.md](PHASE_3_NATIVE_TRAINING_DESIGN.md) | What remains before native training, which files change next, and what proves an actual policy update? |
| [PHASE_1_EVALUATION.md](PHASE_1_EVALUATION.md) | What did the current code and rerun tests establish, and what needs attention before a model is added? |
| [PHASE_2_MODEL_ROLLOUT_DESIGN.md](PHASE_2_MODEL_ROLLOUT_DESIGN.md) | What is the concrete CPU model flow, dependency plan, protocol, and ordered coding checklist? |
| [PHASE_A_STATUS.md](PHASE_A_STATUS.md) | What works today, which commands were tested, and what changed from the draft? |
| [ARCHITECTURE.md](ARCHITECTURE.md) | What is the environment, what changes during an episode, and how does learning connect to it? |
| [ENVIRONMENT_SUPPORT.md](ENVIRONMENT_SUPPORT.md) | How does the core environment connect to native TRL without duplicating its agent loop? |
| [03_ENVIRONMENT_SANDBOX_AND_RLVR.md](03_ENVIRONMENT_SANDBOX_AND_RLVR.md) | How do reset, file tools, isolation, private grading, and binary rewards work? |
| [04_TRAINING_AND_CPU_BACKENDS.md](04_TRAINING_AND_CPU_BACKENDS.md) | What must be true for GRPO, CPU inference, Qwen3.5, LoRA, and optional vLLM to work correctly? |
| [05_HARNESS_ROLLOUTS_AND_EVALUATION.md](05_HARNESS_ROLLOUTS_AND_EVALUATION.md) | What does a built-in or custom harness control, and how do traces and evaluation work? |
| [06_STEP_BY_STEP_IMPLEMENTATION.md](06_STEP_BY_STEP_IMPLEMENTATION.md) | In what order should files and features be implemented, and what proves each stage works? |

Start with the environment diagram and worked episode in `ARCHITECTURE.md`. Read the trainer details after you can explain why two attempts need separate working copies.

## Decisions for the first implementation

| Concern | Starting decision |
|---|---|
| Target model | Preserve `Qwen/Qwen3.5-0.8B` from the spec; allow explicit overrides |
| Environment | One small Python core with an immutable task starter and a fresh workspace per episode |
| Agent tools | `list_files`, `read_file`, `apply_patch`, `run_tests` |
| Execution | Docker workers for generated code; an explicit trusted-fixtures-only local option |
| Grading | Frozen allowed-file changes applied to a separate pristine starter; trusted assertions judge a separate execution worker |
| Reward | One terminal correctness reward: 1.0 or 0.0; infrastructure failure has no reward |
| Training | Native TRL `GRPOTrainer` with a thin `environment_factory` adapter and external task IDs |
| CPU path | Environment demo first, then manual Transformers inference/evaluation; real training is a separate mode |
| Optional backends | CPU vLLM and custom harness start as rollout/evaluation only |
| Operations | Validated YAML, standard logging, local JSONL artifacts, and a small CLI |

The official TRL documentation supports the planned native interface. The existing CPU profile has its own measured inference evidence; native TRL remains uninstalled/unverified in this project's recorded validation. See [environment support](ENVIRONMENT_SUPPORT.md) for the adapter contract and [Phase 3 gates](PHASE_3_NATIVE_TRAINING_DESIGN.md) for installed API, failure, tensor, and optimizer requirements.

## CLI status and outputs

The executable is `coding-rl`. Each command accepts the same `--config` option. `env-demo`, `env-inspect`, and `smoke` use the lightweight base profile. `rollout`, `eval`, and the ungraded `model-probe` use the optional rollout extra and `configs/rollout-cpu.yaml`. `train` remains a future interface.

| Command | Purpose | Completion evidence |
|---|---|---|
| `env-demo` | Show a scripted reset, file edit, observation, private verification, reward, and cleanup | Two attempts visibly start from the same clean state |
| `env-inspect` | Show developer state, tool schemas, limits, and current/frozen diff | Debug data stays outside model messages |
| `smoke` | Run deterministic environment and isolation checks without model weights | Successful and unsuccessful known cases plus cleanup checks |
| `model-probe` | Ungraded CPU compatibility diagnostic; source-read dispatch only | Recorded target generation, parser and public read; no correctness score |
| `rollout` | Let a real model attempt tasks without updating weights | Tool trace, final artifact, verdict, and policy identity |
| `train` | Perform native GRPO updates and save a checkpoint | Genuine model-generated nonzero-variance group, changed trainable weights, reloadable checkpoint |
| `eval` | Judge a checkpoint on held-out tasks | JSON metrics with valid denominator and separate infrastructure errors |

Runtime artifacts belong under `outputs/`, outside fixture repositories. The planned training run adds resolved native configuration, dependency/model revisions, group/tensor evidence, update metrics, checkpoint manifests, and comparable evaluation reports. Private assertions and expected values stay outside model observations and ordinary trajectories.

## Evidence status

Original source review date: **2026-10-14**. [Stage A status](PHASE_A_STATUS.md) records its original Docker evidence. The **2026-10-15** [Phase 2 status](PHASE_2_STATUS.md) records the CPU model forward/generation probes, controller tests, calibrated profile, and remaining Docker/model evaluation gates. The [Phase 3 design](PHASE_3_NATIVE_TRAINING_DESIGN.md) reviews current native documentation/source and defines future acceptance; it performs no model or training run. No optimizer step has been performed.
