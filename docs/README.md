# Minimal coding-agent RL lab: implementation drafts

These documents turn [the build specification](../CODEX_AGENTIC_RL_ENV_SPEC.md) into a small, staged Python implementation plan. **Stage A is now implemented**: see [the run guide](../README.md) and [Stage A implementation status](PHASE_A_STATUS.md). The original architecture drafts explain the design; model inference and training remain future stages.

The central idea is simple: a model uses tools to change a private copy of a tiny repository; independent executable tests judge the final changes; TRL uses rewards from several attempts at the same task to update the model.

## Read in this order

| Document | Question it answers |
|---|---|
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

The existing official TRL documentation supports the planned environment integration. This is documentation-level evidence, not a tested dependency lock. The implementation must record installed versions and pass the compatibility gates before claiming working model or training support. See the evidence table in [ENVIRONMENT_SUPPORT.md](ENVIRONMENT_SUPPORT.md).

## Planned CLI and outputs

The executable is `coding-rl`. Each command accepts the same `--config` option, pointing to `configs/default.yaml` when explicitly selected. `env-demo`, `env-inspect`, and `smoke` are implemented in Stage A. `rollout`, `train`, and `eval` remain future interfaces.

| Command | Purpose | Completion evidence |
|---|---|---|
| `env-demo` | Show a scripted reset, file edit, observation, private verification, reward, and cleanup | Two attempts visibly start from the same clean state |
| `env-inspect` | Show developer state, tool schemas, limits, and current/frozen diff | Debug data stays outside model messages |
| `smoke` | Run deterministic environment and isolation checks without model weights | Successful and unsuccessful known cases plus cleanup checks |
| `rollout` | Let a real model attempt tasks without updating weights | Tool trace, final artifact, verdict, and policy identity |
| `train` | Perform native GRPO updates and save a checkpoint | Genuine model-generated nonzero-variance group, changed trainable weights, reloadable checkpoint |
| `eval` | Judge a checkpoint on held-out tasks | JSON metrics with valid denominator and separate infrastructure errors |

Future runtime artifacts belong under `outputs/`, outside fixture repositories. A training run saves its resolved configuration, dependency/model revisions, metrics, trajectories, and checkpoint together. Private assertions and expected values are not copied into agent observations or ordinary trajectories.

## Evidence status

Source review date: **2026-10-14**. Official library and model documentation were reviewed for the design. Stage A subsequently installed locked dependencies and tested real Docker execution; [PHASE_A_STATUS.md](PHASE_A_STATUS.md) records the evidence. No model forward pass or optimizer step has been performed. Model/token budgets remain proposals for later compatibility gates.
