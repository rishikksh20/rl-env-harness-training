# Codex Build Specification: Minimal Coding-Agent RL Lab (GRPO + RLVR) — v2

> **Primary learning goal:** Understand the environment itself: its state, reset lifecycle, tools/actions, observations, termination, verifier, rewards, and how TRL connects it to a trainable LLM. **Implement these concepts before optimizing training speed.**

## Goal

Build a **small, beginner-friendly, runnable Python repository** that demonstrates end-to-end **agentic reinforcement learning** for a coding assistant. Use **Qwen/Qwen3.5-0.8B** as the target model, **GRPO** for policy training, and **RLVR** (reinforcement learning with verifiable rewards) using real tests in a sandbox.

This is an **educational project, not a production platform**. Prioritize short, readable code and correct learning mechanics over scale, frameworks, dashboards, or abstractions. Use standard libraries for GRPO and build only the environment-specific parts from scratch.

## Architecture decision: what “environment support” means (read this before coding)

An RL **environment** is a stateful, isolated world in which an agent acts; it is **not** the model, inference server, agent harness, or RL optimizer. For this project the world is a Git-style working copy of a tiny Python repo. Reading and editing files changes the state the agent can later observe.

| Component | Owns | Does NOT own |
|---|---|---|
| Task dataset | Stable task ID, initial commit/files, user-visible issue | Runtime edits, private grading files in the prompt |
| Coding environment | Per-episode state; resetting a task; tool behavior; per-step observations; episode status | Policy gradients or model weights |
| Sandbox | Filesystem/process isolation, command output, timeout, resource limits | Reward formula or policy |
| Harness | Prompt/tools schema, response→tool-call plumbing, turn/budget management | Source of truth for hidden-test correctness |
| Verifier | Independent evaluation against trusted hidden regression tests | Agent-visible tools/observations |
| TRL GRPOTrainer | Model sampling, multi-turn rollout loop for supported environments, token-level masks/logprobs, advantages, optimizer steps | How `apply_patch()` edits the repository |

For a newcomer, the central mental model is `observation -> LLM-generated tool action -> sandbox state transition -> new observation`, repeated until termination. Keep **environment state** (actual files, process state) distinct from **observation** (what the LLM is shown). The whole interaction is an episode/rollout; the recorded events form its trajectory.

### Two layers, not one giant agent class

1. **Core `CodingEnvironment` (framework-independent):** supports `reset(task_id)`, safe tool execution, `get_reward()`, cleanup, and inspection/debugging of a single episode. Its internal `step` concept may be useful for teaching, but do not expose a generic `step(action)` to the model in the primary tool-calling interface.
2. **TRL environment adapter (thin):** supplies a no-argument factory to TRL's `environment_factory` and exposes `reset(...)`, `get_reward()`, and individually named tool methods (`list_files`, `read_file`, `apply_patch`, `run_tests`). Do not duplicate TRL's internal generate→parse tool call→feed observation loop in training mode. The adapter must not accidentally expose debugging, verifier, cleanup, or private grader methods as model-callable public tools.

**Choose this as the default integration:** Hugging Face TRL's stateful `environment_factory`, with an **external task dataset** whose rows contain a user-facing `prompt` plus an opaque `task_id`. For each completion, TRL associates that task row with an environment and passes task metadata to `reset(**kwargs)`. Each GRPO group gets identical task IDs and clean initial repository states. Prefer `reset()` returning `None` in dataset-prompt mode so the user message is not duplicated. The trainer can also support an **environment-owned data** mode in which `reset()` selects a task and returns a prompt, but label it optional; ensure GRPO group members share the same task, since random per-rollout tasks make group comparisons noisy.

TRL APIs change. In the version actually installed, inspect official docs and the `GRPOTrainer` signature, write a tiny compatibility test, pin the working versions, and document exact behavior. At the time this spec was written, current TRL docs describe `environment_factory`, reserved `reset/get_reward`, and auto-exposure of public environment tool methods; `transformers>=5.2.0` is indicated for this integration. If version semantics differ, prefer the verified supported contract rather than silently faking integration.

### Annotated conceptual sketch (educational, NOT full implementation)

```python
# Example structure only: Codex must implement safety, lifecycle, typing, and docs.
class TRLCodingEnv:
    def __init__(self):  # or use a zero-argument config-capturing factory
        self.core = make_core_environment()

    def reset(self, task_id: str, **kwargs) -> None:
        # TRL passes task_id from the dataset row; prompt already lives in dataset.
        self.core.reset(task_id)  # isolated repo at the exact starting revision

    def read_file(self, path: str) -> str:
        """Read a file from this episode's repository.

        Args:
            path: Safe relative file path.
        """
        return self.core.read_file(path)

    def apply_patch(self, patch: str) -> str:
        """Modify this episode's working tree.

        Args:
            patch: Unified diff patch to apply.
        """
        return self.core.apply_patch(patch)

    def run_tests(self) -> str:
        """Run agent-visible tests in the isolated sandbox."""
        return self.core.run_visible_tests()

    def get_reward(self) -> float:
        return self.core.evaluate_final_patch_with_private_verifier()
```

The real implementation must also supply `list_files` with a clear typed schema, enforce budgets, and ensure reward is computed on a separate trusted checkout. The adapter's only tool-visible methods should be the allowed tools. Every callable exposed as a tool should have annotations and useful Google-style argument docstrings. `get_reward()` is called after the rollout and must be deterministic/idempotent for a fixed final patch; caching the result is sensible.

**Do not confuse:** `run_tests()` returns a visible observation for the agent. `get_reward()` runs protected grading after the episode. Passing visible tests is NOT automatically success.

### Explicit lifecycle to implement and demonstrate

```text
Task dataset row: {prompt, task_id}
  |
  v
TRL creates/assigns an environment for each rollout
  |
  v
reset(task_id) -> private sandbox: checkout pristine starter repo
  |
  v
LLM chooses read_file(path) ---> tool observation (source text)
  |
  v
LLM chooses apply_patch(diff) --> environment updates only THIS workspace
  |
  v
LLM chooses run_tests() -----> observation (visible pytest output)
  |
  +---- next LLM generation / another tool action (same workspace) ----+
  |                                                                  |
  +------------------------------------------------------------------+
  |
  v
Agent finishes OR max turns/tokens/time -> collect final diff/status
  |
  v
get_reward(): apply diff to independent clean checkout + run hidden tests
  |
  v
reward 0/1 + trace -> TRL computes group advantages and updates policy
  |
  v
cleanup sandbox; next rollout starts clean
```

Terminology to cover with a concrete example in the docs:
- **State:** files and edits (private truth), plus runtime/session state.
- **Observation:** file contents, errors, stdout/stderr presented to model.
- **Action:** model-generated tool call + arguments.
- **Episode:** one independent attempt on one task.
- **Trajectory:** ordered model generations + tool invocations/results + final patch + reward.
- **Termination:** final response, tool/turn/token limit, explicit submission (optional), or timeout.
- **Reward:** trusted post-rollout correctness signal; never exposed as a normal tool result.

### Minimal environment lesson BEFORE any RL training

Add `env-demo` CLI command that requires **no GPU and no model download** and walks through an educational scripted episode. It must print a readable trace, such as:

```text
[reset] task=zero-timeout starting_commit=abc123
[action] read_file('timeout.py')
[observation] ...
[action] apply_patch('...')
[observation] patch applied
[action] run_tests()
[observation] visible tests: 3 passed
[finish] final diff captured
[verify] private tests: 4 passed
[reward] 1.0
[cleanup] workspace removed
```

Run that exact task twice with separate episode IDs and prove no edits leaked. Add `env-inspect` (or a `--show-state` flag) to display tool schemas, current step count, allowed paths, budget remaining, and state/diff **without exposing hidden tests to the model**. The purpose is education, not a web dashboard.

### Rollout handling modes — be explicit

| Mode | Generation and tool loop | Training correctness | First release? |
|---|---|---|---|
| Mock env-demo | Deterministic scripted harness | No policy update | **Required first** |
| Native TRL environment | TRL agent/tool loop via `environment_factory` | Supported path for actual GRPO if model/template work | **Required** |
| Manual Transformers rollout | Our minimal Python harness | Rollout/eval only unless fully integrated with trainer's token/logprob contract | **Required for CPU learning demo** |
| Custom harness | External coding agent controls decisions | Initially rollout/eval only | Optional adapter |
| OpenEnv remote environment | Client/server env behind a thin TRL adapter | Possible, but extra infrastructure | Optional tutorial AFTER local MVP |
| CPU vLLM | Separate inference service | Rollout/eval only until verified correct sync + loss alignment | Optional |

OpenEnv is a useful **reference and possible adapter** for the standardized `reset/step/state` abstraction; it is **not required** for the minimal local first version. Don't implement OpenEnv, Gymnasium, and TRL adapters simultaneously: build one clean core first.

### GRPO task grouping and reward propagation

- Dataset tasks should be stable and reproducible. `task_id` selects an immutable starter checkout and private verification plan; the dataset itself contains no hidden test text.
- With `num_generations=4`, TRL needs **four separate environment states** for four attempts on the same task. Each attempt can search, edit, and fail differently, but must start from the same clean commit. Write an integration test for this invariant.
- Reward is emitted **once per completed rollout**, after the final patch is evaluated. With rewards `[1, 0, 1, 0]`, group-relative advantages have positive and negative signs; with `[0, 0, 0, 0]` there is no within-group reward variance to learn from in standard normalized GRPO.
- Tool output tokens condition subsequent model actions but are **masked out of policy loss**. TRL must retain the model-generated token sequence and required policy probabilities; a JSONL human-readable trace alone is not sufficient.
- Record the reason when a run is truncated by a context/token limit, turns, wall-time, or invalid tool calls. A termination event must not silently be counted as solved.
- Train/eval partitions must prevent contamination; the verifier must have its own no-network, non-root, resource-limited sandbox for untrusted code.

### CPU and Qwen3.5 compatibility gates

- **CPU smoke** = mock environment, not inference. **CPU rollout** = real model generates actions but does not update weights. **CPU full GRPO training** = real optimizer updates, potentially prohibitively slow for 0.8B; do not advertise it as quick.
- Qwen3.5-0.8B is a **post-trained, multimodal-capable model** with hybrid Gated DeltaNet layers. Prefer **text-only** tool-call interactions, load its text-only generation path if supported by pinned Transformers and the model checkpoint, and explicitly test tool-call parsing. Provide a known-compatible small fallback model for validating TRL when the primary model or hardware is unavailable.
- Gated DeltaNet/linear attention introduces **kernel and packed-sequence boundary correctness concerns**, not just performance concerns. Verify that the chosen TRL/Transformers combination produces sound training forward passes and that masks/sequences do not contaminate each other; disable incompatible packing rather than training with silent sequence leakage. Inspect authoritative library documentation; note any remaining limitation.
- Try standard Transformers CPU inference first. vLLM CPU is not a requirement; enable it only if Qwen3.5, tool calling, weight loading, and the installed CPU backend are tested. Keeping vLLM CPU in *rollout/eval-only* mode by default is fine.
- Support `device` per **mode**, not pretend that selecting a CPU rollout backend inside a GPU GRPOTrainer automatically enables a mixed-device training execution path.

### Required documentation: `docs/ENVIRONMENT_SUPPORT.md`

Besides README and ARCHITECTURE.md, write a beginner-oriented explanation covering: `environment_factory` vs stateless `tools` vs manual `rollout_func`; dataset-driven vs environment-owned `reset`; public tool discovery and tool-call schema; one sandbox per rollout; state vs observation; how agent-visible tests differ from hidden verification; where rewards go; when TRL creates/returns trainable tokens; lifecycle sequence chart; common failure modes; and an explicit, runnable env-demo transcript. Cite the exact tested TRL version and official docs URLs.

## Preferred technology choices

- Python 3.11+, `pyproject.toml`, `src/` package layout, type hints, minimal dependencies.
- **Hugging Face TRL `GRPOTrainer`** + Transformers + Datasets + PyTorch for real training. Use the tested stateful **`environment_factory`** contract, an external task dataset, and named tool methods. Verify the pinned APIs and required Transformers version before implementation. Do **not** reimplement GRPO loss, advantages, token masking, or optimizer.
- Default rollout backend: **Transformers on CPU or CUDA**. Optional `vllm` backend for inference / rollout testing on CPU or CUDA, only after checking that *this model and hardware* work. **Do not assume a CPU vLLM server can automatically synchronize weights with a GPU TRL trainer.** If live sync and correct per-token behavior are not supported, mark CPU-vLLM **rollout/eval-only** and use the compatible Transformers path for true on-policy training. Document the limitation.
- Default model: **`Qwen/Qwen3.5-0.8B`** (instruct/post-trained). Allow model override, including a smaller model for fast CPU smoke tests. Verify model-loading, text/tool-call chat templates, architecture dependencies, and GRPO compatibility before training.
- Standard Python `logging`, JSONL event traces, YAML config, pytest tests. No mandatory database, message queue, distributed cluster, UI, or SaaS service.

## Required end-to-end flow

`Task dataset -> isolated repository -> harness/model -> tool calls -> observations -> final patch -> independent verifier -> reward -> GRPO update -> evaluation`

Explain the distinction between a **task**, **episode/rollout**, **trajectory**, **observation**, **model action**, **terminal reward**, and **RL batch** in docs. A rollout must be **stateful within one attempt**, and a new attempt must get a **fresh independent repository**. For GRPO, multiple rollouts of the same task must start from **identical task state**, with no shared edits or tests.

## Smallest sensible architecture

Implement these concepts with simple classes or protocols, avoiding heavy design patterns:

1. **Task**: ID, instruction, fixture repo + revision, visible tests, private verifier tests, budgets.
2. **CodingEnvironment**: core stateful logic with `reset(task_id)`, safe tool methods, observation/state inspection, terminal `get_reward()`, and cleanup. Add a thin adapter for TRL's `environment_factory`, keeping only intended tool methods public.
3. **Sandbox**: creates an isolated temporary workspace and safely executes allowed actions. Use Docker for executing untrusted agent-generated code; disable network access, run as non-root, set CPU/memory/time limits, never mount secrets or host directories with write permissions. An optional local subprocess backend must be explicitly labeled **trusted-fixtures-only / unsafe for untrusted code** and never become the default for untrusted execution.
4. **Harness interface**: choose `minimal` (built-in agent using TRL's supported tool calling) or `custom` (a small adapter interface for an existing agent harness). Keep harness concerns separate from environment and verification. Clearly state that external harnesses are **eval-only** unless token IDs, behavior logprobs, masks, compatible model generation, and policy-weight synchronization are supported for correct GRPO training.
5. **Verifier / reward**: apply the submitted patch to an independent clean checkout, run hidden tests inaccessible to the agent, and return a binary reward: **1.0 if verified correct, 0.0 otherwise**. Track timeout / invalid patch / infra failure separately; exclude or retry infrastructure failures rather than silently treating them as model mistakes. Protect private grader files and disallow changes to evaluation infrastructure.
6. **Trajectory recorder**: emit human-readable JSONL events for task, model generations, tool calls/results, test outputs, final diff, reward, policy version, step count, timestamps, and termination reason. Preserve model-token / logprob information through TRL rather than assuming JSONL alone is sufficient for learning. Tool output is context, **not trainable model-generated tokens**.
7. **Trainer adapter**: thin, config-driven setup around TRL `GRPOTrainer`; supports `rollouts_per_task`, checkpoint saving, LoRA/full-parameter choice, `beta`, budgets, seed, and logging. Avoid writing a second custom agent rollout loop if TRL already handles it correctly.
8. **Evaluator**: run held-out tasks before and after training, report pass rate, mean reward, average tool calls, failure reasons, time per episode, and rewards by task. Do not train on held-out tasks.

### Minimal coding tools

Expose only `list_files`, `read_file`, `apply_patch`, and `run_tests` initially. Optionally add an allowlisted `run_command` after the core works. Validate relative paths to prevent traversal, cap output size and execution time, and return structured errors rather than crashing the episode. Do not let the model modify hidden tests or the reward checker.

### Small demo task set

Include **3-5 tiny Python repository bug-fix tasks** that are executable offline (e.g. zero timeout handling, wrong boundary check, incorrect default parameter). Supply reproducible starting files, visible tests, and hidden regression tests. Include at least one task that fails and one that succeeds under a deterministic test harness. Ensure there is no solution leakage in prompts or tool observations.

## Repository layout

The implemented module boundaries are documented in [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).
Application entry/configuration live directly under `src`; RL code and shared
infrastructure have dedicated packages.

```text
coding-rl-lab/
├── pyproject.toml
├── configs/
├── src/
│   ├── main.py                     # CLI entry point
│   ├── config.py                   # validated configuration
│   ├── coding_rl/
│   │   ├── training/               # native adapter, GRPO, checkpoints, supervision
│   │   └── rollout/                # model episodes, evaluation, diagnostics
│   ├── environments/
│   │   ├── coding.py               # stateful four-tool environment
│   │   ├── patches.py
│   │   ├── verifier.py             # independent protected assertions
│   │   └── execution/              # sandbox and standalone candidate worker
│   ├── models/                     # backend, model profiles, inference worker
│   ├── harnesses/                  # scripted, OpenCode/custom drivers, protocol
│   ├── tasks/
│   │   ├── catalog.py
│   │   ├── dataset.py
│   │   └── fixtures/               # immutable starters and protected grading data
│   └── utils/                      # shared processes, IPC, records, paths and traces
├── tests/
├── docs/
└── README.md
```

Merge files if the implementation is simpler; **avoid one-line wrapper classes, registries, dependency injection frameworks, and multiple abstraction layers with only one implementation**. Use Protocol/Strategy patterns only for genuine backend/harness variation.

## Single-source YAML configuration

Everything operational should be configurable through one documented file, with command-line overrides only where helpful:

```yaml
model:
  name: Qwen/Qwen3.5-0.8B
  device: cpu                 # cpu | cuda
  train_mode: full           # full | lora

rollout:
  backend: transformers      # transformers | vllm (optional)
  device: cpu
  generations_per_task: 4
  max_tool_calls: 6
  max_new_tokens: 256
  max_context_tokens: 2048
  temperature: 0.8

environment:
  task_source: dataset       # dataset (recommended) | self_sampled (optional)
  show_state_in_debug: true

harness:
  kind: minimal              # minimal | custom
  adapter: null              # optional import path for custom harness

sandbox:
  kind: docker               # docker | trusted_local
  timeout_seconds: 20
  network: false

reward:
  kind: hidden_tests
  correct: 1.0
  incorrect: 0.0

training:
  enabled: false             # turn on only for real training
  algorithm: grpo
  beta: 0.0
  learning_rate: 0.000005
  max_steps: 10
  gradient_checkpointing: true
  output_dir: outputs/checkpoints

logging:
  level: INFO
  trace_dir: outputs/trajectories
  report_to: none

seed: 42
```

Treat this YAML as an **illustrative schema**, not guaranteed direct TRL parameters. Validate all settings and translate them into version-correct trainer arguments. For CPU-only testing, default to **rollout/eval smoke mode**, since full GRPO training of 0.8B on CPU may be very slow. Support real model-weight updates on a compatible training device. Do not silently call inference-only tests “training.”

## Developer API and CLI

Expose a small, importable Python API: `load_config(...)`, `make_environment(...)`, `run_episode(...)`, `evaluate(...)`, and `train(...)` (names may vary). Avoid mandatory FastAPI / REST servers, but keep interfaces clear enough for a future API wrapper.

Provide working commands for:

- `env-demo`: explicitly print reset -> state/action/observation transitions -> hidden verification -> reward -> cleanup, without downloading weights.
- `smoke`: perform isolated mock/deterministic episodes and security/reward assertions, without downloading weights.
- `rollout`: run a real model through a coding task and save a readable trajectory, without updating weights.
- `train`: run at least one **genuine** GRPO optimizer step and save a new checkpoint; explicitly check that trainable weights changed.
- `eval`: evaluate a chosen checkpoint on held-out tasks; save JSON metrics.

Commands should take `--config configs/default.yaml` and use the same configuration path. Supply hardware-specific examples: CPU-only smoke / inference, CUDA training + CPU rollout if supported by the chosen generation path, optional vLLM CPU rollout-only, and a lower-memory mode if needed. Never promise combinations that do not work.

## Documentation requirements

Create a concise README (setup, dependencies, exact commands, expected output, common issues) and a highly readable `docs/ARCHITECTURE.md` containing:

- One ASCII or Mermaid data-flow diagram of the full training loop, plus a separate environment state transition diagram.
- A line-by-line conceptual example of one trajectory: prompt -> file read -> patch -> failing test -> correction -> test -> reward.
- How task data, tool observations, generated tokens, masks, logprobs, rewards, and RL training batches differ.
- A short explanation of GRPO group-relative advantages and sparse RLVR rewards, and why the model learns from its actions but not from tool-output tokens.
- What the harness controls versus what the coding environment controls.
- CPU versus GPU rollout/training limitations; model and dependency compatibility; whether the vLLM option actually supports live training-weight synchronization.
- A short “how to extend” explanation showing where to add a task, tool, reward rule, or harness adapter.

## Correctness checks / acceptance criteria

1. `pytest` passes and tests cover environment reset isolation, tool bounds, path traversal rejection, private grader isolation, reward success/failure, sandbox timeout, trajectory schema, and config validation.
2. Two rollouts of the same task begin with identical clean files and cannot see each other's changes; run `env-demo` and verify it visibly shows state transitions and final reward.
3. A real model can call tools; its observations and final patch are recorded.
4. Reward comes from executable **independent hidden tests**, not merely whether the model claims success.
5. A dry run works without a GPU; a real training run performs an optimizer step and saves a changed model checkpoint on supported hardware.
6. Correct token-level action masking / logprobs / prompt reconstruction are handled through the trainer's supported mechanism. The custom harness must not falsely be advertised as GRPO-compatible if it lacks these pieces.
7. CPU vLLM is optional; if its use with live GRPO training is unsupported, provide an honest working fallback and document it.
8. Logging clearly distinguishes successful, failed, timed-out, and infrastructure-corrupted rollouts. Report both evaluation metrics and GRPO metrics, including reward variance / zero-variance groups.
9. Provide exact commands and briefly explain every file/class without generating unnecessary boilerplate.

## Implementation order

**Stage A:** Build `env-demo` FIRST: minimal deterministic stateful environment, 3–5 task fixtures, Docker sandbox, independent hidden verifier, and mock rollout tests. Confirm reset, action, observation, reward and cleanup behavior without an LLM.

**Stage B:** Real Qwen3.5-0.8B inference with Transformers, a compact harness, JSONL trajectories, and held-out evaluation. Provide a CPU option.

**Stage C:** Connect native TRL `environment_factory` support using dataset `task_id` -> `reset(task_id)` to actual GRPO updates. Validate same-task grouped rollouts, clean separate workspaces, token masking, reward propagation, and checkpoint updates. Add full / LoRA toggle.

**Stage D (optional):** Add CPU vLLM inference backend and a custom harness adapter, only when tested and demonstrably compatible with the intended mode.

At each stage, **run the tests**, fix what fails, and keep the code compact. Produce functioning files, not pseudocode.

## Important Codex instructions

Before coding, inspect the current official TRL GRPO agent-environment APIs and model-loading documentation. Prefer supported APIs (`environment_factory` / `tools`) over a handwritten trainer or complicated custom `rollout_func`. Keep dependencies pinned to mutually compatible versions. Make a small, working vertical slice first. Avoid premature concurrency, Kubernetes, databases, dashboards, microservices, elaborate event buses, or enterprise security claims.

If a genuine blocker requires a user choice, ask at most 2-3 focused questions; otherwise make reasonable defaults and document assumptions. **Do not stop after producing a plan: implement the repository, tests, and docs, and show how to run them.**

### Official references

- TRL GRPO: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL OpenEnv integration: https://huggingface.co/docs/trl/en/openenv
- vLLM CPU: https://docs.vllm.ai/en/stable/getting_started/installation/cpu/
- Qwen model: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Agent Lightning (harness design reference only): https://github.com/microsoft/agent-lightning

### Additional authoritative references to validate during implementation

- Current TRL environment contract and dataset-aware `reset`: https://huggingface.co/docs/trl/grpo_trainer
- OpenEnv vs TRL integration: https://huggingface.co/docs/trl/openenv
- Standardized OpenEnv training concepts: https://huggingface.co/docs/openenv/guides/training
- Qwen3.5 text-only/inference kernel notes: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Padding/sequence-boundary caveat for Gated DeltaNet: https://huggingface.co/docs/transformers/padding_free

### Final emphasis for Codex

**Treat the environment as the subject of the tutorial.** Make it easy to open `environment.py`, understand one reset and one state-changing action, run `env-demo`, and read a complete trajectory before ever looking at `trainer.py`. A correct, small environment with a transparent GRPO integration is the deliverable; adding fashionable tools/frameworks is not the goal.
