# Step-by-step implementation plan and release gates

Status: Stage A is implemented and tested; Stages B–D remain future work. This document retains the staged checklist. See [Stage A status](PHASE_A_STATUS.md) for actual commands, results, and implementation choices.

## 1. Delivery sequence

| Stage | Learner's question | Deliverable | Gate before progressing |
|---|---|---|---|
| A | What is a stateful coding environment? | Offline tasks, four tools, isolated execution, independent reward, scripted demo | Reset isolation, bounds, positive/negative verification, cleanup |
| B | How does a model act in this world? | CPU Transformers harness, real tool calls, trajectories, held-out evaluator | Actual generated tool call and independently judged final source |
| C | How does reward change the policy? | Thin native TRL integration, genuine GRPO, full/LoRA checkpointing | Group/tensor correctness and changed weights from a genuine informative group |
| D | Can I swap the harness or inference backend? | Optional custom adapter and CPU vLLM evaluation | Equivalent environment behavior and truthful capability reporting |

Do not build optional adapters before the environment lesson works. Within each stage, implement one working path, run its relevant checks, and fix failures before expanding the surface.

## 2. Stage A: environment lesson without a model

### A1. Establish the smallest package and configuration

Create Python 3.11+ package metadata with a `src` layout, a single CLI entry point, and lightweight dependencies for configuration and authored tests. Model/training/vLLM extras remain optional. Start with validated configuration and ordinary dataclasses or typed records, not a plugin registry.

Define IDs and records for task, episode state, tool result, frozen submission, verification verdict, and episode summary. Separate model-facing observations from developer metadata. Add a versioned event schema before several modules start emitting incompatible logs.

Completion evidence: invalid limits/backend settings fail before allocation; model-free commands import without a training stack; resolved configuration can be saved with a run.

### A2. Build four immutable task fixtures

Use `zero-timeout`, `range-boundary`, and `empty-default` for the training/demo partition, and `chunk-tail` for the initial held-out partition. Define every callable's JSON input/result contract and valid input domain. These tiny functional tasks permit strong separate-process grading while preserving the coding-agent experience of file inspection, editing, and visible testing.

Each fixture contains an issue, starter source, allowed source paths, authored visible tests, protected private cases, budgets, and an immutable content/revision identity. Protected grading material is not copied into the agent working copy or public dataset. Include known-good and known-bad repairs in test fixtures outside agent-visible paths.

Completion evidence: buggy source fails intended checks; a human-authored correct repair passes; a repair passing only visible tests fails a private case; repeated fixture loads have the same source hash. Confirm no related task variants cross partitions.

### A3. Implement workspace ownership and Docker execution

Create one private authoritative directory per episode from the immutable starter. Source edits happen through trusted file operations, never by executing candidate code on the host. Each visible test uses a disposable contained worker with a read-only source/test snapshot and bounded temporary storage.

Configure no network, non-root execution, dropped capabilities, resource/output caps, process timeout, and guaranteed cleanup. The worker image includes offline dependencies. Preflight checks prove the host supports required limits before generated code runs. Keep a local backend only behind an explicit trusted-fixtures-only choice.

Completion evidence: workers cannot write the authoritative source, access another episode or private grader, reach the network, or outlive cancellation. A deliberate infinite-loop fixture is terminated and classified correctly. Docker unavailable is reported as infrastructure failure, never “incorrect solution.”

### A4. Implement the readable core environment

Implement reset, the four named tools, budget counters, developer inspection, freeze, terminal grading coordination, and idempotent cleanup. Reset first releases stale state and then allocates a fresh workspace. Tools require an active episode; mutations after freeze are rejected.

Begin with restricted unified diffs modifying existing allowlisted regular text files. Reject traversal, symlinks, unexpected file modes, protected test/config changes, binary patches, renames, additions, and deletions in this profile. Parse and validate every hunk before atomic mutation. A future richer task profile may support new/deleted files, with explicit manifest capture and corresponding checks.

Completion evidence: read-only actions leave source hashes unchanged; one accepted patch changes the expected bytes; invalid patches leave all files unchanged; unsuccessful calls consume the documented attempt budget; inspection cannot become a model-callable tool.

### A5. Implement independent executable RLVR

Freeze allowed source contents and reconstruct them on a separate pristine starter. Restore authored test infrastructure. A trusted private assertion controller submits bounded JSON cases to a separate candidate-execution Docker worker; only inputs and actual results cross the boundary. The controller never imports candidate modules.

Return 1.0 only for a valid completed pass and 0.0 for a valid completed failure. Classify submitted-code exceptions/timeouts separately from controller/container failures. Retry infrastructure failures against the same frozen snapshot; do not cache an incomplete failure as a correctness verdict.

Completion evidence: final source is independently reconstructed, grader assertions are absent from the candidate worker, forged stdout does not imply success, a correct candidate earns one, an incorrect candidate earns zero, repeated grading returns the cached completed result, and all resources are released.

### A6. Deliver the model-free lesson

Implement `env-demo`, `smoke`, and `env-inspect` using the scripted harness. Show reset, action, observation, source change, visible test outcome, freeze, private verdict, reward, and cleanup. Run the same task in two episode IDs; print starter hashes proving the second did not inherit the first edit.

Completion evidence: readable actual transcript saved for the future setup README; tests cover reset isolation, tool bounds, protected paths, private grading, timeouts, traces, configuration, and cleanup after exceptions. No model download is required.

## 3. Stage B: real model rollout and evaluation

### B1. Resolve and record a CPU-compatible model stack

Preserve `Qwen/Qwen3.5-0.8B` as the target. Inspect the current official text-only loading route, supported architecture class, tokenizer/template, and CPU fallback. Test generation and tool parsing before running a full task. Pin a working CPU profile with model revision and hardware metadata. The older plain-attention small fallback described in the training document may validate infrastructure when the target gate is unresolved; report which model actually ran.

Completion evidence: a real CPU forward/generation succeeds, produces a parsable allowed tool call, and records memory and latency. Plain text generation alone is insufficient evidence of tool support.

### B2. Add the compact manual harness

Implement the episode flow in [the harness document](05_HARNESS_ROLLOUTS_AND_EVALUATION.md). Keep generation, parsing, dispatch, budget accounting, and cancellation readable. Preserve raw generations and exact tool observations in the trace. Do not infer success from the final assistant message.

Completion evidence: a real model can read and attempt an edit, receives visible feedback, and ends with a frozen candidate judged independently. A failed model attempt still demonstrates valid rollout plumbing; do not inject the scripted repair into its prompt or score.

### B3. Add held-out evaluation and CPU benchmarks

Implement `rollout` and `eval`, saved JSON summaries, per-task outcomes, and coverage/error counts. Evaluate the original checkpoint before training with fixed settings. Record model load time, generation throughput, peak resident memory, tool execution time, and grading time.

Completion evidence: valid metrics denominators, protected held-out partition, unchanged model weights in rollout mode, repeatable fixture state, and clear distinction between model failure and infrastructure outage.

## 4. Stage C: native TRL and genuine updates

### C1. Pin an actually supported native environment contract

Inspect installed `GRPOTrainer`/configuration signatures and corresponding versioned official documentation. Create the no-argument factory and narrow adapter. Use an external dataset of public prompts plus task IDs; reset produces no duplicate user prompt. Verify exact public tool discovery and document environment reuse behavior.

Completion evidence: exact versions recorded; only four tools discovered; dataset metadata reaches reset; native tool loop runs; repeated adapter reset allocates pristine separate state. Do not silently replace the required integration with an eval-only manual loop.

### C2. Prove grouping, budgets, and learning sequence correctness

Collect a group of four same-task attempts with distinct episode resources and identical starter hashes. Validate completion ordering, generation/batch divisibility, seed behavior, terminal reward alignment, and complete-group infrastructure handling.

Inspect the native learning sequence: prompt/observation conditioning, generated call tokens, masks, and applicable log probabilities. Check chat-template prefix preservation. For Qwen3.5, start with independent padded sequences and disabled packing; verify batched versus independent behavior and recurrent-state boundaries.

Deliberately exercise token/context/tool limits, including a patch call near a native completion boundary. Prove that accepted side effects match the recorded trainable action history and that native truncation policy is applied consistently. A supported hook or explicit fail-before-update behavior is necessary when this cannot be guaranteed. Avoid a private-method monkeypatch presented as supported integration.

Completion evidence: all invariants pass, loss inputs are aligned, and no unsafe or infrastructure-corrupted group reaches an optimizer. If a compatibility gate fails, finish the working environment/CPU rollout deliverables and document the precise native-training blocker.

### C3. Perform one real informative GRPO update

Configure a small single-device run using native Transformers generation. Let the actual policy generate the attempts. Obtain at least one valid nonzero-variance reward group without substituting scripted trajectories or edited rewards. If all groups have equal rewards, use legitimate easier training tasks, longer verified budgets, or a more capable explicitly selected model; report the change.

Use TRL for advantages, selected GRPO-family loss, masks, probabilities, and optimization. Start with one update per collected batch, a documented low learning rate, and explicit KL coefficient. Record the chosen loss type rather than relying on an evolving library default. Keep CPU full training experimental and measured; CUDA is the preferred practical training device when available.

Completion evidence: finite loss and gradients, incremented optimizer/global step, changed intended trainable parameters beyond recorded numerical tolerance, unchanged frozen parameters where applicable, saved checkpoint, and successful reload. A step counter or newly written checkpoint directory alone is insufficient.

### C4. Add LoRA/full modes and compare held-out behavior

Add PEFT only after the native full-path mechanics are understood, or use it as an explicitly tested memory profile when needed. Verify correct target modules, trainable parameter count, checkpoint/base-model identity, and reload behavior. Full and LoRA modes have separate compatibility results.

Evaluate the saved checkpoint under the Stage B protocol. Report before/after verified metrics, equal-reward group rate, sampling/resource settings, and small-sample limits. A lack of improvement is acceptable evidence if the update itself is valid.

## 5. Stage D: optional backends after the core works

### D1. Custom harness adapter

Add the narrow rollout/evaluation adapter without direct workspace access. Test action dispatch, trace correlation, cancellation, and unchanged verification semantics. Mark training capability unavailable unless a separately tested native tensor/weight-synchronization contract exists.

### D2. CPU vLLM inference

Resolve an appropriate CPU build for the actual hardware and checkpoint. Test architecture loading, kernels, chat-template/tool parsing, bounded outputs, memory, and latency against Transformers. Use local inference/evaluation only initially; pin this profile independently if necessary.

The native TRL server path must not be assumed compatible with a CPU inference server. Promote a training backend only after proving policy synchronization, supported probability outputs, token/mask alignment, and actual optimizer-to-inference weight refresh. Otherwise keep the working Transformers training fallback.

### D3. Other environment interfaces

An OpenEnv or Gymnasium adapter is a later teaching exercise over the same core, not a second core implementation. Add it only for a concrete use case. No database, cluster, REST service, UI, or queue is required by the first four stages.

## 6. Acceptance matrix

| Requirement | Stage | Evidence |
|---|---|---|
| State versus observation is understandable | A | Demo trace and state diagram correspond to actual operations |
| Fresh reset and no edit leakage | A/C | Repeated task plus four independent group states |
| Safe tools and bounded errors | A | Paths/patches/output/call limits and atomic rejection checks |
| Contained untrusted execution | A | Default Docker limits and no host candidate import |
| Independent protected executable reward | A | Known-good, known-bad, visible-only false positive, cached verdict |
| Timeout and infrastructure classification | A/C | Candidate timeout versus engine outage; no invalid group update |
| Real model-generated tools | B | Raw generation, parsed call, observation, and frozen submission |
| CPU support honestly described | B | Measured inference gate; training separately identified |
| Exact native API/tool discovery | C | Pinned versions, signatures, schemas, dataset reset test |
| Correct masks/history/probabilities | C | Native tensor inspection and near-boundary mutation check |
| Qwen3.5 sequence boundaries | C | Padded independent behavior; no unsupported packing |
| Genuine policy update | C | Informative generated group, gradients, changed weights, reloadable checkpoint |
| Held-out evaluation | B/C | Fixed partitions and before/after reports with coverage |
| Full/LoRA behavior | C | Trainable/frozen parameter and checkpoint reload checks |
| Custom/vLLM mode limits | D | Explicit capability table and tested eval path |
| Clear docs and compact code | All | Every module has a purpose; no unsupported command/result claims |

## 7. Implementation-time documentation and handoff

When Stage A works, add the repository setup README with exact dependency setup, runnable CLI invocations, actual demo output, and common failures. Keep these design documents under `docs/` and mark decisions that changed. The present `docs/README.md` is a design index, not a setup guide for a package that does not yet exist.

At each successful gate, save the resolved configuration, installed versions, model/tokenizer revisions where applicable, sandbox image identity, command, hardware, result, and known limitation. Finish with a short capability table separating model-free smoke, real CPU rollout, native training, optional vLLM, and custom harness.

Use relevant tests rather than broad repeated test runs. Model-free environment tests are meaningful because filesystem changes, private grading, resource boundaries, and cleanup are core behavior. Hardware/model integration checks run only in the corresponding profile and must never be mislabeled as exercised when skipped.

## 8. Explicit decisions and unresolved gates

| Item | Design decision / next action |
|---|---|
| Current scope | Stage A implemented; Stage B real model rollout is next |
| Runtime architecture | Small local core; native TRL training adapter; manual eval harness |
| Security profile | Functional JSON tasks; separate trusted assertions and contained candidate execution |
| Patch scope | Existing allowed source files only initially; richer changes require a new tested profile |
| CPU expectation | Measure real target inference; never promise quick full GRPO training |
| Dependency versions | Stage A locked with uv; model/training profiles await their compatibility checks |
| Target native tool support | Qwen3.5 family documentation is evidence; exact 0.8B/checkpoint/device gate remains |
| Native hard budgets and side effects | Verify source behavior; reject history/state mismatch before training |
| Native model-generation trace export | Establish supported export/correlation points without inventing callbacks |
| CPU vLLM | Optional eval capability, subject to checkpoint/hardware tests |
| Mixed CPU generation / GPU training | Outside initial supported profiles; requires demonstrated synchronization and probability contract |
| Scale and benchmark strength | Four tasks demonstrate plumbing and learning mechanics; larger independent task families needed for general claims |
