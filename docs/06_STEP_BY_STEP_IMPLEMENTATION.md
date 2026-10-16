# Step-by-step implementation plan and release gates

Status updated **2026-10-16**: Stage A, CPU rollout/evaluation, and the native training implementation are present. Stage C has installed native fixture tensor/update/full/LoRA reload tests; target-model development, baseline, informative update and held-out comparison remain acceptance gates. Stage D remains deferred. See [Phase 3 status](PHASE_3_STATUS.md) and the original [native training design](PHASE_3_NATIVE_TRAINING_DESIGN.md).

The current codebase audit and all remaining work are consolidated in
[Phase 4: completion plan](PHASE_4_COMPLETION_PLAN.md), including report/trace
integrity fixes, unrun Stage B/C acceptance, and the original optional Stage D
extensions. Phase 4 preserves the A–D structure below; it does not rename Stage D
or imply the remaining gates have passed.

This is the original ordered implementation roadmap. The status documents record executed evidence; future milestones below do not imply installed dependencies or completed runs.

## 1. Delivery sequence

| Stage | Learner's question | Deliverable | Gate before progressing |
|---|---|---|---|
| A | What is a stateful coding environment? | Offline tasks, four tools, isolated execution, independent reward, scripted demo | Reset isolation, bounds, positive/negative verification, cleanup |
| B | How does a model act in this world? | CPU Transformers harness, real tool calls, trajectories, held-out evaluator | Actual generated tool call and independently judged final source |
| C | How does reward change the policy? | Thin native TRL integration, genuine GRPO, full/LoRA checkpointing | Group/tensor correctness and changed weights from a genuine informative group |
| D | Can I swap the harness or inference backend? | Optional custom adapter and CPU vLLM evaluation | Equivalent environment behavior and truthful capability reporting |

Do not build optional adapters before the environment lesson works. Within each stage, implement one working path, run its relevant checks, and fix failures before expanding the surface.

| Current milestone | Evidence state | Immediate action |
|---|---|---|
| A: environment and scripted lesson | Implemented; original Docker results recorded; cleanup/cache correction tested | Preserve as the shared core |
| B: CPU loading/parser/read | Genuine target read validated and saved | Reuse the pinned profile and tokenizer EOS fix |
| B: manual harness/evaluator | Implemented; genuine development/baseline evidence open | Phase 4 P4-01/P4-05/P4-09 |
| C: native adapter, grouping and tensors | Implemented and fixture-tested; audit found correlation/error/reporting gaps | Phase 4 P4-02/P4-03/P4-04/P4-06 |
| C: optimizer, checkpoint and comparison | Full/LoRA fixture updates/reloads tested; real-model proof/comparison open | Phase 4 P4-01/P4-06/P4-07/P4-08 |
| D: custom/vLLM backends | Deferred and optional | Phase 4 P4-11/P4-12 after native acceptance |

The retained Phase 3 combined suite reports 163 passes, including native and Docker
checks. It does not replace the outstanding genuine target-model gates. The
Phase 4 planning review did not rerun that suite.

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

Implementation is present; [Phase 2 status](PHASE_2_STATUS.md) is the evidence record. The [original design](PHASE_2_MODEL_ROLLOUT_DESIGN.md) remains useful for its protocol and acceptance checklist. Its B0 cleanup/cache prerequisite is fixed; do not rebuild the working manual harness as part of training.

### B1. Resolve and record a CPU-compatible model stack

**Implemented and partially validated.** The optional rollout stack, immutable target revision, text-only loading, template identity, CPU forward/warm generation, and genuine read dispatch are recorded. Preserve the tokenizer's EOS/pad IDs and non-thinking profile. Additional devices and Linux CPU execution require their own runtime checks.

Remaining evidence: measured complete episodes, including edit/test calls and independent grading. The saved read diagnostic alone cannot certify a model repair or training feasibility.

### B2. Add the compact manual harness

**Controller implemented; real grading gate remains open.** On a Docker-capable host, build the existing execution worker, verify effective limits, and run the real Docker tests. Use the three training-partition tasks for development. Run a genuine model read/edit/test attempt through `run_model_episode`, freeze the accepted source, and independently verify it. Keep the scripted `run_episode` separate.

Completion evidence: raw model generations, validated proposed edit, visible observation, source/diff hash, verifier verdict, actual stop reason, and completed cleanup. An incorrect but validly graded attempt can pass this integration gate; an injected reference repair cannot.

### B3. Add held-out evaluation and CPU benchmarks

**CLI and scheduler implemented; reported baseline remains open.** Declare sampling, four attempts per held-out task, seeds, budgets, model/template identity, sandbox image, verifier identity, and report location before running the original checkpoint on `chunk-tail`. The supplied profile is greedy by default; repeated greedy attempts are not independent samples. Keep development tuning on the training partition.

Completion evidence: all per-attempt artifacts, planned/started/valid/incomplete counts, pass rate over valid attempts, nullable rate when none are valid, and generation/tool/grading/cleanup costs. First-token and decode-only timings remain unavailable until instrumented. CPU request timing does not predict backward-pass cost.

**B-close gates the training runtime proof:** B2's real grading evidence and B3's fixed baseline are the first remaining delivery. Native adapter tests and dependency investigation may proceed independently while Docker access is unresolved.

## 4. Stage C: native TRL and genuine updates

The detailed flow, profiles, interfaces, ownership and acceptance requirements are
in [PHASE_3_NATIVE_TRAINING_DESIGN.md](PHASE_3_NATIVE_TRAINING_DESIGN.md). The code
and fixture results are in [Phase 3 status](PHASE_3_STATUS.md); the remaining
implementation corrections and genuine acceptance runs are assigned in
[Phase 4](PHASE_4_COMPLETION_PLAN.md). C1–C7 below retain their original gate definitions.

### C1. Resolve a native dependency and hardware profile

Preserve the base/rollout installation and resolve an optional native training profile. Inspect installed `GRPOTrainer`/configuration signatures, documented extension points, parser/template selection, and text-model loading. Record exact versions/source revisions and measure a forward/backward memory and time probe before admitting an expensive run. CPU FP32 training is experimental; CUDA requires a separate compatible wheel/device profile.

Completion evidence: reproducible optional dependency graph, usable native environment API, finite forward/backward on the selected device/dtype, measured headroom, and unchanged base/CPU rollout packaging. Fail incompatible extra/device combinations before allocation.

### C2. Implement the thin adapter and public training rows

Add `src/coding_rl/training/adapter.py` with a no-argument factory and exactly four public tool methods plus reserved `reset`/`get_reward`. Keep core, run owner, cleanup, and inspection private. Construct train-only rows from public projections, with system instructions and editable paths once; dataset-driven reset returns no duplicate prompt. Reuse strict argument validation and bounded observations without embedding the manual rollout loop in TRL.

Completion evidence: exact native tool discovery/schemas, held-out ID rejection, no grading/reference metadata in prompts, genuine native tool execution, stable policy identity, and pristine independent resets when adapters are reused.

### C3. Admit complete groups with honest budgets and failures

Collect four same-task attempts with distinct episode resources and identical starter/grading identities. Correlate collection, policy version, native completion position, episode, and terminal reward. Validate actual generation/effective batch relationships and complete-group ordering.

Map each native token/context/iteration/call limit explicitly. The manual controller's all-proposal counter and one-call policy need a verified native implementation point before they can be claimed. Exercise a patch and tool result near a completion cutoff; accepted effects and observations must survive in the exact trainer-held history.

Latch tool/grader/recorder/cleanup infrastructure failures in trusted state so a native loop cannot turn them into ordinary model errors and later admit the group. Retry grading only against the same frozen submission. Require a supported complete-group admission/abort point before learning; do not silently replace a failed member with zero or resample only that member.

Completion evidence: complete group/reward alignment, bounded unknown/malformed calls under declared native semantics, boundary-history agreement, and fault injection proving infrastructure-corrupted groups cannot reach an optimizer. If required native hooks are unavailable, mark that profile unsupported and retain the working inference path.

### C4. Audit native tensors and prove cancellation ownership

Inspect actual IDs, action/loss masks, behavior/current probabilities, and reference probabilities if configured. Verify non-thinking prefixes/EOS, zero loss on observations/padding, declared handling of truncated sequences, and independent versus padded Qwen3.5 behavior. Start without packing and ensure recurrent/cache state does not leak between samples.

Supervise the native training process for hard deadlines. Register exact run-owned containers and temporary roots so the parent can release them after forced child termination; extending only the inference-worker timeout is insufficient. Preserve the last complete checkpoint and reject partial saves.

Completion evidence: supported native tensor export/audit, finite aligned learning inputs, and graceful/forced cancellation tests during generation, execution, grading, and save with no live candidate resources. JSONL re-tokenization cannot replace this evidence.

### C5. Perform one genuine informative GRPO update

Use the admitted native model to generate a group containing both verified rewards zero and one. Start with one device, four attempts, one update per collection, explicit loss/scaling/sampling, `beta=0.0`, low learning rate, and zero weight decay for the first proof. Set a finite feasibility pilot and record equal-reward/fully masked groups rather than declaring them successful learning.

TRL computes advantages, probabilities, masks, loss, and optimization. No scripted trajectories, edited rewards, or reference repairs may supply the mixed group. Legitimate task/model/budget changes create a new declared training-only pilot and compatible comparison profile.

Completion evidence: finite nonzero task-reward gradients, advanced optimizer/global step, intended trainable tensors changed above declared numerical tolerances, frozen tensors unchanged where applicable, and subsequent generation using the updated policy.

### C6. Save, reload, and compare under the fixed evaluator

Add content-identified local full/LoRA policy sources and completion manifests. Preserve existing immutable Hub configurations. Record checkpoint/base/adapter identity, native template, evaluation tokenizer, dependencies, configuration, update step, grader/image identity, and file hashes. Verify reload before publishing a checkpoint as usable; resume state is a separate contract from inference loading.

Extend the current inference worker to load trained artifacts offline. Evaluate the original and updated policy with the same B3 partition, seeds, sampling, evaluator template, limits, verifier, and denominator. Do not accidentally change the evaluation prompt by saving a native training-patched tokenizer.

Completion evidence: complete reloadable artifact, stable deterministic reload outputs/logits within tolerance, matched before/after reports, coverage/error counts, and measured costs. A lack of pass-rate improvement is acceptable if the update and comparison are valid.

### C7. Validate the second full/LoRA mode

Complete one mode first, then add the second with the same admission, tensor, update, reload, and evaluation gates. For LoRA, inspect actual matched modules, adapter settings, trainable count, frozen-base fingerprints, and active-adapter generation. Keep quantization and optional samplers outside this first proof.

Completion evidence: separate results for full and LoRA modes on the exercised device. A single certified mode supports a limited native-training release; the complete Stage C full/LoRA deliverable requires evidence for both. Publish remaining device/profile limitations explicitly.

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

The root README provides setup, scripted lessons, CPU preparation, rollout/evaluation
and experimental native training commands. It links to fixture-tested status and
does not claim a certified target-model profile. Phase 4 P4-10 completes measured
profiles, current transcripts and the release capability table. Keep historical
evidence distinct from current runtime support claims.

At each successful gate, save the resolved configuration, installed versions, model/tokenizer revisions where applicable, sandbox image identity, command, hardware, result, and known limitation. Finish with a short capability table separating model-free smoke, real CPU rollout, native training, optional vLLM, and custom harness.

Use relevant tests rather than broad repeated test runs. Model-free environment tests are meaningful because filesystem changes, private grading, resource boundaries, and cleanup are core behavior. Hardware/model integration checks run only in the corresponding profile and must never be mislabeled as exercised when skipped.

## 8. Explicit decisions and unresolved gates

| Item | Design decision / next action |
|---|---|
| Current scope | Stage A/B/C implementation exists; Phase 4 closes audit findings and genuine Stage B/C evidence before optional Stage D |
| Runtime architecture | Small local core; native TRL training adapter; manual eval harness |
| Security profile | Functional JSON tasks; separate trusted assertions and contained candidate execution |
| Patch scope | Existing allowed source files only initially; richer changes require a new tested profile |
| CPU expectation | Measure real target inference; never promise quick full GRPO training |
| Dependency versions | Base/rollout and CPU native/LoRA profiles locked; installed API/tokenizer checks recorded; target-model feasibility and other devices remain open |
| Target native tool support | Manual target read and native protocol/fixture tests exist; genuine native multiline edit/update proof remains |
| Native hard budgets and side effects | Sequential-call native semantics and cutoff rejection implemented; Phase 4 completes normal-error handling and genuine boundary evidence |
| Native model-generation trace export | Admitted native IDs/tensors retained; Phase 4 adds initial event identity, rejected-collection evidence and stop/correlation detail |
| Native infrastructure exceptions | Failure latching/group rejection fixture-tested; extend actual lifecycle and healthy-error classification in Phase 4 |
| Hard native cancellation | Registered names/private roots and parent cleanup implemented; actual-stage interruption and committed-checkpoint recovery coverage remain |
| Checkpoint evaluation | Local full/LoRA identities/loaders implemented; stronger evidence schema and genuine matched comparison remain |
| CPU vLLM | Optional eval capability, subject to checkpoint/hardware tests |
| Mixed CPU generation / GPU training | Outside initial supported profiles; requires demonstrated synchronization and probability contract |
| Scale and benchmark strength | Four tasks demonstrate plumbing and learning mechanics; larger independent task families needed for general claims |
