# Phase 4: complete real-model acceptance and the remaining roadmap

Review date: **2026-10-16**. This plan covers the current working tree, including
the Phase 3 implementation. It consolidates the unfinished work from
[the step-by-step roadmap](06_STEP_BY_STEP_IMPLEMENTATION.md),
[the original specification](../CODEX_AGENTIC_RL_ENV_SPEC.md), and
[the Phase 3 design](PHASE_3_NATIVE_TRAINING_DESIGN.md).

Phase 4 has two parts: finish the required Stage B/C delivery, then address the
original optional Stage D extensions. Phase 4 is not synonymous with Stage D.
Existing environment, manual evaluator, native adapter, and verifier code should
be reused. This document plans work; it does not implement or certify that work.

## 1. Audit scope and evidence

The review covered all 22 runtime modules, the test modules and shared fixtures,
the four task families, YAML profiles, package/lock configuration, Docker build,
and design/status/run documentation. The retained
[Phase 3 validation record](evidence/phase3-validation.json) reports 163 passing
tests, including installed native and Docker checks. That suite was **not rerun**
for this planning review.

Three small synthetic probes were run in temporary directories: native reset
trace identity, prerequisite-report consistency, and duplicate-completion tensor
matching. They loaded no target weights, performed no optimizer update, and
sampled no held-out task. Other findings below come from source/test inspection.

The principal acceptance gap is unchanged: no complete genuine target-model
development episode, fixed held-out baseline, target-model forward/backward
feasibility measurement, informative target-model update, or matched held-out
comparison has been retained. Fixture-generated completions cannot close those
gates. Several code-level gaps should be fixed before collecting that evidence.

### Concrete findings to address first

| Finding | Evidence in the current code | Required Phase 4 action |
|---|---|---|
| Prerequisite evidence is insufficiently validated | `trainer.validate_prerequisites` accepts a synthetic baseline with a four-attempt profile but `planned == valid == 0`, no attempt records, and a development outcome with `reward: null` and an incomplete verdict | P4-01: validate report structure, recompute coverage, and cross-check verdict/source/trace evidence |
| Native reset trace has the wrong policy | `NativeEnvironment.reset` calls `core.reset` before `RunOwner.register` assigns recorder policy; the first reset probe records `kind: scripted` | P4-02: establish collection/policy correlation before the first event |
| Completion identity can be ambiguous after shuffling | `RunOwner.audit_tensors` chooses the first matching prompt/completion IDs; the duplicate-ID probe with different rewards and reordered advantages is falsely rejected | P4-02: validate duplicate-aware correlation; this is conservative rejection, not demonstrated wrong optimization |
| Ordinary native errors are conflated with missing effects/history | Admission requires every native tool observation to equal an adapter-recorded observation; unknown names or signature failures can generate native error observations without entering the adapter | P4-02: distinguish healthy rejected calls from infrastructure corruption while retaining whole-group admission |
| Rejected native collections lose useful diagnostic data | Native IDs are written only after all admission checks succeed; group records have no explicit rejection reason, per-attempt native stop reason, or trace-directory reference | P4-02/P4-09: retain available rejection evidence before raising |
| Feasibility measurement is incomplete and tightly coupled to training | `execute_training` requires B-close and constructs model/trainer before `feasibility_probe`; `train --probe` checks only API/tokenizer compatibility | P4-04: add independently runnable, supervised model feasibility and pre-allocation checks |
| Feasibility does not check all gradients | `feasibility_probe` accepts `any` finite nonzero trainable gradient; its three-token backward does not establish four-attempt training or reload headroom | P4-04: validate every present trainable gradient and measure the actual native layout and optimizer/save costs |
| Checkpoint evidence schema is permissive | `_validate_metadata` requires nonempty evidence dictionaries and `reload.passed`, but does not enforce their detailed contract or link consistency | P4-01/P4-07: validate proof fields and references, not just content digests |
| Cancellation coverage is narrower than the original acceptance gate | Ownership tests use sleeping children and synthetic directories for named stages, plus one live Docker worker; no test interrupts every actual native/grading/save stage or proves preservation after a previous real checkpoint commit | P4-03: exercise the real lifecycle and recovery boundaries |
| Benchmarks and training summaries remain partial | First-token/decode-only timings are `null`; evaluation has totals but lacks several planned means/rates; native groups have an informative flag but no aggregate reward-variance/rejection summary | P4-09: implement and define the missing measurements |
| Public model override is not a working fallback profile | Inference loading and checkpoint reload hardcode `Qwen3_5ForCausalLM`; native compatibility hardcodes the Qwen parser and EOS/pad IDs | P4-04: provide the originally requested small pretrained fallback through an explicit tested profile |
| Optional adapters do not exist | No custom harness contract/config, inference backend selector, CPU vLLM dependency profile, or OpenEnv/Gymnasium adapter exists | P4-11–P4-13, with the original optional/conditional scope retained |
| Current documentation contains conflicting statuses | Roadmap tables still call native code unimplemented; architecture/environment-support opening statements still describe training as future or TRL as untested | P4-10: finish the documentation handoff; current status pointers are corrected with this plan |

Content hashes establish file consistency. Strengthening report validation does
not authenticate arbitrary externally authored evidence; no signing service or
new trust framework is required for this local lab.

## 2. Coverage of the original step-by-step plan

| Original item | Current state | Remaining Phase 4 work |
|---|---|---|
| A1: package/configuration | Implemented; optional imports and packaging checks recorded | Preserve during changes; P4-10 regression checks and documented Python/platform coverage |
| A2: four immutable fixtures | Implemented; three train families and one held-out family | Preserve identities/splits; family-level duplicate checks only when adding task variants, P4-13 |
| A3: workspace/Docker isolation | Implemented; real Docker tests recorded | P4-03 extends cancellation/recovery through the actual native lifecycle |
| A4: four-tool core | Implemented; restricted atomic patches, bounds, freeze/cleanup | Reuse; P4-02 covers native argument/error/trace integration |
| A5: independent executable RLVR | Implemented; frozen reconstruction, private JSON cases, retries/cache | Reuse; P4-01 validates evidence and P4-03 exercises grading interruption |
| A6: model-free lesson | Implemented; original transcript retained | P4-10 supplies a current schema-v2 transcript and accurate walkthrough |
| B1: CPU-compatible model stack | Pinned stack and genuine recorded read; no complete current-host target episode | P4-04/P4-05 establish measured model/device compatibility |
| B2: manual model harness | Implemented; genuine read/edit/test/verdict gate open | P4-05 executes a development episode without oracle repairs |
| B3: held-out evaluation/benchmarks | Scheduler, coverage, profile checks and costs implemented; baseline unrun | P4-01, P4-05, P4-09 |
| C1: native dependencies/hardware | Installed stack/API/tokenizer tests recorded; target capacity unmeasured | P4-04, with conditional device work in P4-13 |
| C2: thin adapter/public rows | Implemented and fixture-tested | P4-02 fixes initial event correlation; P4-06 exercises genuine native calls |
| C3: groups/budgets/failures | Fixture-tested; sequential calls and limited counters explicitly declared | P4-02/P4-06 complete errors, identity, genuine boundary and stop-reason evidence |
| C4: tensors/cancellation | Actual fixture tensors/probabilities tested; target lifecycle unrun | P4-02/P4-03/P4-04/P4-06 |
| C5: informative GRPO update | Fixture optimizer proof only | P4-06 supplies genuine mixed rewards, gradients, changed weights and updated-policy generation |
| C6: checkpoint/evaluation comparison | Full/LoRA artifacts and evaluation loaders exist; real comparison unrun | P4-01/P4-07 |
| C7: second parameter mode | Both modes fixture-tested; neither target mode certified | P4-08 independently repeats the real gates |
| D1: custom harness | Absent; originally optional | P4-11 after the required path works |
| D2: CPU vLLM | Absent; originally optional and compatibility-dependent | P4-12 after the required path works |
| D3: other interfaces | Absent; conditional on a concrete teaching/use case | P4-13; no extra server/framework requirement |
| Acceptance matrix and handoff | Good fixture evidence; several real-model rows still open | All required work below, finalized in P4-10 |

## 3. Required completion work

### P4-01. Validate baseline, development, and checkpoint evidence

**Priority:** before a real training proof. **Owners:**
[trainer.py](../src/coding_rl/trainer.py),
[evaluate.py](../src/coding_rl/evaluate.py),
[checkpoints.py](../src/coding_rl/checkpoints.py),
[cli.py](../src/coding_rl/cli.py).

Replace the current counter-only prerequisite checks with a small validated report
contract. Require the planned total to equal selected tasks times attempts per
task and be nonzero. Recompute started/valid/incomplete/not-started counts from
actual attempt records; validate task IDs, seeds, duplicates, policy identity,
starter/grading/image identities, binary reward/verdict agreement, and cleanup.
Cross-check a development outcome with its real episode trace and frozen-source
artifact, including successful public visible-test execution feedback, an accepted
edit, completed independent grading, and matching source hashes. A valid incorrect
repair remains acceptable development evidence.

Validate detailed checkpoint metadata: kind, immutable base/tokenizer identity,
step, intended trainable parameters, finite update measurements, source/config/
dependency/template fingerprints, reload tolerances/result, and collection/tensor
references. Require the files needed for each artifact kind. Reject incomplete or
contradictory records with useful diagnostics. Preserve existing valid artifacts
through an explicit schema/version policy rather than silently reinterpreting them.

Protect the declared baseline from accidental overwrite by an evaluation using
the same report path. Continue comparing evaluator behavior independently of
policy weights and report locations.

**Done when:** negative tests reject an empty/inconsistent baseline, duplicate or
missing attempts, reward-less development verdicts, unrelated traces/snapshots,
changed profiles, baseline overwrite, and contradictory checkpoint proof fields;
complete valid evidence is accepted without relying on a handwritten pass counter.

### P4-02. Complete native correlation, failure classification, and audit artifacts

**Priority:** before native pilots. **Owners:**
[trl_env.py](../src/coding_rl/trl_env.py),
[trainer.py](../src/coding_rl/trainer.py),
[trace.py](../src/coding_rl/trace.py).

Set native run/collection/policy/attempt identity before reset emits an event.
Retain a direct episode-to-trace reference in the group ledger and link the
native conversation, exact IDs, tensor/probability artifacts, frozen source and
verdict. Preserve available raw native collection data and an explicit rejection
reason even if admission fails halfway through a group. Where a supported native
export exists, retain turn/generation boundaries and actual stop reasons. Mark
unavailable counters as unavailable.

Test duplicate prompt/completion IDs after native shuffling, including different
rewards/advantages. Use supported correlation metadata or a validated matching
rule that accounts for every member once; do not assume the first identical
sequence uniquely identifies an episode. Do not inject private identity into
model prompts or override private TRL collection methods to make this work.

Differentiate ordinary unknown tools, missing/extra arguments, malformed responses,
and healthy rejected calls from latched infrastructure failures. Verify the native
error result against actual IDs/masks without inventing an executed core action.
Retain rejection of any trajectory whose committed effect or observation vanished
at a cutoff. If a native guarantee still lacks a supported hook, keep the profile
limitation explicit and prevent a stronger claim.

Preserve the declared Phase 3 choices: four attempts per complete group, sequential
native multi-call execution, six entered-core calls by default, and no advertised
all-proposal counter or atomic single-call guarantee. Strict single-call parity
is a new profile gate if pursued, not a reason to duplicate TRL's tool loop.

**Done when:** the first and reused reset events carry current native identity;
accepted and rejected groups have inspectable correlated evidence; duplicate IDs,
normal parser/signature errors, partial patches, multi-call proposals, completion
and context cutoffs, and recorder/cleanup faults have tested classifications;
corrupted or incomplete groups cannot reach backward/optimizer admission.

### P4-03. Prove operational cleanup and committed-checkpoint recovery

**Priority:** before native runtime acceptance. **Owners:**
[training_worker.py](../src/coding_rl/training_worker.py),
[sandbox.py](../src/coding_rl/sandbox.py),
[environment.py](../src/coding_rl/environment.py),
[checkpoints.py](../src/coding_rl/checkpoints.py).

Exercise graceful interruption and forced child termination during actual native
generation, visible tests, private grading, and checkpoint save/reload/publication.
Extend the existing process/Docker tests rather than replacing them. Include a
run with a previously committed checkpoint followed by an interrupted later save,
and a stop immediately after publication but before the child reports success.

Test RAM-cap enforcement, child crash/EOF, registry truncation, Docker control
failure, cleanup failure, and result/group-record recovery. Always retain a final
incomplete/cancelled record when the child cannot finish, without publishing a
failed attempt as reward zero. Recovered group/checkpoint statuses must agree with
verified completion manifests. Preserve only exact owned names/roots; unrelated
containers and paths remain untouched.

**Done when:** no candidate worker, writable episode/snapshot root, or usable
partial checkpoint survives; earlier committed checkpoints still load; a correct
verdict with failed cleanup stays incomplete; real-stage interruption results and
stop reasons are retained. Polling RSS is measured process supervision, not a
claim of instantaneous kernel-enforced memory containment for the trainer.

### P4-04. Establish model feasibility and explicit supported profiles

**Priority:** before expensive collection; independent of held-out sampling.
**Owners:** [probe.py](../src/coding_rl/probe.py),
[trainer.py](../src/coding_rl/trainer.py),
[models.py](../src/coding_rl/models.py),
[inference_worker.py](../src/coding_rl/inference_worker.py),
[config.py](../src/coding_rl/config.py), profiles and dependency locks.

Keep the existing API/tokenizer-only `train --probe` identity. Add a separately
named, supervised model feasibility operation that can run without a completed
baseline or starting training. Its CLI spelling is an implementation choice; no
new probe command is implemented by this plan.

Check host/package/device constraints and conservative weight/gradient/optimizer
capacity before large allocations where possible. Then measure real target
loading, forward, backward, representative four-attempt sequence lengths, audit
overhead, first optimizer-state allocation, checkpoint staging and reload headroom.
Check every present trainable gradient for finiteness and require a nonzero task
gradient when claimed. Record trainable/frozen names/counts and actual LoRA target
modules/rank/scaling/dropout; reject empty or unexpectedly broad selections.
Validate recurrent/cache state and independent/padded behavior on genuine inputs.
Restore gradients/random state after diagnostics.

Record CPU model, cores/threads, relevant instruction support, available/total RAM,
OS/Python, exact versions/source/lock identity, dtype, model class/loading report,
template/parser/EOS settings, measured costs and budget decision. Distinguish a
three-token smoke forward from a capacity result for the actual native layout.
Keep batch four/accumulation one unless a separately verified complete-group
admission design supports another layout.

Provide the original specification's small **pretrained** fallback profile. The
tiny random fixture model is not that fallback. Model selection must include an
explicit loader, parser/template/token identity, checkpoint reload contract and
capability record; changing `model.name` alone is insufficient. Preserve Qwen as
the target, pin the fallback, and report any fallback pilot under its own name and
baseline. No silent fallback or reference completion is permitted.

**Done when:** Qwen's actual capacity is measured or the profile refuses it clearly;
one explicit fallback is runnable; reports separate API probe, model smoke, real
feasibility, and training. Full/LoRA and CPU/device differences have honest status.

### P4-05. Close Stage B with a genuine development episode and baseline

**Type:** execution/evidence, plus fixes exposed by integration.
**Owners:** [rollout.py](../src/coding_rl/rollout.py),
[evaluate.py](../src/coding_rl/evaluate.py), CPU profiles and evidence records.

Build the no-extra Docker worker and verify effective limits. On the intended
runtime host, prepare the pinned target and run an authentic training-partition
read/edit/test episode through the manual harness. Save actual generated IDs/text,
parsed calls, accepted edit/source, visible feedback, frozen submission/diff,
independent verdict, stop reason, cleanup and costs. A valid failed repair can
close this integration gate. A source-read diagnostic or scripted repair cannot.

Declare the evaluation profile before sampling `chunk-tail`: model/tokenizer and
template identity, sampling, four attempts, seeds, budgets, image and grader. Use
the training partition for prompt/budget development. Then save all four planned
held-out attempts with actual coverage and nullable denominators. The current
training prerequisite demands complete valid coverage; address infrastructure
failures under an explicit scheduling/retry record rather than deleting attempts
or regenerating for a better score. Lock and retain the resulting baseline.

**Done when:** the strengthened P4-01 validator accepts the genuine development
and baseline artifacts, and the Linux/CPU or other selected runtime has measured
evidence. The historic macOS read and current fixture tests remain separately named.

### P4-06. Demonstrate the first informative native target-model update

**Prerequisites:** P4-01–P4-05; performance instrumentation needed for the declared
profile should be completed before collecting its comparison data.
**Owner:** [trainer.py](../src/coding_rl/trainer.py).

Start with one measured parameter mode, normally LoRA if full training is too
large. Retain four independent same-task attempts, one live policy/device,
explicit native GRPO/group scaling, `beta=0`, zero weight decay and the finite
eight-group pilot. Native TRL owns generation, learning tensors and optimization.

Use genuine policy generations, including a native multiline patch, and
independently verified rewards. Retain all complete, rejected, equal-reward and
truncated groups. An informative group requires both zero and one, finite nonzero
task gradients, actual trainable parameter changes above tolerance, unchanged
frozen parameters where applicable, and generation from the updated live policy.
Keep real native IDs/masks/probability and sequence-boundary evidence.

**Done when:** one genuine informative update is demonstrated and correlated to
its source verdicts. No mixed group within the pilot means the gate remains open;
report that outcome. Do not fabricate variance, select only successful members,
or describe Adam momentum after an equal-reward group as new task learning.

### P4-07. Verify a real portable checkpoint and matched held-out comparison

**Prerequisites:** P4-06 and the checkpoint integrity/recovery work above.
**Owners:** [checkpoints.py](../src/coding_rl/checkpoints.py),
[inference_worker.py](../src/coding_rl/inference_worker.py),
[models.py](../src/coding_rl/models.py),
[evaluate.py](../src/coding_rl/evaluate.py).

Save a completed content-identified full or LoRA artifact from the genuine update.
Tie base/adapter bytes, update step, group/tensor/probability evidence, dependencies,
source/config, tokenizer/EOS, native/evaluation templates, task/verifier and worker
image identities together. Verify deterministic logits/generation reload with a
representative coding/tool prompt as well as the current short synthetic probe.
Retain evaluation tokenizer/template behavior independently of training templates.

Load offline through the actual spawned inference worker and invoke the ordinary
held-out evaluator with the P4-05 profile. Before/after reports retain per-attempt
results, coverage, infrastructure exclusions, costs and per-task verdicts. If
valid coverage differs, show both denominators and explain comparability; a zero
denominator never yields a pass rate. A pass-rate improvement is not required.

**Done when:** a genuine checkpoint loads through the normal evaluation path and
has a retained matched before/after report. No test loader substitution or
scripted completion supplies this acceptance evidence.

### P4-08. Independently complete the second full/LoRA mode

Repeat feasibility, grouping, masks, parameter/frozen-base checks, genuine update,
save/reload, inference loading and matched evaluation for the other parameter mode.
Verify active-adapter generation and exact resolved modules for LoRA. Exercise
both artifact kinds in the spawned inference worker; currently only full has that
specific fixture subprocess test, while LoRA fixture loading is tested in-process.

**Done when:** each supported mode has its own actual result. If full CPU training
does not fit, retain the measured refusal and describe a limited LoRA release;
the original complete full/LoRA Stage C deliverable remains open. Additional
hardware requires its own profile, not a `device` override on a CPU-only stack.

### P4-09. Finish metrics and measured CPU benchmarks

**Owners:** [inference_worker.py](../src/coding_rl/inference_worker.py),
[models.py](../src/coding_rl/models.py),
[rollout.py](../src/coding_rl/rollout.py),
[evaluate.py](../src/coding_rl/evaluate.py),
[trainer.py](../src/coding_rl/trainer.py).

Instrument actual first-token and decode-only timings if reporting them; keep
`null` until implemented. Separate cold load/warmup, prompt prefill, generation,
tools, private grading, cleanup, and complete attempt cost. Document that existing
worker peak RSS is a process-lifetime high-water mark, rather than falsely calling
it incremental per-request memory. Include benchmark host/thread/context settings.

Add the planned average attempted/accepted calls, parser/error rates, per-task
mean reward and costs, and limit-caused successes without changing valid-attempt
denominators. Native training summaries need reward variance, equal-reward and
rejection fractions, truncation/masked-token counts where observable, stop/reject
reasons, loss/gradient/step evidence, policy versions and resource costs. Persist
these alongside run artifacts rather than relying only on transient TRL stdout.

**Done when:** definitions, units and denominators are documented and tested;
unavailable measurements remain explicit; one reproducible CPU benchmark report
describes observed costs without promising general training latency. Pass-at-k
and broad benchmark claims are not required for the initial lab.

### P4-10. Complete the tutorial, profiles, packaging, and release handoff

**Owners:** README/design/status documents, [config.py](../src/coding_rl/config.py),
[cli.py](../src/coding_rl/cli.py), profiles, [pyproject.toml](../pyproject.toml),
[Dockerfile](../docker/Dockerfile), and focused regression tests.

Supply distinct measured baseline, full/LoRA training and trained-evaluation
examples, with existing commands and accurate required preparation. Do not make
the learner invent an undocumented checkpoint configuration. Explain any
illustrative original YAML settings that remain deliberately unsupported. Keep
group size/finiteness, KL-zero, disabled packing/checkpointing and CPU FP32 as
certified-profile restrictions until separately extended.

Update the actual environment walkthrough to schema-v2 order: verification,
cleanup, then usable reward. Align architecture/environment-support status,
method-level explanations, importable API paths, troubleshooting and capability
tables with executed evidence. Preserve dated Phase A/2/3 results as historical
records; do not rewrite their measurements as Phase 4 runs.

Keep one readable core and native GRPO owner. The current `trainer.py` is about
1,170 lines: extract compatibility/probe or checkpoint coordination only where it
materially improves readability, leaving admission rules explicit and shared
loading/prompt validation in one place. No registry, event bus or generic agent
framework is needed.

Run relevant base, native and Docker checks, wheel installation/fixtures/import
checks, and no-extra worker checks after changes. Preserve lazy model imports and
the optional dependency graph. Record the existing shared-lock FSSpec adjustment
and validate supported Python/platform profiles; a declared Python 3.11 minimum
is not evidence of an executed 3.11 training run. Automation is useful, but a new
CI service is not an original completion requirement.

**Done when:** a learner can reproduce the certified path from exact commands,
each capability has an evidence link, and no current-facing document calls
implemented native code absent or equates a fixture optimizer step with target
training acceptance.

## 4. Original optional Stage D work

### P4-11. Add the custom rollout/evaluation harness boundary

After the required native path works, add a minimal driver contract and explicit
configuration for a developer-selected trusted custom harness. Pass public tasks,
schemas, validated dispatch, read-only budgets, events and cancellation. Keep
workspace paths, private verification and reward ownership in the core owner.
In-process trusted Python is not sandbox isolation.

Exercise the same task/source/verifier semantics, bounded errors, cancellation,
traces and evaluator metrics as the built-in driver. Include harness identity in
declared evaluation profiles. Reject custom-harness training unless a separately
supported exact-token/probability/mask/current-weight contract is implemented.

**Done when selected:** a working trusted custom example runs real rollout/eval
through the four tools, passes parity checks, and truthfully reports `training:
unsupported`. CLI/config names for this extension are future implementation work.

### P4-12. Add CPU vLLM inference only after a measured compatibility gate

Resolve a CPU build for the actual host/checkpoint using official documentation
and installed probes at implementation time. No compatible build is asserted by
this plan. Isolate its lock/profile when Torch or Transformers requirements
conflict, and test incompatible extra/device rejection.

Add an inference backend boundary when there are two real implementations. Start
with one local request and bounded context; verify text loading, CPU operations,
memory, chat/tool parsing, cancellation, and final-source grading. Benchmark
against Transformers under declared profiles. Retain the working Transformers
path if the vLLM gate fails.

**Done when selected:** an actual CPU inference/evaluation run and capability
record exist, or the incompatibility is explicitly recorded. Native training
remains unavailable until live updated-weight synchronization, required
probabilities, IDs/masks and optimizer-to-generation refresh independently pass.
A CPU server plus GPU trainer is not established by configuration alone.

### P4-13. Track conditional extensions without turning them into release gates

| Extension mentioned in the original plans | Activation condition / required contract |
|---|---|
| OpenEnv or Gymnasium adapter (D3) | Concrete use case; thin wrapper over the same core, lifecycle/reward/isolation parity; no second verifier/environment |
| Environment-owned/self-sampled tasks | Explicit grouped task-selection key, independent resets, no duplicated dataset prompt or private-data export |
| CUDA training or lower precision | Tested hardware and wheel/index profile, incompatible-extra rejection, dtype operations, masks, memory, genuine update and reload gates |
| Longer training, different group/batch layout, gradient checkpointing or positive KL | A new recorded native profile; complete-group admission, finite tensors, reference/frozen-base semantics and memory/reload checks as applicable |
| Optimizer resume | Complete optimizer/scheduler/RNG state and fresh rollouts under the restored policy; weight loading alone remains evaluation |
| CPU-generation/GPU-training, vLLM training, async/distributed collection | Explicit probability and current-weight synchronization; independent group/state ownership; original supported path remains available |
| Packing, quantization or compilation | Architecture/device-specific forward/backward, sequence-boundary, probability and reload proof |
| More task families/variants | Immutable identities and family/generation-recipe split checks, independent evaluation partition; needed for generalization claims, not for four-task integration acceptance |
| Richer patches or `run_command` | Explicit new tool/path/permission/budget/source-manifest contract; do not loosen the four-tool/existing-file profile implicitly |

These items are recorded here so none disappear from the roadmap. Their absence
does not make the original optional feature a mandatory Phase 4 release blocker.
Databases, queues, a dashboard, Kubernetes and mandatory remote services are not
part of the original small local implementation.

## 5. Execution order and file-level handoff

1. Fix P4-01/P4-02 evidence and identity failures; complete focused negative tests.
2. Develop P4-03 cleanup/recovery and P4-04 capacity/fallback checks using small
   fixtures. Finish instrumentation that the real comparison will rely on.
3. Run P4-05 genuine development and declared held-out baseline after capacity
   checks. Keep tuning confined to training tasks.
4. Run P4-06 first-mode native pilot, then P4-07 real reload/evaluation.
5. Run P4-08 second mode with its own headroom and proof. Record unsupported hosts
   rather than declaring the missing mode complete.
6. Complete P4-09/P4-10 reports and reproducible handoff. Start optional P4-11–P4-13
   only after the required path is established and an extension is selected.

| Files | Work to carry into implementation |
|---|---|
| `config.py`, `cli.py`, `configs/` | Distinct probes/measured profiles, baseline path protection, future optional selectors only with actual implementations |
| `trainer.py`, `trl_env.py`, `trace.py` | Evidence validation, initial correlation, duplicate/error handling, rejected/native trace artifacts, feasibility and aggregate metrics |
| `training_worker.py`, `sandbox.py`, `environment.py` | Actual-stage interruption/fault coverage and consistent final recovery records; preserve the existing core and resource allowlist |
| `checkpoints.py`, `models.py`, `inference_worker.py` | Strong artifact schema, explicit model profiles/fallback, representative reload, both artifact kinds in the inference child, timing instrumentation |
| `evaluate.py`, `rollout.py`, `probe.py` | Verified baseline/after reports, complete denominators/means/costs, genuine integration and independent feasibility entry |
| `harness.py` and an optional new custom adapter | Preserve scripted reference-only behavior; add only the narrow eval driver boundary for P4-11 |
| Optional new CPU vLLM backend/profile | P4-12 only; isolated dependencies if needed, honest inference-only capability |
| `tasks.py`, `fixtures/`, `patches.py`, `verifier.py`, `worker.py` | Preserve working Stage A contracts; change only for a demonstrated defect or explicitly activated extension |
| `tests/` | New negative evidence/identity cases, actual lifecycle cancellation, genuine opt-in model gates; fixture tests remain separately labeled |
| `pyproject.toml`, `uv.lock`, `docker/`, docs | Supported dependency/platform checks, no-extra packaging, exact run instructions and retained evidence |

## 6. Phase 4 acceptance checklist

- [ ] The confirmed prerequisite/reset/duplicate-matching findings have regression
  coverage and an implemented resolution.
- [ ] Healthy invalid native calls and infrastructure failures have distinct,
  tested eligibility/reporting behavior; rejected groups retain diagnostic evidence.
- [ ] Actual native-stage interruption, RAM/deadline failure and checkpoint recovery
  leave no owned candidate resources or apparently usable partial saves.
- [ ] Independent model capacity reports and a real small fallback profile exist;
  each device/mode limitation is explicit.
- [ ] Genuine target CPU development read/edit/test/grading and the declared
  four-attempt held-out baseline are retained and internally validated.
- [ ] The first native parameter mode has a genuine informative update, exact
  learning evidence, updated-policy generation and completed checkpoint.
- [ ] That checkpoint reloads offline through the normal evaluator and has a
  matching before/after report with honest coverage and costs.
- [ ] The second full/LoRA mode passes independently, or its absence is explicitly
  recorded and the complete Stage C deliverable remains open.
- [ ] Planned metrics, current walkthroughs, measured profiles, capability table,
  and package/worker regression results are available with exact commands.
- [ ] Optional D1/D2/D3 work is either exercised with its own evidence or remains
  explicitly optional/unsupported; it is never presented as completed by this plan.

The required Stage B/C scope is complete only when its genuine model evidence
exists. A documentation update, successful API probe, fixture pass count, or
checkpoint directory by itself does not close that scope.
