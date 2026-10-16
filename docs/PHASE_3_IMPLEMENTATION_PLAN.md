# Phase 3 implementation plan

Implement the contracts in `PHASE_3_NATIVE_TRAINING_DESIGN.md` using the existing
environment as the authority for source changes and rewards. Keep optional model
libraries out of base imports. Do not implement another rollout engine or GRPO.

1. **Compatibility first.** Inspect a released TRL package, constructor/config
   signatures, native environment discovery, parser, token limits, and public
   extension points. Resolve a CPU extra without changing the rollout pins.
   Retain a machine-readable compatibility report. Missing safe admission points
   must prevent optimizer execution, rather than become undocumented assumptions.
2. **Configuration and identities.** Add strict native settings separately from
   manual rollout settings. Start with CPU FP32, four attempts, one device and one
   update per collection. Represent Hub, full, and LoRA sources explicitly.
3. **Adapter.** Export public training prompts with system instructions and allowed
   paths once. Expose four typed tools and reserved lifecycle methods only. Reuse
   strict validation and bounded observations. Reject evaluation tasks and invalid
   row identities; latch tool, grading, and cleanup infrastructure failures.
4. **Run ownership and admission.** Correlate collection/policy/task/attempt identity.
   Reject incomplete or corrupted groups before learning. Use supported native
   extension points for actual sequence/mask/probability checks; gate unsupported
   guarantees. Own workspaces and labelled Docker workers from the parent before
   launch, and clean only those resources after child termination.
5. **Native execution.** The training child owns the sole model and optimizer.
   Record measured forward/backward feasibility, native configuration translation,
   group outcomes and update evidence. Bound the pilot to eight groups. Preserve
   equal-reward groups as reported zero-contribution collections. An informative
   update needs genuine mixed correctness rewards, finite gradients, changed
   intended parameters, and subsequent generation using that updated policy.
6. **Artifacts and evaluation.** Stage checkpoint files, verify content and reload
   metadata, then atomically publish a completion manifest. Load local full/LoRA
   artifacts through the existing inference worker while retaining the evaluation
   tokenizer/template. Compare evaluation profiles independently of policy identity
   and output paths, with explicit valid-attempt denominators.
7. **Validation and handoff.** Run the existing base suite, focused adapter,
   ownership, group failure and checkpoint tests, Ruff, and wheel checks. Native
   model and real Docker checks are opt-in. Record actual results and unresolved
   B-close/C1–C7 gates; do not report fixture tests as a real training proof.

Implementation order follows these dependencies. Keep audit logic in `src/coding_rl/training/trainer.py`
until splitting it makes the code clearer; use small typed outcome records and
explicit composition rather than a framework of trainer abstractions.
