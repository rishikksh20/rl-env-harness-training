# Phase 4 implementation and measured status

Updated **2026-10-16**. Use [the training guide](TRAINING_GUIDE.md) for commands.
[The Phase 4 plan](PHASE_4_COMPLETION_PLAN.md) remains the dated audit/checklist.
Implementation and empirical learning acceptance are separate.

## Delivered code

| Plan area | Delivered behavior |
|---|---|
| P4-01 | Recompute evaluation coverage/metrics from scheduled attempts; reject duplicates, incomplete coverage, inconsistent task/policy/image identities, and reward-less development. Cross-check development trace, accepted edit, healthy tests, frozen source, and verifier. Protect baseline output paths. |
| P4-01/P4-07 | Checkpoint schema 2 validates immutable identities, trainable parameters, finite positive update proof, step/parent/generation links, reload tolerances/IDs, and file digests. Portable native/tensor/probability audit files are included. Schema 1 is explicitly rejected. |
| P4-02 | Assign native policy identity before reset; match duplicate IDs using masks/advantages once each; check tool defaults/extra arguments; admit reproducible pre-dispatch errors while preserving failure latches and cutoff rejection. |
| P4-02/P4-09 | Retain raw available collections before admission, rejection reasons, trace-directory references, exact admitted IDs/tensors/probabilities, and aggregate reward variance/statuses. Unsupported native hook counters remain unavailable. |
| P4-03 | Continue owned cleanup after Docker-control failures, retaining a retry registry while removing writable state/staging. Recover malformed result/ledger failures as incomplete. Extend actual native interruption and checkpoint-preservation tests. |
| P4-04 | Explicit Qwen/LFM strategies cover loading, template, parser, EOS/pad, and reload. Supervised `train --feasibility` works before baseline admission: four-sequence forward/backward, every present gradient, diagnostic Adam allocation, and save/reload. |
| P4-09 | Measure first-token/subsequent-decode timing through the Transformers streamer. Add episode/token means, parser/rejection rates, per-task costs, and native summaries. |
| P4-10 | Model preparation/task commands, LFM/OpenCode profiles, detailed execution guide, capability table, and retained validation results. |
| Selected P4-11 | OpenCode/custom trusted subprocesses use a local text-chat/four-tool MCP bridge. Controller owns grading/lifecycle. Harness identity enters evaluation profiles; external outcomes cannot satisfy native prerequisites. |
| Coding curriculum | `dataset-import`, `task_root`, and public-only exports support local Python repair catalogs with protected JSON cases and no reference repair requirement. |

New modules isolate model strategies (`src/models/profiles.py`), evidence validation
(`src/coding_rl/training/evidence.py`), disposable capacity measurements (`src/coding_rl/training/feasibility.py`), dataset import
(`src/tasks/dataset.py`), local protocols (`src/harnesses/bridge.py`), and external ownership
(`src/harnesses/external.py`). They reuse the existing environment/sandbox/verifier and
native GRPO loop. No web framework, MCP SDK, or inference-server dependency was
added.

Both LFM evaluator and trainer normalize wrapped schemas to its flat tool format
and record actual/Hub template hashes. The upstream lax argument parser preserves
escaped newlines in single-quoted patches; the public parser-registry extension
uses literal AST decoding of the complete call list. Generated Python is never
executed during parsing.

## Measurements and evidence

See [the validation record](evidence/phase4-validation.json). Raw local runs are
under `outputs/`. The real policy was **LiquidAI/LFM2.5-350M** at
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f`, CPU FP32, four threads, on a host with
about 31.1 GiB physical RAM and 12 logical CPUs.

The full native/Docker/bridge regression run passed **183 tests with no skips**.
Ruff, offline wheel/source builds, guide shell/Python syntax, and the documented
dataset import also passed. A targeted ownership rerun covers the final malformed
registry error classification.

- A genuine model probe emitted and executed `read_file`, retaining IDs, loading
  information, first-token latency, and subsequent-decode timing.
- LoRA capacity exercised 354,656,000 total and 172,032 trainable parameters. A
  four-sequence layout of 967 tokens peaked at approximately 10.4 GiB RSS.
  Representative forward/backward, diagnostic Adam allocation, and
  save/reload passed. A larger layout hit its RAM cap and was terminated without
  leaving a usable partial checkpoint.
- Genuine manual/OpenCode development episodes ran. The small model attempted an
  invalid patch, then claimed success. Visible tests passed; independent grading
  returned 0. These samples do **not** satisfy the accepted-edit development gate.
- A fixed four-attempt genuine held-out baseline completed with four valid
  attempts and zero verified successes. This proves coverage/failure scoring,
  not general coding ability.
- Installed tiny Qwen/LFM tests use authored completions with actual TRL dispatch,
  tensor/probability audits, backward, and optimizer operations. Full/LoRA reload,
  normal tool errors, native interruption boundaries, and preservation of a
  committed checkpoint during a later interrupted real save are integration
  evidence, not pretrained learning evidence.

## Capabilities and remaining gates

| Capability | Status |
|---|---|
| Model-free core and Docker verification | Implemented and regression tested |
| Manual CPU inference/evaluation | Implemented; genuine LFM runs measured |
| LFM native CPU LoRA capacity | Measured for the reported layout; rerun for your config |
| Qwen/LFM native GRPO | Installed fixture-model tests; genuine informative pretrained update remains open |
| Portable full/LoRA checkpoints | Both fixture-tested with real weights/optimizer/reload |
| OpenCode local model + MCP | Exercised with OpenCode 1.15.13 and genuine LFM generation |
| Custom subprocess harness | Standard-library example and bridge/subprocess regression tests |
| OpenCode/custom external training sampler | Unsupported; native TRL owns training sampling |
| Custom functional Python repair datasets | Import/public-private separation implemented and tested |
| SWE-Gym/SWE-bench repository environments | No adapter; require richer environment/grading support |
| CUDA, CPU vLLM, OpenEnv/Gym, resume, distributed/long runs | Optional/conditional roadmap items; unsupported profiles |

P4-05's accepted-edit development gate remains open for the measured LFM policy.
Native admission correctly refuses these samples. P4-06's informative pretrained
update, P4-07's genuine checkpoint comparison, and P4-08's independent second-mode
learning evidence therefore remain open. Qwen pretrained capacity/learning was
not run in this delivery. No oracle or invented reward closes those gates.

Graceful interruption is tested at actual native generation, visible-test, and
grading boundaries; a real later interrupted save preserves the earlier committed
policy. A live Docker worker is forcibly terminated and cleaned. Forced OS kills
at every pretrained lifecycle stage and the precise post-publication/pre-IPC stop
case are not exhaustively certified by these tests.
