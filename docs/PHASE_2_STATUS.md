# Phase 2 implementation and validation

Validation date: **2026-10-15**. Phase 2 corresponds to Stage B. The CPU backend, manual harness, and evaluator are implemented. A fully graded target-model episode and held-out baseline remain unexecuted because this validation host has no Docker CLI/worker. Do not treat synthetic test trajectories or the compatibility diagnostic as a model benchmark.

## Implementation

- Fixed the Phase 1 cleanup/cache defect: completed verification is retained, pending cleanup is retried, and one usable reward is published only after release. Trusted commit I/O failures report whether a revision committed.
- Added strict optional model, rollout, and evaluation configuration without changing the base installation requirements. Model and tokenizer revisions must be immutable commits. The supported profile is CPU FP32 Transformers; training, CUDA, and vLLM are outside this phase.
- Added one persistent spawned inference process. Only that process loads weights. Startup includes a finite forward pass and short warm generation before episode reset. Missing language weights fail loading. Requests contain primitive token IDs/masks and whitelisted generation settings, with stable derived seeds and no reusable generation cache between requests.
- Added the actual checkpoint's XML tool protocol, rather than assuming JSON tool output. A bounded fallback parser receives the actual non-thinking generation prefix, preserves raw and typed arguments, and accepts one call per turn. Unknown calls, booleans in integer fields, duplicate parameters, extra arguments, and multi-call proposals cannot dispatch partial effects.
- Added model episodes using public task projections, exact four-tool schemas, tokenizer-aware context reserves, bounded valid JSON observations, separate call counters, and explicit stop reasons. Truncated generations never execute a new action. Source, private grading, and cleanup still belong to the existing environment.
- Added schema-v2 traces with actual policy identity, raw generation IDs/text, parsing/dispatch records, shown observations, grading fingerprints, source artifacts, and nullable infrastructure outcomes. Scripted lessons remain labeled scripted. These artifacts are not native GRPO batches.
- Added sequential evaluation, stable per-attempt seeds, fresh episodes/conversations, per-task verdicts, explicit coverage/exclusions, and an atomic report after each attempt. Only genuine model outcomes enter model metrics. Verified horizons and incorrect candidates count in the denominator; infrastructure failures do not.

## Exact CPU profile

| Component | Selected value |
|---|---|
| Checkpoint | `Qwen/Qwen3.5-0.8B` |
| Model/tokenizer commit | `2fc06364715b967f1860aea9cf38778875588b17` |
| Text loader | `Qwen3_5ForCausalLM` |
| PyTorch | `2.14.1` |
| Transformers | `5.19.0` |
| Hugging Face Hub | `2.2.0` |
| Dtype/device | FP32 / CPU; four threads; one sequence |
| Template SHA-256 | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Host exercised | macOS arm64, Python 3.12.14 |
| Linux wheel selection | Explicit PyTorch CPU index through uv; not executed on this host |

The single `uv.lock` includes the optional stack. Docker's no-extra install continues to select only execution dependencies. Heavy model imports occur only in model operations. No Accelerate, vision application stack, TRL, optimizer, or remote-code loading is required.

The model's [official loading guidance](https://huggingface.co/docs/transformers/model_doc/qwen3_5#usage-tips-and-notes) informed the text loader; the actual downloaded tokenizer template established the XML protocol. CPU source selection follows [uv's PyTorch guidance](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Executed validation

| Check | Result |
|---|---|
| Base/model-free regression and fake-backend controller tests | 112 passed; eight Docker tests skipped |
| Real spawned-process deadline, crash, and mismatched-response checks | Passed without model libraries |
| Cleanup failure after a correct verdict, then retry | Stable verdict, no rerun, release before exactly one reward |
| Ruff lint/format and `git diff --check` | Passed |
| Offline uv lock check | Passed |
| Optional rollout extra installation | Passed on Python 3.12 macOS arm64 |
| Installed wheel with rollout extra outside checkout | Clean install, dependency versions, architecture import and pinned tokenizer/template admission passed |
| Source distribution and wheel build | Passed |
| Base wheel import without Torch/Transformers | Passed in a fresh temporary environment |
| Installed base wheel outside checkout | Nine authored smoke cases passed |
| Genuine target CPU generation, parser, and read dispatch | Passed; [saved evidence](evidence/phase2-cpu-read.json) |
| Default real rollout preflight | Correctly reports Docker unavailable with no reward |
| `trusted_local` with model rollout | Rejected before model generation |

The fake-backend tests exercise accepted authored repairs, positive/negative independent verification, malformed and multiple calls, wrong argument types, context/token/tool horizons, cancellation, backend crashes, recorder errors, cleanup, fresh history, and honest metrics. These tests explicitly label their driver `test`; the evaluator rejects them as baseline inputs.

## CPU compatibility diagnostic and remaining gates

`model-probe` loads the target and requests a generated `read_file` call. It dispatches only a source read and never executes candidate Python or awards correctness. Full results, raw token IDs, loading diagnostics, timing, and peak resident memory are saved in `outputs/trajectories/model-probe.json`.

The initial 90-second generation diagnostic hit its limit after successful text loading and warming. A 300-second diagnostic then generated a genuine read call in 135.49 seconds (28 completion tokens, about 0.207 tokens/second including prefill), with peak RSS 4,872,667,136 bytes. Startup was 10.93 seconds, including 2.03 seconds loading and 5.71 seconds warming. All missing/unexpected/mismatched language weight sets were empty.

That generation exposed a checkpoint/text-config EOS mismatch: the tokenizer ends turns with `<|im_end|>`, while the text generation config continued to `<|endoftext|>`. The parser now accepts the observed terminal suffix and the worker explicitly uses the tokenizer's EOS/pad IDs. A fresh post-correction diagnostic **passed**: 687 input tokens, 26 completion tokens, 140.80 seconds generation, 12.72 seconds startup, and peak RSS 4,961,746,944 bytes. The generated `read_file("timeout.py")` parsed and executed successfully. [The saved real probe](evidence/phase2-cpu-read.json) preserves the actual raw text, IDs, identity, loading diagnostics, and public observation. The supplied CPU profile now allows 300 seconds per generation and 900 seconds per episode; these are limits, not runtime promises.

First-token latency and decode-only throughput are not instrumented; reports use `null` for these rather than deriving them from total generation time. Per-generation throughput includes prefill.

The following acceptance gates still require execution before declaring Phase 2 fully validated:

1. A complete model-generated read/edit/test episode under the real Docker worker, with frozen source and an independent verdict.
2. A fixed held-out baseline with declared sampling, measured full-episode costs, coverage, and infrastructure exclusions.
3. The CPU profile on Linux and the corresponding CPU wheel/runtime gate if that platform is used.

No held-out pass rate has been published. No model repair is claimed from reference-driven smoke tests.

## Commands and artifacts

The [README](../README.md#cpu-model-rollouts-and-evaluation) contains installation, immutable snapshot download, probe, rollout, and evaluation commands. Use `uv run --extra rollout` for model commands. The default profile loads offline; set `model.offline: false` explicitly only when preparing through the backend instead of the Hub CLI.

Each run saves resolved configuration under its trace directory. Episode files contain the conversation, raw generation events, frozen source/diff, and final outcome. `eval` saves the configured JSON report after each attempt and does not retry model sampling to seek a better answer. A generation horizon can force a model reload before the next attempt; startup/reload measurements remain separate from episode time.

Schema v2 publishes `verify`, then `cleanup`, then the usable `reward`. Consumers of Stage A's schema v1 should account for this intentional ordering/schema change. Correctness, stop reason, and completed run status are independent fields.

Phase 3 still needs native TRL environment integration, same-task grouping, action masks/probabilities, truncation alignment, and a genuine informative update with changed weights.
