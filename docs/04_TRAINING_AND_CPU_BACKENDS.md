# Training, CPU rollouts, and compatibility gates

Design draft dated **2026-10-14**. This document describes intended implementation and acceptance checks; it does not claim that packages, model loading, CPU throughput, or GRPO training have been tested in this repository. Official documentation and upstream source were reviewed on that date. References to moving `main` branches establish current documented behavior, not a dependency lock.

Read [the architecture](ARCHITECTURE.md) and [environment integration](ENVIRONMENT_SUPPORT.md) first. The environment is the lesson: one task, one clean workspace, actions, observations, independent verification, then a terminal reward. Training adds a standard optimizer around that lesson.

## 1. Choose the smallest training path

Use Hugging Face TRL's `GRPOTrainer` for sampling, probabilities, masking, group advantages, loss, and optimizer steps. Implement only the coding world, sandbox interaction, verifier, trace recorder, and a thin trainer setup. The initial trainable path is native `environment_factory` with Transformers generation on the same device as the training model. The initial user default is training disabled and CPU rollout/evaluation mode.

Keep three different milestones visible:

| Milestone | What actually runs | Evidence required |
| --- | --- | --- |
| CPU environment smoke | Scripted actions, workspace transitions, visible tests, hidden verification | No model download; passing isolation and reward checks |
| CPU model rollout | A real checkpoint generates actions; the harness executes tools | Parsed tool calls, recorded observations, final candidate and verifier result |
| Genuine GRPO training | Real policy generations, rewards, backward pass and optimizer update | Finite gradients and a changed trainable parameter, then a reloadable checkpoint |

An inference server produces actions. It does not by itself perform RL. Selecting a CPU inference backend also does not establish a CPU/GPU training combination.

## 2. Current official API checkpoint

The reviewed GRPO documentation describes `environment_factory` for stateful rollouts, `tools` for standalone functions, and optional external dataset metadata passed into `reset`. Only intended tool methods should be public; `reset` and `get_reward` are reserved. The documented integration requires Transformers at least 5.2.0. The trainer owns the tool loop. Its documented custom `rollout_func` requires prompt IDs, completion IDs and log probabilities and is experimental. [Official GRPO trainer documentation](https://huggingface.co/docs/trl/en/grpo_trainer#agent-training).

The default adapter design is:

1. Supply a no-argument factory that captures trusted task and sandbox configuration.
2. Preserve the dataset's `prompt` and opaque `task_id`; forward task metadata into the adapter's reset operation.
3. Return no extra prompt from reset in dataset mode.
4. Expose exactly `list_files`, `read_file`, `apply_patch`, and `run_tests`, with typed arguments and helpful descriptions.
5. Freeze the final candidate in `get_reward`, cache a completed verification result, and release episode resources in a guaranteed cleanup path.

The upstream trainer source currently warns that `environment_factory` is also experimental. Its response parser dependencies vary with the Transformers version. Treat both as items in the pinning probe, rather than presenting a minimum version as a complete working stack. [Official trainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py).

Environment-owned task sampling is optional later. Dataset-driven tasks make same-task groups and held-out partitions easier to inspect. Reused adapter objects must clean stale state before every reset; episode IDs, clean files, worker ownership, counters, caches, and terminal status must be new even when a Python object survives between trainer steps.

## 3. Mental model: a reward belongs to one attempt

For one task, sample four independent attempts from the same current policy. Each begins from identical starter bytes and gets its own writable episode workspace. Each can read, edit, test, revise, and finish differently. A trusted controller verifies each final candidate independently and supplies one reward.

The core reward rule stays binary: **1.0 for a completed, verified-correct candidate; 0.0 for an evaluated incorrect candidate**. No reward is added for a confident final message, number of tool calls, visible-test success, or formatting. Keep diagnosis separate from the numeric reward.

For teaching, the group-relative advantage can be written as

\[
A_i = \frac{r_i-\overline r}{\sigma_r+\varepsilon}.
\]

The same-task group supplies a comparison baseline instead of a separately trained critic. TRL performs the actual calculation with its pinned reward-scaling and loss settings. This is the GRPO idea introduced in the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300).

| Four terminal rewards | Useful interpretation |
| --- | --- |
| 1, 0, 1, 0 | Successful attempts get positive relative advantage; unsuccessful attempts get negative relative advantage |
| 0, 0, 0, 0 | No within-group reward difference; this group's task-reward signal cannot distinguish its actions |
| 1, 1, 1, 1 | All succeeded; no within-group difference either |

Do not manufacture a successful reward to force an update. Initial tasks should be small enough that the model sometimes succeeds, while retaining real hidden verification. If every group fails, inspect task difficulty, parser success and tool budgets before changing the reward. If every group succeeds, add genuinely harder training tasks. With nonzero KL or other auxiliary losses, parameters may still move despite zero task-reward advantage; that does not demonstrate learning from correctness.

## 4. What a training batch contains

A task is a stable specification. An episode is one attempt. A trajectory is the history of that attempt. A GRPO group contains several trajectories for the same task. A training batch contains complete groups, not arbitrary lines taken from a trace.

Let the number of distinct tasks per generation batch be B, and attempts per task be G. There are B × G episode states. Four attempts for one task is a useful first integration target. Distinct episode IDs must retain the same task ID and starter digest; workspace paths and container IDs must differ.

The current configuration source requires an effective batch divisible by `num_generations`; it also distinguishes `generation_batch_size` from `steps_per_generation`. These settings are mutually exclusive. Reward scaling, loss type, and truncation masking are independent choices. A zero `beta` avoids loading a reference model; a positive value activates KL regularization. [Official GRPO configuration source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py).

For the first single-process probe, propose G = 4, one sequence per forward microbatch, four accumulation microsteps, one generation batch of four sequences, and one optimizer update. This is a candidate layout to validate against the installed constructor and sampler; it is not a claim that every TRL release accepts that combination. Print the resolved batch layout before generation. Keep a single-process run until complete-group ordering and reward alignment are proven.

The training record needs exact model-token history:

| Data | Role in the method | Owner |
| --- | --- | --- |
| Task prompt and tool schemas | Initial conditioning context | Dataset plus adapter/template |
| Generated assistant tokens | Sampled policy actions and final response | Trainer's generation path |
| Tool result tokens | External feedback conditioning the next action | Environment, encoded by trainer |
| Token IDs and positions | Exact sequence used for probabilities | Trainer |
| Action/loss mask | Select which sampled tokens contribute to policy loss | Trainer |
| Behavior policy probabilities | Identify how the sampled actions were produced | Supported generation/trainer contract |
| Current policy probabilities | Differentiate the objective | Trainer forward pass |
| Terminal verifier reward | Score the whole attempt | Verifier through adapter |
| Human-readable JSONL trace | Explain and debug the episode | Recorder |

Do not rebuild training sequences by decoding text and tokenizing it again. Chat formatting, delimiters, special tokens, and model-generated reasoning can change. The trace is a readable audit artifact; it does not replace the tensor-level training record.

The inspected trainer source distinguishes model tokens from tool suffix tokens and combines that mask with completion validity for loss. Its experimental custom path also recognizes an external-feedback mask. These are source-level observations to recheck in the installed version. [Official token-mask implementation](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L2080).

A mask audit should demonstrate that generated tool names, arguments, patch text and assistant responses are eligible policy tokens; user instructions, file contents returned by tools, test output and padding are not policy targets. External feedback still influences later predictions through context. Do not apply supervised next-token loss indiscriminately to the entire conversation.

## 5. Freeze policy versions and admit complete groups

Use one policy version to sample a complete generation batch. Record the model/checkpoint revision, adapter revision, optimizer step, tokenizer/template identity, task revision and sampling settings with every group. Verification must finish before that group's advantages are accepted for an update. The first release uses one update per sampled batch and no asynchronous rollout pipeline.

Classify failures before assigning reward:

| Outcome | Numerical treatment | Reason to preserve |
| --- | --- | --- |
| Hidden assertion fails | Valid 0.0 | Candidate is incorrect |
| Candidate exceeds its execution limit | Valid 0.0 | Candidate timeout, if trusted startup checks passed |
| Rejected path, invalid patch or allowed tool misuse | Recoverable action error; consume its call budget and grade the final valid snapshot | Invalid action has no side effect; it need not make a later correct solution fail |
| Budget termination with a valid frozen candidate | Independently verify it; allow 1.0 only if verified correct | Tool, token, context, iteration or episode deadline; mark limited-but-verified explicitly |
| Missing fixture/image, daemon failure, host memory failure or verifier-controller exception | No correctness label | Infrastructure failure |
| Cleanup failure | Record and investigate separately | Resource leak; do not disguise it as a candidate mistake |

For infrastructure failure, first retry the same frozen candidate under a fresh verifier worker with a bounded retry count. No model regeneration is needed when only grading failed. If correctness remains unknown, the smallest safe native strategy is to abort the generation batch before optimizer mutation and resume from the previous committed checkpoint. Discard or retry complete groups together when the installed trainer explicitly supports that operation. Never drop one failed member and quietly compare the three remaining members as the intended four-attempt group.

The initial adapter should propagate a typed infrastructure exception rather than returning 0.0, NaN, or a placeholder reward. Verify that the pinned native path propagates it before updating. If it converts the exception into a scalar failure or partial update, training readiness fails until a supported batch-level handling mechanism is established. A reward-function convention for an inapplicable score is not evidence that a failed verifier can be skipped safely.

Trusted hidden assertions remain in the controller outside candidate execution. That controller calls a narrow JSON worker in a fresh candidate container, compares returned behavior to expected results, and stores trusted grading status. Candidate source and all returned values are untrusted. Transfer only allowlisted candidate file bytes onto an independent clean starter. Keep private assertion text and expected values out of rollout observations and public traces.

## 6. Budget mapping must be explicit

These are proposed educational defaults, not direct TRL arguments. The separate per-turn and episode-generated token counts are manual-harness proposals; the native path must report its actual supported accounting:

| Budget | Proposed default | Meaning and enforcing owner |
| --- | --- | --- |
| Attempted tool calls | 6 | Manual dispatcher counts proposals; core counts entered invocations and rejects effects after exhaustion; complete native accounting needs a verified dispatch hook |
| Tool-calling iterations | 6 | Limit assistant/tool cycles separately; trainer or manual harness |
| Generated tokens per assistant turn | 256 | Cap each model continuation; generation path |
| Generated tokens per episode | 768 | Sum model-generated continuations across the episode; generation path |
| Total context | 2048 tokens | Initial prompt, schemas, assistant history, observations and required delimiters; tokenizer-aware generation path |
| Candidate command timeout | 20 seconds | Visible/verification worker execution limit; sandbox |
| Episode wall time | Calibrated and bounded | Entire inference/tool attempt; episode supervisor |
| Tool observation bytes | Explicit small cap | Bound file/test feedback before templating; core |

A turn can contain several tool calls, so six iterations do not enforce six attempted calls. A character or byte cap is not a token cap. A context check must include tool schemas and formatting overhead, and reserve room for the next generation.

Unknown tools and malformed arguments can be rejected by native TRL before the core sees them. Verify supported dispatch/error metadata before claiming an all-attempt native counter. Without it, document the narrower core-invocation cap and separately bounded native parser/iteration failures.

**Native integration gate:** inspect what `max_completion_length` counts in the pinned tool loop. The reviewed implementation limits the accumulated completion region, including tool suffixes; it can execute a tool and then remove overlong feedback from the recorded conversation. Environment mutations are not automatically undone by removing messages. [Official tool-loop source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1788).

Consequently, do not translate “256 tokens per assistant turn” directly into the native completion limit and claim the resulting six-turn episode matches these defaults. Establish exact, supported mappings for the three token budgets. Admit only tasks whose initial rendered prompt fits. Limit tool output before return. Demonstrate that a side-effecting action cannot survive while its model history is discarded. If independent limits lack supported hooks, configure the native bounds it actually provides and document the different accounting. Fail closed or abort the complete generation group before update if mutation/history consistency cannot be guaranteed. Masking tokens alone does not prove consistency. No internal monkeypatch or second hidden training agent loop should bypass this gate.

The manual evaluation harness can enforce the distinct budgets directly before each generation and tool action. The native path needs its own compatibility probe. An episode deadline checked only at tool entry does not interrupt a long model generation; an outer supervisor must be able to stop the trainer process safely and preserve the last committed checkpoint.

Define truncation before training: freeze the last valid candidate on a final response or enforced limit and independently verify it. Termination and solved are separate fields; a limited but verified-correct attempt can receive 1.0 and must be labeled accordingly. Never apply an incomplete or invalid patch. Separately decide whether incomplete trajectories contribute to loss. The initial conservative proposal masks incomplete trajectories out of policy loss when the verified pinned API provides correct termination detection. Report excluded tokens and zero-contribution groups rather than silently treating a masked batch as a successful learning update.

## 7. Model loading and tool parsing gates

The target is `Qwen/Qwen3.5-0.8B`, a post-trained checkpoint with a vision encoder and hybrid language backbone. Its official card documents `qwen3_coder` for vLLM tool parsing, a language-model-only serving option, and warns that the 0.8B model can loop in thinking mode. These serving examples do not establish CPU training compatibility. [Official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).

Transformers documents `Qwen3_5ForCausalLM` with the text configuration for text-only generation. Optional DeltaNet kernels accelerate the hybrid layers; without them, the implementation can fall back to slower, more memory-hungry PyTorch operations. [Official Qwen3.5 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5#usage-tips-and-notes).

Prefer that documented text-only loading route, then validate the actual checkpoint-to-text-class mapping: config extraction, weight-key mapping, tied embeddings, tokenizer, generation configuration and absence of unintentionally random parameters. Inspect missing/unexpected weight reports. Do not assume generic automatic causal loading strips the vision encoder correctly. If the pinned release cannot load the documented text path from this checkpoint cleanly, use a documented full conditional-generation path only after its text-only forward and GRPO contracts pass; record the additional footprint.

Use non-thinking mode initially and keep it fixed across baseline, rollouts and training. Record its template setting rather than removing reasoning tags after generation. Current TRL ships Qwen3.5 training templates that preserve conversation prefixes, including a non-thinking flavor for smaller checkpoints. [Official TRL chat-template guide](https://huggingface.co/docs/trl/chat_templates#qwen3_5_think_trainingjinja--qwen3_5_nothink_trainingjinja).

A rendered conversation must retain identical earlier token IDs after tool feedback is appended. Test the actual selected template, not merely the original model card. Parser checks must cover a file read, a multiline patch, escaped quotes, invalid arguments, unknown tool names, multiple proposed calls and a normal terminal assistant response. Keep observations bounded and the four tool schemas brief enough for a small checkpoint to use.

Use `Qwen/Qwen3-0.6B` as a proposed fallback for isolating library integration; it is an [upstream environment example model](https://huggingface.co/docs/trl/en/grpo_trainer#environments). It still needs local parser/forward checks and is not a promised coding-task solver. Do not confuse a random tiny model used for tensor-path tests with the real checkpoint acceptance run.

## 8. Hybrid layers require sequence-isolation checks

Start with independent padded sequences and packing disabled. Do not add an invented GRPO `packing` parameter: absence of packing is a design choice and a check on actual tensor layout.

The official padding-free guide warns that Qwen3.5's Gated DeltaNet cannot infer packed boundaries from `position_ids` alone. It requires boundary metadata including `seq_idx` and compatible kernels; reference fallbacks can ignore those boundaries and silently mix samples. [Official Transformers padding-free guide](https://huggingface.co/docs/transformers/padding_free#linear-attention-and-convolution-models).

This is a correctness gate, even if generation appears fluent. For the chosen device, dtype and kernel path:

1. Score a short sequence alone and in a padded batch; compare non-padding token logits with documented numerical tolerance.
2. Change another batch member drastically; the first member's logits must remain unchanged within that tolerance.
3. Reset generation state and repeat an episode; cached recurrent state must not leak into the next attempt.
4. Run a finite forward/backward pass and check gradients under the actual training configuration.
5. If packing is ever added, compare packed versus independent outputs and gradients with explicit boundaries across both full-attention and DeltaNet layers.

Passing padded-batch checks does not certify packed batches. GPU kernel availability does not certify CPU kernels. Keep compilation, continuous batching and specialized packing outside the first vertical slice.

## 9. CPU expectations and benchmark procedure

An approximately 0.8-billion-parameter text model needs roughly 3.2 GB for FP32 weights or 1.6 GB for two-byte weights, using parameter-count arithmetic and decimal units. This is a planning estimate for language weights only. The full checkpoint, activations, hybrid caches, tensors, allocator overhead and operating system require more memory. Full training also adds gradients and optimizer state. A model that fits for inference may not fit for training.

Start with plain Transformers CPU inference, one episode at a time, a short context, and an explicitly chosen dtype. FP32 is the initial correctness baseline. Try BF16 only if the hardware and chosen PyTorch operations support it and measurements show acceptable quality and performance. A smaller numeric representation does not guarantee faster CPU execution. Keep unverified quantization and GPU-specific kernels outside required CPU dependencies.

Run a bounded benchmark before planning a training experiment:

1. Record processor model, physical cores, instruction support, RAM, OS, Python/library versions, dtype and checkpoint revision.
2. Time model load and a small warm-up separately.
3. Measure first-token latency, generated tokens per second and peak resident memory on fixed short prompts.
4. Run one read/patch/test episode with the real tool schemas; count parse failures and wall time as well as tokens.
5. Run four independent attempts for one task, sequentially first; measure complete-group cost.
6. On a supported training device, time generation, verification, scoring, backward and optimizer phases separately for one real group.
7. Estimate a longer run from measured groups and report the estimate's assumptions.

Inference time is approximately rendered-prompt work plus decode work over all turns; repeating long histories makes prefill significant. Training cost includes those rollouts plus probability scoring, gradients and optimization. A 20-second candidate-test timeout is not a reasonable universal model-inference deadline; calibrate those limits separately.

## 10. Optional vLLM CPU backend

vLLM provides CPU inference builds with platform-specific requirements. Its x86 instructions specify Linux, recommend AVX-512, and label AVX2 support as limited. Use the appropriate CPU wheel/build, not an assumed accelerator wheel. [Official x86 CPU installation source](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/cpu.x86.inc.md).

The current CPU validation matrix lists tested model/hardware combinations but does not list Qwen3.5. That absence is evidence of missing published validation in that matrix, not proof that the architecture is impossible on CPU. [Official CPU validated-model matrix](https://github.com/vllm-project/vllm/blob/main/docs/models/hardware_supported_models/cpu.md).

Make this a separate optional inference environment so its PyTorch constraints do not destabilize the training stack. Enable it only after text-only checkpoint loading, CPU dtype, short bounded context, tool parser, cancellation, candidate verification and memory limits all pass on the actual host. Start with one concurrent request. CPU core binding and cache allocation affect service behavior; reserve capacity for the controller and workers. [Official vLLM CPU runtime guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#related-runtime-environment-variables).

TRL's documented vLLM training integration uses generation endpoints plus updated-weight transfer; the reviewed server path uses CUDA/NCCL and processed sampling log probabilities. A chat-completions endpoint alone is insufficient. [Official TRL vLLM integration](https://huggingface.co/docs/trl/vllm_integration#how-trl-uses-the-server-).

Therefore CPU vLLM is rollout/evaluation-only initially. To promote any custom service or external harness to training, prove all of these together:

- Identical tokenizer, special tokens, prefix-preserving template and exact prompt history.
- Exact generated token IDs, aligned external-feedback mask, termination metadata and probabilities under the correct sampling distribution.
- Complete same-task group ordering and correct reward-to-trajectory association.
- Current policy weights, including active LoRA state if used, synchronized atomically before sampling; version identity recorded.
- Correct current/reference probability scoring and the trainer's supported mismatch correction.
- Failure, cancellation and retry behavior that preserves complete groups and avoids partial updates.

Compatibility on one device pair does not imply another. Keeping a service's weights frozen after training starts makes its samples stale. Do not turn an evaluation trace into an on-policy training batch by adding guessed probabilities.

## 11. Full training, LoRA, KL and memory choices

Expose full-parameter training and LoRA as alternatives. Full training is conceptually simple and helps reveal total memory cost. LoRA trains a small set of adapter parameters while retaining a frozen base. Gradient checkpointing trades extra computation for lower activation memory. [Official TRL memory guide](https://huggingface.co/docs/trl/en/reducing_memory_usage#peft-for-parameter-efficient-fine-tuning).

For LoRA, inspect actual Qwen3.5 module names and select documented trainable linear projections; do not copy a target-module list from an unrelated architecture. Record adapter rank, scaling, targets, trainable parameter count and base revision. Verify generation sees the active adapter and a saved adapter reloads with the exact base. A smaller optimizer state does not remove model inference, activation or backward costs.

Start `beta` at 0.0 for the minimal lab, with one update per sampled batch and no extra text-format reward. Pin the explicit loss variant and scaling choice after the compatibility probe so a future library default does not alter the experiment. If positive KL is added, define the fixed reference checkpoint and verify its behavior, memory footprint and adapter handling. Reference-policy behavior and rollout-weight synchronization are separate responsibilities.

Avoid bundling quantization, packed sequences, custom inference synchronization, compilation and LoRA into the first training proof. Validate one working full or LoRA mode first, then enable the second with the same mask, reset and reward gates.

## 12. Dependency pinning and training readiness

Choose Python 3.11 or later from the versions mutually supported by the candidate training packages. Separate environment-only dependencies from model/training dependencies and optional vLLM extras. Core dependencies include the fixture test runner and config reader; real training adds PyTorch, Transformers, TRL, Datasets and Accelerate, plus PEFT only for LoRA. A model download and Docker image preparation happen before offline episodes; candidate containers require no network.

Use this sequence to create a verified lock during implementation:

1. Select candidate released packages satisfying the documented API floor and Qwen3.5 architecture needs. If a feature needs unreleased code, pin an exact reviewed commit and explain it.
2. Resolve dependencies in a fresh environment for the intended hardware; capture the package graph and reject conflicting constraints.
3. Inspect the installed trainer/config signatures, response-parser dependencies, model classes and tool-discovery behavior.
4. Run the framework-only probes, then Qwen3.5 loading/parser/forward probes, then one environment group.
5. Run a genuine nonzero-variance model group through one optimizer update and save/reload the result.
6. Pin exact package and checkpoint revisions, container image digest and template identity only after those gates pass.
7. Record outcomes by mode: environment smoke, CPU inference, native training, LoRA and optional CPU vLLM each get their own status.

Do not publish “tested TRL version” until an actual run supplies it. A source review, import test, mocked optimizer, or scripted successful patch is insufficient.

| Readiness gate | Acceptance evidence |
| --- | --- |
| Environment contract | Dataset metadata reaches reset; tool list is exactly four; private methods stay undiscoverable |
| Group integrity | Four identical starter digests, four independent workspaces, correct reward order |
| Model contract | Correct text-path weights, real tool parsing, stable template prefixes |
| Sequence contract | Independent padded forwards, finite gradients, no cache leakage |
| Budget contract | Verified per-mode semantics; no mutation/history mismatch; bounded cancellation |
| Reward contract | Independent controller verification, idempotent snapshot, no public hidden assertions |
| Failure contract | Failed infrastructure cannot silently become a valid 0.0 or partial group update |
| Optimizer proof | Real generated attempts yield task-reward variance, finite gradients and changed trainable tensors |
| Checkpoint proof | Saved full model or adapter reloads and evaluates with recorded task/template revisions |

Evaluation uses held-out tasks and fixed declared budgets before and after training. Report verified pass rate, mean reward, tool-call count, parser failures, termination reasons, episode time, infrastructure failures and per-task results. Training logs additionally need reward variance, zero-variance groups, masked trajectory/token counts, gradient norm, loss and optimizer steps. Several tiny fixtures validate mechanics; improvement on them alone is not evidence of general coding ability.

The first trainable release is ready only when these gates pass for one pinned native path. Optional backends can remain explicitly unverified without blocking a working CPU rollout tutorial and an honest supported-device GRPO demonstration.
