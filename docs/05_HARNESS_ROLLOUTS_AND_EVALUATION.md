# Harnesses, rollouts, traces, and evaluation

Status: original design for current and future harnesses. Stage A implements the scripted driver. Phase 2 adds the CPU model harness and evaluator; see [Phase 2 status](PHASE_2_STATUS.md) for actual validation and the single-call-per-turn protocol that supersedes this draft's multi-call proposal. Custom harnesses and native training remain future work.

## 1. Harness responsibility

A harness decides how to present a task and observations to a model, how to parse its response, which allowed tool to invoke, and when the attempt ends. It does not own hidden correctness or the repository's authoritative state.

For example, after reading a file, the harness appends the returned text to the conversation. The model generates a patch call. The harness validates the call envelope and dispatches it. The environment validates the patch itself and applies it. The sandbox later executes visible tests. The verifier ultimately determines correctness.

The smallest design has three ways of driving the same core:

| Driver | Decision source | Loop owner | Use |
|---|---|---|---|
| Scripted lesson | Authored deterministic actions | Small scripted runner | Environment demo, smoke checks, negative fixtures |
| Minimal manual harness | Local Transformers model | Project's compact inference loop | CPU/CUDA rollout and held-out evaluation |
| Native training | Current trainable policy | TRL's supported environment loop | Genuine GRPO |

`harness.kind=minimal` therefore means manual generation in standalone rollout/evaluation mode and native TRL generation in training mode. These use the same tool semantics but different loop owners. Do not instantiate both loops for one training attempt.

## 2. Manual Transformers episode, step by step

1. Resolve a task from the requested split. Allocate an episode ID and record the exact model/checkpoint, model revision, tokenizer/template, seed, tools, and limits.
2. Reset the environment to a clean starter. Construct a system message explaining the allowed tools and a single user message containing the issue. Avoid oracle hints or scripted solution actions.
3. Render using the selected model's verified tool-aware chat template. Count complete context tokens before generation, reserving space for the next response and bounded observation.
4. Generate within the remaining assistant-token budget. Preserve the raw assistant output for the developer trace. Parse using the supported model/tool protocol, never by evaluating generated text as Python.
5. If a well-formed final response contains no tool call, record the final response and submit the current working copy. If a valid tool call exists, validate its name and arguments and dispatch it to the core.
6. Append each bounded tool observation using the model's required tool role and call ID. Multiple calls from one assistant turn are executed serially against the same episode state. Stop dispatch when the individual-call budget is exhausted.
7. Continue until a final response or an explicit tool, turn, token, context, or wall-time limit. Malformed actions consume budget; repeated parser failures end with a recorded reason. Never guess a patch from free-form assistant prose.
8. Freeze accepted source changes and invoke independent verification. An incomplete generated patch is not applied. A limited episode can have a valid correct artifact, but remains marked with its actual stop reason.
9. Record verdict, reward, timings, final artifact, and cleanup. Ensure resource release even if model generation, parsing, or grading fails.

Start sequentially: one model process, one episode, one test worker at a time. GRPO's conceptual group size does not require parallel CPU generation. Concurrency can be added after independent state and grouping are proved.

## 3. Custom harness boundary

A custom harness is an optional Python adapter chosen by a developer-controlled import path. It receives public task data, allowed tool descriptions, a narrow tool-dispatch interface, limits, and an event sink. It returns its finish reason and optional final text; the environment owner freezes and verifies the actual files.

| Input or capability | Why the adapter needs it |
|---|---|
| Public task ID and issue | Establish the problem |
| Exact tool schemas | Select valid operations |
| Narrow tool dispatcher | Perform bounded environment actions |
| Read-only budget snapshot | Decide whether another action is possible |
| Generation/event reporting interface | Explain decisions and timing |
| Cancellation deadline | Stop and release resources |

Its supported interface exposes neither the full core object nor its filesystem path, grading controller, hidden test plan, unrestricted shell, or Docker socket. By contract, it dispatches all actions through validated tools and cannot supply its own reward or “passed” verdict.

An installed in-process Python adapter is trusted controller code: a narrow interface is not process isolation, and arbitrary Python could inspect host files or objects. Only developer-selected trusted implementations use this import-path extension. Supporting an untrusted harness implementation requires its own process/container boundary later; model-generated decisions remain untrusted and tool-validated in every mode.

An existing coding harness that assumes arbitrary shell access may need a translation layer or a reduced tool configuration. The first contract supports the four tools. State this limitation clearly rather than granting broad access to make an adapter appear compatible.

The custom harness initially supports rollout and evaluation only. Its traces can be useful research data, but correct native GRPO integration additionally requires sampled token IDs, exact observation conditioning, loss masks, appropriate behavior probabilities, group ordering, compatible generation semantics, and synchronized policy weights. A remote closed model or stale inference server does not meet this merely by returning logs.

If a later supported `rollout_func` path is implemented, treat it as its own compatibility milestone. Keep it outside the first native training adapter until it passes the token and policy checks in [the training plan](04_TRAINING_AND_CPU_BACKENDS.md).

## 4. Trace records and artifacts

Use standard Python logging for human progress and a versioned JSONL event schema for durable episode records. One writer serializes each episode's events in sequence. No database is needed.

| Common field | Meaning |
|---|---|
| Schema and event type | Versioned interpretation and event category |
| Run / group / task / episode IDs | Correlation without sharing episode state |
| Sequence number | Stable event ordering independent of timestamps |
| UTC timestamp and monotonic elapsed time | Human date plus reliable durations |
| Policy/model/tokenizer identity | Which weights and rendering produced the attempt |
| Starter and current/frozen hashes | Reproducible state identity |
| Limits and consumption | Tool attempts, turns, tokens, context, and elapsed time |
| Status and reason | Lifecycle, generation termination, tool result, or verifier outcome |
| Artifact reference | Large bounded outputs, final diff, and frozen file manifest |

| Event | Useful payload |
|---|---|
| Reset | Public task metadata, starter hash, backend, fresh workspace identity |
| Observation | Public prompt or bounded tool content and truncation metadata |
| Generation | Raw model-generated text or exported native completion artifact; rendering identity |
| Tool call | Exact name, call ID, validated arguments, whether accepted |
| Tool result | Bounded result/error, duration, public test summary |
| Termination | Actual final response/limit/cancellation reason and counters |
| Submission | Frozen allowed-file manifest, diff reference, and hash |
| Verification | Verifier version, binary correctness or infrastructure category, public-safe summary |
| Reward | Exactly one scalar for a completed valid verdict |
| Cleanup | Resources released or explicitly recorded cleanup error |

Large source/test outputs are capped before they reach both the model and ordinary trace. Save developer artifact references when useful without silently substituting full output into context. Never write hidden assertion text or expected outputs into the normal trace. Restricted verifier diagnostics may live in a separate developer-only artifact outside model-accessible paths.

For manual inference, the harness can record each raw generation directly. For native training, inspect supported logging/callback/export points in the pinned trainer. Establish how native generations, task ordering, and adapter episodes are correlated before claiming a complete native trajectory. Environment call events alone do not recover omitted assistant prose. If the trainer exports only an aggregate completion, label it accurately and retain the ordered tool events; do not invent per-turn generations.

The trainer retains learning tensors independently. Optional tensor diagnostics must preserve masks, sampled IDs, and policy version; a human JSONL trace is never treated as a complete GRPO replay buffer.

## 5. One configuration with mode-aware validation

Configuration is an operational description translated into installed-library arguments. It is not a direct copy of GRPO constructor keywords. Store the resolved configuration with every run.

| Section | Required decisions |
|---|---|
| Model | Name/revision, tokenizer, dtype, training device, full or LoRA mode |
| Rollout | Backend/device, generations per task, sampling, assistant/context budgets |
| Environment | Dataset task selection, debug visibility, call/turn budgets, allowed task set |
| Harness | Minimal or custom; optional developer-supplied adapter path |
| Sandbox | Docker or explicit trusted-local; command/episode/grading deadlines and resource caps |
| Reward | Binary private-test rule, verifier version, infrastructure retry policy |
| Training | Enabled flag, GRPO settings, learning rate, steps, KL coefficient, checkpoint path |
| Logging | Level, trace path, standard-library reporting, artifact caps |
| Reproducibility | Seeds, task splits, dependency/model revisions, worker image |

Starting proposals: training disabled, CPU Transformers rollout, four generations per task for training experiments, six attempted calls, six tool turns, 256 generated assistant tokens per manual turn, 768 per manual episode, 2,048 total context tokens, and a 20-second test-command timeout. Use a separate episode deadline and a grading deadline so retries cannot run indefinitely. Benchmark values before promoting them to working defaults.

Native training budgets follow the actual installed trainer semantics, which can differ from manual inference. Record the exact mapping and reject unmapped hard-limit requirements. Do not pass fictional per-turn settings to TRL.

Validation rejects unknown task IDs, duplicate split membership, writable protected tests, nonpositive limits, unsupported device/backend combinations, an external harness in unsupported training mode, and inconsistent group/batch settings. Select device separately for each mode. A CPU evaluation profile does not automatically alter training generation or implement CPU/GPU weight transfer.

Keep heavier libraries behind mode-specific imports/extras. Model-free demo and smoke need no PyTorch, Transformers, or downloaded weights. LoRA adds PEFT to a working training stack. Optional CPU vLLM should use a separate inference environment when its dependencies conflict with training. Pin each supported profile after tests rather than installing all optional stacks together.

## 6. Held-out evaluation protocol

Use four tiny tasks initially: three for demonstrations/training and one held-out behavior family. Keep fixtures offline and immutable. A single held-out task is enough to test the evaluation plumbing, but far too small to support a broad performance claim.

Partition by task family or generation recipe, not by arbitrary row alone. Closely related buggy starters, oracle solutions, and duplicate issues must remain in the same split. Human-authored scripted repairs are acceptance fixtures; they are never counted as model evaluation results.

1. Evaluate the initial checkpoint on the held-out partition with fixed tools, limits, model/template revisions, sandbox image, and sampling settings.
2. Train using only the training partition. Reward design and grader versions remain fixed during a comparison.
3. Reload the saved checkpoint and repeat held-out evaluation under the same protocol, using a documented seed set.
4. Record per-task attempt results and aggregate summaries. Do not train on evaluated traces or repeatedly tune against the same held-out cases without relabeling them as development data.
5. Report small-sample limits plainly. A genuine changed checkpoint can still show no improvement; an optimizer update is a correctness milestone, not proof of useful learning.

| Metric | Definition |
|---|---|
| Verified pass rate | Independently verified successes divided by completed valid evaluations |
| Mean reward | Mean binary reward across those same valid evaluations |
| Coverage | Valid evaluations, planned attempts, infrastructure exclusions, and retries |
| Average tool calls | Mean attempted calls; accepted calls reported separately if useful |
| Time per episode | End-to-end duration with inference, execution, and grading breakdown |
| Stop reasons | Final response, call/turn/token/context/time limit, parser failure, cancellation |
| Verdict categories | Correct, incorrect, invalid submission, candidate runtime failure, grader infrastructure error |
| Per-task results | Counts, pass rate, mean reward, and resource usage |
| Training diagnostics | Reward dispersion, equal-reward groups, gradient/weight-change evidence, policy version, selected loss settings |

Timeout categories distinguish submitted code exceeding a declared limit from a broken Docker engine or missing fixture. The former is a valid failing evaluation; the latter has no model reward and is excluded with an explicit count. In training, retry/abort preserves complete collection units and group identity.

Record limit-caused verified successes separately. Passing visible tests or writing persuasive final text never increments the verified success count. Keep evaluation attempts independent rather than selecting the best trace invisibly.

For the initial lab, report independent attempt pass rate. If later reporting pass-at-k, document the number of sampled attempts, aggregation rule, independence assumptions, and estimator. Do not label four GRPO training samples as a held-out benchmark score.

## 7. Developer API and CLI contract

The proposed importable API has operations for loading configuration, constructing an environment, running an episode, evaluating a checkpoint, and invoking training. They return structured outcomes/artifact paths rather than printing as their only interface. The CLI formats those outcomes for a learner.

| Planned interface | Mode and ownership |
|---|---|
| `env-demo` | Scripted owner; traces readable state transitions and repeats a clean reset |
| `env-inspect` | Developer owner; tool schemas and public runtime state only |
| `smoke` | Known-positive/negative environment acceptance, no weights |
| `rollout` | Manual minimal/custom driver, no optimizer |
| `train` | Native TRL owner; compatible training profile required |
| `eval` | Held-out manual driver using selected checkpoint |

All commands use the same configuration file path and explicit overrides. CPU smoke, real CPU rollout, GPU/native training, and optional CPU vLLM evaluation get separately validated profiles. A REST service, Gymnasium wrapper, OpenEnv server, distributed workers, or web dashboard is unnecessary for this API.
