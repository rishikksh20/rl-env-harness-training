# Coding environment, sandbox, and binary RLVR

Status: original environment design, now implemented for Stage A. Read this after [the architecture draft](ARCHITECTURE.md); see [Stage A status](PHASE_A_STATUS.md) for actual validation. The source requirements are in [CODEX_AGENTIC_RL_ENV_SPEC.md](../CODEX_AGENTIC_RL_ENV_SPEC.md).

## 1. Understand the world before adding the model

The environment is one editable copy of a tiny Python repository. A model action can inspect that copy, change an allowed source file, or ask for visible tests. Reading changes the model's knowledge; patching changes the world. The harness chooses actions, while the environment enforces their meaning and limits.

For a zero-timeout task, the state is the actual timeout module and current edits. The observation is the source text returned by a file read. The action is a named patch call. One independent attempt is an episode; its ordered model messages, calls, results, final source snapshot, and reward form the trajectory. A new episode starts from the immutable starter, even when the same task is selected again.

Keep the first implementation in three cohesive modules: task manifests and fixture loading, the core environment with workspace operations, and sandbox execution plus verification. Split sandbox and verifier into separate files when that makes ownership clearer. A thin TRL adapter delegates to this core; it does not introduce another state machine.

## 2. Define the task before its runtime state

| Task field | Meaning and visibility |
| --- | --- |
| Task ID and issue text | Stable public identity and user-facing request. |
| Starter revision and file manifest | Immutable version plus hashes of authored files. A Git-style revision label is sufficient; runtime copies need no Git metadata. |
| Editable source paths | Exact allowlist of existing UTF-8 files the agent can change. |
| Visible tests | Readable files provided with the starter, protected from edits. |
| Private verifier reference | Trusted controller lookup; never serialized into prompts or tool observations. |
| Invocation contract | Authored module/function names and JSON-compatible inputs and results. |
| Budgets and split | Execution limits and train/evaluation assignment. |

Export dataset rows with the public prompt and opaque task ID. Keep private paths, expected answers, and reference fixes out of exported metadata. At load time validate fixture hashes, regular-file types, allowed paths, and the consistency of issue text with its visible API. Loading a fixture never imports its source module on the host.

Use four authored tasks, with no packages, downloads, external services, clocks, or randomness:

| Task | Public issue and visible coverage | Private coverage | Partition |
| --- | --- | --- | --- |
| `zero-timeout` | A timeout normalizer must distinguish zero from an omitted value; visible tests cover ordinary positive values and omission. | Zero, preserved positive values, and default behavior. | Train and scripted demo. |
| `range-boundary` | A membership helper must implement the documented inclusive lower and exclusive upper bounds; visible tests cover interior and exterior values. | Both endpoints, negative bounds, and an empty interval. | Train. |
| `empty-default` | An option chooser should use its default only when the option is absent; visible tests cover ordinary values and omission. | Empty string, false, zero, and omitted value. | Train. |
| `chunk-tail` | A chunking helper must preserve every input item in order, including a short final chunk; visible tests cover exact multiples. | Remainder, empty list, and chunk size larger than input. | Held-out evaluation. |

All four expose a fixed function callable through bounded JSON inputs and outputs. Specify valid input types, null/omitted semantics, and positive chunk-size requirements before writing tests. Fix the Python/pytest versions, locale, timezone, and case ordering; exclude wall-clock timestamps from correctness comparisons. The issue specifies semantics; it contains no reference patch. Private assertions may be readable by the human developer in this educational repository, but they never enter the agent workspace. Keep authored reference fixes under test assets outside exported task data. Four tasks demonstrate mechanics; they cannot establish broad coding ability or credible generalization.

## 3. Represent state explicitly

| Core state | Purpose |
| --- | --- |
| Episode ID, task ID, starter hash | Identify an attempt and prove a clean start. |
| Authoritative workspace and current file hashes | Source of truth for edits across tool calls. |
| Lifecycle phase | New, active, sealed, graded, or closed. |
| Core-visible invocation count and remaining budgets | Enforce limits, including actions rejected within the core. |
| Start time, deadline, and last tool result | Account for execution and explain outcomes. |
| Frozen submission hash and cached verification | Make final grading stable and idempotent. |
| Terminal reason and infrastructure status | Distinguish stopping from correctness. |

Observations are bounded public projections of this state: file contents, patch receipts, visible test output, and structured errors. They contain no host paths, private grader output, controller configuration, or credentials. A developer-only inspection view can show the public tool schemas, hashes, diff, counters, and limits. It must never become an automatically discovered model tool.

## 4. Implement reset and cleanup first

1. Resolve a task ID through the trusted manifest. Reject unknown IDs without allocating resources.
2. Clean up any previous episode owned by this environment instance. Refuse to continue if an old live worker cannot be stopped safely.
3. Create a fresh temporary workspace with a unique episode ID. Copy only manifest-listed regular files, without Git metadata or private assets.
4. Check the resulting starter hash, protected file hashes, and editable allowlist. Assign fresh counters and a monotonic deadline.
5. Enter the active phase. In dataset-driven TRL mode, return no additional prompt because the dataset already supplies it.
6. On any partial-reset failure, remove allocated resources and report an infrastructure error.

The authority for edits is a host-side plain-file workspace operated on by trusted file routines. Model text is treated as data. The host never imports, installs, runs, or discovers plugins from submitted Python. Disposable Docker workers receive read-only snapshots to execute tests; their scratch files cannot change the authoritative episode state.

Cleanup stops and removes only episode-owned workers and deletes only registered temporary directories. It is idempotent, preserves recorded traces, and never deletes task assets or unrelated containers. A manual harness invokes it in a guaranteed exit path. The TRL adapter owns cleanup inside final reward handling and on stale reset, rather than exposing a cleanup tool or assuming an undocumented trainer callback.

## 5. Expose exactly four tools

| Tool | Public arguments | Result and state effect |
| --- | --- | --- |
| `list_files` | Relative directory, default repository root. | Sorted public paths with an entry cap; no filesystem mutation. |
| `read_file` | Relative file, starting line, maximum lines. | Bounded UTF-8 text and truncation metadata; no mutation. |
| `apply_patch` | One restricted unified diff. | Applied/rejected receipt, changed paths, and source hash. Atomic source update on success. |
| `run_tests` | None. | Bounded visible test stdout/stderr and process status from a fresh snapshot worker. |

Give arguments types and clear Google-style docstrings. The adapter exposes these names plus reserved reset and reward methods only. Internal state, verifier, inspection, and cleanup stay private to tool discovery.

Every core invocation checks phase and budgets before execution. Validation failures within the core consume its call budget. The manual dispatcher also counts proposed calls rejected before method entry. Native TRL may reject unknown names or invalid arguments without invoking the core, so complete attempted-call accounting needs verified dispatch metadata; native iteration/token limits bound such failures separately when that metadata is unavailable. Return an error category, short explanation, and remaining public budget instead of crashing. Examples include invalid path, protected file, malformed patch, context exhaustion, and command timeout. Test failures are useful observations; they do not terminate the episode by themselves.

An initial manual-harness configuration proposes six attempted tool calls, six tool iterations, 256 generated tokens per turn, 768 generated tokens per episode, and a 2,048-token context limit. Native training requires a separately verified mapping to its supported accounting. Calls and iterations are separate counters because one generation may contain several calls. Handle multiple calls serially. Suggested file/patch limits are 64 KiB each, 200 listed paths, and 16 KiB of displayed command output. Token-aware model context limits may require much smaller observations than these outer byte caps. These are configurable starting values, not measured performance guarantees.

## 6. Make path and patch rules small enough to audit

Accept repository-relative POSIX paths only. Reject absolute paths, parent segments, NULs, backslashes, ambiguous encodings, and paths outside manifest-listed directories. Normalize once before allowlist checks. Reject symbolic links in any path component, hard-linked files, special files, and executable-mode changes. Check parent directories and regular-file type at the actual read/write operation; a string-prefix check is insufficient.

Support text edits to existing allowed source files only in the first release. Reject additions, deletions, renames, binary patches, symlink or mode directives, ambiguous headers, and changes to tests, runner files, dependency manifests, startup hooks, or Git metadata. Require exact matching context, without fuzzy application. Validate every hunk and resulting file in memory before replacing files; a rejected multi-file patch leaves all originals intact.

Do not pass model patch text to a shell. A bounded patch parser can use normal Python text operations for this narrow format. If implementing the parser grows difficult, reduce the supported format and document that choice; never widen privileges to compensate. Serialize file operations within an episode, and use atomic replacement plus rollback for multi-file writes.

## 7. Execute Python only behind the sandbox boundary

For each visible-test run, create a fresh worker from a pinned, prebuilt Python/pytest image and a read-only snapshot containing source and visible tests. Install dependencies during image preparation, not during episodes. Keep scratch/cache output in bounded temporary storage; disable bytecode writes and pytest caching where practical. Read-only snapshots mean an imported module cannot edit the next agent observation.

| Worker setting | Planned policy |
| --- | --- |
| Network | Docker's `none` network driver, with no published ports. |
| Identity and privilege | Fixed non-root user, all capabilities dropped, and no privilege escalation. |
| Root filesystem | Read-only; small temporary scratch mount only. |
| Resources | Start with one CPU, 512 MiB memory, bounded swap configuration, 64 processes, and bounded scratch size. |
| Time | Twenty seconds per test command, enforced by the host supervisor. |
| Mounts | Only the episode snapshot read-only; no writable host mounts, home directory, secret paths, or Docker socket. |
| Runtime confinement | Retain default seccomp and available AppArmor/SELinux confinement. |
| Invocation | Trusted fixed executable and argument vector; no model-controlled shell command. |

Docker documents the non-root identity, read-only filesystem, process limits, and privilege controls in its [container-run reference](https://docs.docker.com/reference/cli/docker/container/run/). Its [`none` driver](https://docs.docker.com/engine/network/drivers/none/) supplies a loopback-only network stack. Retain the supported [default seccomp profile](https://docs.docker.com/engine/security/seccomp/). Docker does not impose resource limits automatically; configure hard memory and CPU limits and check actual enforcement. [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)

Prefer [rootless Docker](https://docs.docker.com/engine/security/rootless/) when supported. Rootless resource enforcement depends on cgroup v2 and host configuration; preflight must verify limits rather than assume them, and refuse untrusted execution if hard caps cannot be enforced. Record which confinement features are available; rootless Docker has different limitations, including AppArmor support. [Rootless troubleshooting](https://docs.docker.com/engine/security/rootless/troubleshoot/)

A deadline kills the container and its child processes, not just the host Docker client. Collect container status before removal so a candidate memory violation is distinguishable from a launcher failure. Capture output incrementally and bound Docker logging as well as displayed text. A text-output cap alone does not prevent host disk exhaustion.

This is containment for a local educational lab, with host-kernel and runtime dependencies. It is not a promise that Docker prevents every escape. Only the trusted orchestrator controls Docker arguments and daemon access. [Docker Engine security](https://docs.docker.com/engine/security/)

## 8. Keep the local backend strictly for authored fixtures

An optional trusted-local backend is useful for the deterministic demo when Docker is unavailable. It may execute only complete file snapshots whose hashes match authored starter or reference-fixture manifests. It uses the same budgets and output contract but has no security isolation.

Real model patches and custom-harness submissions must use Docker. A configuration flag alone cannot make their Python trustworthy. If Docker is unavailable, those modes fail preflight; they never fall back to host execution. The local demo prominently records its trusted-fixture restriction.

## 9. Capture a trustworthy final submission

Stopping seals the episode against further writes. Read final source bytes from the authoritative allowlisted regular files, recheck types and caps, and compare them against the immutable starter. Construct a canonical patch and submission hash through trusted file comparison. Do not trust a model-provided diff, Git index, configuration, hooks, test output, or claim of success.

Rebuild an independent clean grading checkout from the starter and apply this canonical submission through the same restricted validator. Retain protected visible files from the starter. Verification operates on this checkout, never on the mutable rollout tree, stale interpreter state, or a worker that previously ran visible tests.

## 10. Protect hidden assertions by separating execution

For these four function-level tasks, the original draft proposed a private pytest assertion controller. Stage A uses a smaller authored JSON-case assertion controller plus a dedicated Docker grading worker. Protected assertions also repeat visible regression cases; pytest remains the agent-visible test runner. The controller sends fixed allowed function calls with JSON inputs and receives bounded JSON values or structured execution failures. The candidate module is loaded only inside the grading worker.

Expected answers and assertions remain in the controller. The worker has no private grader mount, expected-result data, host access, or shared controller interpreter. No candidate pickle, object, expression, traceback evaluator, plugin, or module enters the controller; validate JSON type, size, nesting, and numeric rules before trusted comparisons. A forged success message or worker exit code is never sufficient for reward: the controller checks actual returned values against its assertions.

Use a fresh grading process for every independent case where import/global-state isolation matters. Keep invocation names fixed by task metadata; the model cannot select a grader function. Disable plugin autoload and use explicit trusted pytest configuration in the controller. Pytest documents [plugin loading controls](https://docs.pytest.org/en/stable/how-to/plugins.html). Run candidate workers with Python's isolated startup and controlled import paths; Python's [`-I` mode](https://docs.python.org/3.11/using/cmdline.html#cmdoption-I) limits startup influence but is not a security sandbox.

This separation has a precise limit: submitted Python can inspect the dispatcher, its own process, and inputs delivered for the current case. It may monkeypatch its own interpreter or spoof its value channel. Hiddenness means assertions are withheld from rollouts and kept outside that interpreter; it does not make all aspects of Python execution unobservable or prevent hardcoded answers. Keeping private pytest tests in the same interpreter as submitted Python would allow stack/module inspection and assertion manipulation even with read-only files, so it is not the default.

A later arbitrary-repository grading profile needs a separate design. Running ordinary hidden pytest tests that directly import candidate code is simpler but weakens grader protection. Do not silently replace the function contract with that profile or describe it as equivalent.

## 11. Emit one binary reward with complete status

Reward is 1.0 only if the final submission passes every required trusted check, including protected visible regression cases and hidden cases. Otherwise a validly evaluated submission receives 0.0. There is no reward for test-running, confident prose, fewer calls, or an apparent visible pass.

| Situation | Episode/verifier meaning | Training treatment |
| --- | --- | --- |
| Final response or call/turn/token limit | A stopping reason; independent of correctness. | Verify the sealed final source once. |
| Wrong result, submitted syntax/import failure | Candidate failure. | Reward 0.0. |
| Candidate infinite loop, OOM, malformed value channel | Candidate resource/protocol failure under healthy limits. | Reward 0.0 with a specific status. |
| Submission violates source/path contract | Invalid submission. | Reward 0.0; retain rejection details. |
| Docker unavailable, missing trusted image, broken fixture, trusted grader crash | Infrastructure failure, not evaluated correctness. | No reward value; retry/exclude as described below. |

`get_reward` first seals an active episode, freezes the submission, and computes its cache key from task/starter, submitted contents, grader version, image digest, limits, and deterministic settings. It caches the complete successful grading outcome, including reward and status. Repeated calls return that outcome without rerunning tests. Infrastructure failures do not become cached reward-zero results. Cleanup runs in a guaranteed exit path after capture; the closed phase means resources are removed, while metadata and frozen submission remain available for inspection, cached returns, or infrastructure retries.

Retry a transient verifier failure against the same frozen submission. If recovery fails, exclude the entire same-task GRPO group, not only its failed member; otherwise group-relative advantages change their comparison basis. If the chosen trainer cannot exclude groups safely, fail the batch before an optimizer update and retry through a supported boundary. Never substitute zero, change group size, or replay old trajectories after a policy update without re-establishing on-policy validity.

## 12. Build and prove this in a model-free vertical slice

1. Author the four fixtures, protected tests, task manifests, and JSON callable contracts. Validate all starter hashes without importing task modules.
2. Implement reset, inspection, bounded listing/reading, and idempotent cleanup. Prove the same task starts identically in two episode IDs.
3. Implement restricted atomic patching. Prove an edit is visible in its episode and absent from another episode.
4. Add the Docker visible-test worker and timeout supervision. Prove it cannot mutate authoritative source or mount private grader files.
5. Add canonical capture, fresh grading checkout, private assertion controller, and binary reward cache.
6. Script a correct zero-timeout episode, a no-op incorrect episode, and a visible-passing but hidden-failing boundary patch. Show public observations followed by developer-only verification and reward.
7. Run the correct demo twice with different episode IDs; show matching starter hashes, independent edits, identical reward, and cleanup.
8. Add group tests with four independent copies of one task and reward outcomes such as 1, 0, 1, 0. Inject an infrastructure failure and prove no partial group reaches an optimizer.

Focused acceptance tests cover traversal and symlink attacks; protected-test edits; atomic rejection; patch/read/output caps; post-seal action rejection; timeout child cleanup; reset after partial failure; reward idempotence; and stdout containing a forged pass message. Include candidate probes for controller-file reads, stack inspection, pytest monkeypatching, import-path shadowing, process spawning, network access, and output flooding. Execute all probes in Docker. Assert observed containment and trusted assertion results without making broader security claims.

Only after this slice is understandable and passes should model inference and the native TRL adapter be introduced. They choose actions and consume observations; they do not replace environment state, alter private correctness checks, or weaken execution containment.
