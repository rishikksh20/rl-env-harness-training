# Stage A: implemented model-free environment

Stage A delivers a working Python package managed by `uv`, with no model download or training stack. The next milestone is Stage B real Transformers CPU rollout. This status record distinguishes executed checks from future design gates.

This is the original implementation acceptance record. The [follow-up Phase 1 evaluation](PHASE_1_EVALUATION.md) reran the suite and identified a cleanup/cache failure path not covered by those tests. Its fix is planned as the first Phase 2 prerequisite.

## What exists

The core keeps one private repository per episode, exposes four bounded tools, freezes actual source bytes, reconstructs an independent candidate, verifies real executable behavior, records one completed binary reward, and removes its resources. Resetting creates a fresh episode ID and reproduces the immutable starter hash.

Four bundled offline fixtures cover zero timeout, interval boundaries, empty defaults, and chunk tails. Three belong to the training/demo partition; chunk tails belong to the held-out partition. Scripted reference repairs are teaching/acceptance assets, not model evaluation results. Protected grading cases contain all visible regression cases plus additional behavior checks.

The CLI implements `env-demo`, `env-inspect`, and `smoke`. Python callers can load configuration, construct the environment, operate its tools directly, or run an authored episode. `env-inspect` produces JSON Schema descriptions for exactly the four tools; public core inspection/cleanup methods are not included.

Runtime dependencies are PyYAML and pytest. Ruff belongs to the development group. `pyproject.toml` defines the package, Python floor, entry point, dependency group, build backend, pytest configuration, and linting. `uv.lock` locks resolved packages; Docker installs from that same lock without installing the project or development tools. Both Docker base images are pinned by digest.

## Commands

```bash
uv sync --locked
docker build -f docker/Dockerfile -t coding-rl-sandbox:stage-a .
uv run coding-rl env-demo --config configs/default.yaml
uv run coding-rl smoke --config configs/default.yaml
uv run coding-rl env-inspect --config configs/default.yaml
CODING_RL_DOCKER_TESTS=1 uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

For the authored-fixture lesson without Docker:

```bash
uv run coding-rl env-demo --sandbox trusted_local
uv run coding-rl smoke --sandbox trusted_local
```

The local backend rejects snapshots that differ from bundled authored starters/reference variants. It has no untrusted-code isolation and is never selected automatically.

## Verification record

Validation date: **2026-10-14**.

| Check | Executed result |
|---|---|
| Complete suite with real Docker enabled | 68 tests passed; eight exercise real Docker |
| Lint and formatting | Ruff check and formatting check passed |
| Dependency lock | uv lock check and locked offline sync passed |
| Distribution build | Source distribution and wheel built successfully |
| Installed wheel outside checkout | Nine-case authored local smoke passed in an independent temporary virtual environment |
| Default Docker image | Built successfully; preflight verifies effective non-root identity and cgroup v2 CPU/memory/PID caps |
| Local demo | Two independent repairs earned 1.0; unchanged attempt earned 0.0 despite visible tests passing |
| Default Docker demo and smoke | Demo earned 1.0, 1.0, 0.0; all nine authored smoke cases passed |

The real Docker tests cover final correctness and cleanup; read-only source snapshots; no external network; absence of grader paths; private-loop timeout; output flooding; memory exhaustion; forged success output; candidate-chosen exit code 125; and a submitted pytest skip. Tests also cover reset and group isolation, traversal/symlink/hardlink/mode attacks, atomic patch rejection, protected tests, bounds, configuration, cached rewards, same-snapshot infrastructure retry, private-controller JSON validation, and subprocess descendants/closed output streams.

Host execution used Python 3.12.3, uv 0.9.1, pytest 8.4.2, PyYAML 6.0.3, and Ruff 0.17.0. The Docker worker uses Python 3.12.15 and the same locked runtime dependency versions. Python 3.11 is the declared minimum; this record does not claim a separate successful Python 3.11 run.

## Actual demo excerpt

The Docker demo produced this sequence; episode IDs and elapsed timings vary. The local demo uses the same operations and verdicts.

```text
[reset] task=zero-timeout ... starter=b5c6379bbb37
[action] list_files
[action] read_file
[action] apply_patch
[action] run_tests
[observation] ... 3 passed ...
[finish] ... scripted_finish ...
[verify] private checks: 5/5 status=correct
[reward] 1.0
[cleanup] ... released ...
[reset] task=zero-timeout ... starter=b5c6379bbb37
... independent repair and verification ...
[reward] 1.0
[reset] task=zero-timeout ... starter=b5c6379bbb37
... no repair, visible tests pass ...
[verify] private checks: 0/5 status=incorrect
[reward] 0.0
[cleanup] ... released ...
[isolation] two clean independent repairs verified; negative reward=0.0
```

The full run saves ordered JSONL, resolved configuration, source/diff artifacts, submission hashes, private-safe verdicts, and worker image identity under `outputs/trajectories/`. The complete Docker demo transcript is saved locally as `outputs/phase-a-docker-demo.txt`; generated outputs are ignored by version control.

## Choices that refine the original draft

- The private controller performs ordinary trusted assertions against authored JSON cases, rather than embedding a second pytest orchestration layer. Candidate code remains outside the controller process. Visible tests still execute through pytest.
- Protected assertion cases repeat visible regressions as well as hidden cases. A candidate's ability to manipulate its own pytest process or print a pass message cannot replace these checks.
- The grading profile supports fixed JSON-callable function tasks. It is not a general hidden-test harness for arbitrary repositories.
- Docker preflight checks effective resource settings in a prepared worker. It does not rely on a CPU capability field removed from newer daemon versions.
- Pytest exit numbers and Docker-reserved exit numbers can be chosen by submitted code. Healthy worker failures stay candidate failures; trusted container startup/control failures remain infrastructure exceptions.
- File revisions commit through a workspace pointer swap after complete validation. Test workers receive independent read-only snapshots, not the writable authoritative directory.
- Patches modify only existing allowed newline-terminated source files, using restricted unified diff headers and exact contexts. Full Git patch directives, additions, deletions, and renames are rejected.
- Episode/core call limits are implemented. Model token budgets, native TRL dispatch accounting, and trainer masks are still later-stage gates.

## Remaining stages

Stage B must load the target checkpoint, parse real tool calls, record generations, and evaluate held-out tasks. Stage C must prove native same-task grouping, action/history consistency, token masking, policy probabilities, and genuine changed weights. Stage D's custom harness and CPU vLLM remain optional.

This release implements the environment and verifiable reward lesson. It does not perform inference or optimize model weights.
