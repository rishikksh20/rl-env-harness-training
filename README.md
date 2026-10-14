# Minimal coding-agent RL lab

Stage A implements a model-free, stateful coding environment. Read a file, apply a patch, run visible tests, freeze the changes, and obtain an independent binary correctness reward. Each attempt starts from an immutable tiny repository. No model weights, PyTorch, TRL, GPU, inference server, or optimizer are required yet.

## Setup with uv

Python 3.11+ and [uv](https://docs.astral.sh/uv/guides/projects/) are required. All Python dependencies live in `pyproject.toml`; `uv.lock` locks the resolved versions. Pytest is a runtime dependency because the environment executes authored visible suites; Ruff is in the development group. There are no separate requirements files.

```bash
uv sync --locked
uv run coding-rl --help
```

The default sandbox requires Docker with cgroup v2 and working memory, CPU, and PID limits. Build the offline worker image once. It uses the same Python dependency lock and contains neither task solutions nor private grading cases.

```bash
docker build -f docker/Dockerfile -t coding-rl-sandbox:stage-a .
uv run coding-rl env-demo --config configs/default.yaml
uv run coding-rl smoke --config configs/default.yaml
uv run coding-rl env-inspect --config configs/default.yaml
```

Docker runs generated code as a non-root user with no network, a read-only filesystem/source snapshot, dropped capabilities, bounded memory/CPU/processes, bounded output, and a host-supervised deadline. The resolved worker image ID is recorded in verification events. No writable host directory or Docker socket is mounted into workers. Stage A deliberately requires cgroup v2 rather than claiming untested resource enforcement on every platform.

## Model-free demo without Docker

The explicit local backend executes only complete snapshots matching the bundled authored starter or reference fixtures. It refuses arbitrary edits, and provides no untrusted-code isolation. It never becomes an automatic fallback for Docker.

```bash
uv run coding-rl env-demo --sandbox trusted_local
uv run coding-rl smoke --sandbox trusted_local
```

The demo repairs the same task twice in different episode IDs, asserts matching pristine starter hashes and rewards, and then grades an unchanged failing attempt. Expect three visible tests per run, five protected assertion cases for a correct zero-timeout repair, rewards 1.0, 1.0, and 0.0, followed by cleanup and isolation confirmation. JSONL traces and frozen-source artifacts are saved under `outputs/trajectories/`. Private expected values are never copied into these traces or agent observations.

`env-inspect` displays current source identity, limits, allowed editable paths, tool descriptions, and the diff. It does not run candidate code and needs no model or Docker daemon. Its debugging interface is not a model tool.

## Tests

```bash
uv run pytest
uv run ruff check .
```

Real Docker integration tests are opt-in and require the prepared image:

```bash
CODING_RL_DOCKER_TESTS=1 uv run pytest -m docker
```

Tests distinguish executable authored local cases, simulated infrastructure failures, and real Docker tests. Skipped Docker checks are reported explicitly.

## Follow the implementation

| File | Responsibility |
|---|---|
| `config.py` | Strict YAML records and validation |
| `tasks.py`, `fixtures/` | Immutable starters, public dataset export, authored private cases |
| `environment.py` | Reset, four tools, budgets, snapshot, cached reward, cleanup |
| `patches.py` | Restricted exact-context unified diff parser |
| `sandbox.py` | Disposable snapshots, bounded processes, Docker/local policies |
| `worker.py` | Candidate callable execution; no expected values or assertions |
| `verifier.py` | Trusted assertions against values returned by independent workers |
| `trace.py` | Ordered JSONL events and final artifact recording |
| `harness.py`, `cli.py` | Scripted educational attempts and commands |

The public tools are `list_files`, `read_file`, `apply_patch`, and `run_tests`. Only existing allowed source files can be edited. Patches cannot add/delete/rename files or change protected tests. The final submission is reconstructed from actual source bytes, not from the agent's claim or a supplied Git diff.

Private grading uses a small authored JSON-case assertion controller, rather than embedding pytest in the controller. It restores protected visible tests and runs those through pytest, then compares real function outputs to private expected values. These protected assertions also cover every visible regression case, so candidate-controlled pytest behavior cannot replace correctness checks. The controller never imports candidate modules; each case uses a fresh worker. Same-process Python introspection therefore cannot reach controller assertions, although submitted code can inspect its own worker and current inputs or hardcode answers. This functional grading profile is not an arbitrary-repository hidden-pytest framework.

Infrastructure failures raise a distinct exception and produce no reward. Wrong results, submitted exceptions, resource violations, and malformed worker values produce 0.0 under healthy infrastructure. A frozen correct artifact earns 1.0 even if a limit ended the episode; termination and correctness remain separate fields.

## Next phases

Stage B adds real Transformers CPU rollouts and held-out evaluation. Stage C adds a verified native TRL `environment_factory` adapter and genuine GRPO updates. Stage D optionally adds custom harnesses and CPU vLLM evaluation. No training commands are advertised in this release.

See [the design index](docs/README.md), [mental model](docs/ARCHITECTURE.md), and [implementation plan](docs/06_STEP_BY_STEP_IMPLEMENTATION.md). Docker unavailable, missing worker image, or unsupported hard limits fail clearly; fix that infrastructure or explicitly use the authored-fixture local lesson.
