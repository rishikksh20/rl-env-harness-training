# Source layout and entry points

`src` is the Python source root. Application entry/configuration live directly
under it; RL, environment, model, harness, task, and utility code have separate
packages. Imports use package names such as `models.backend`, without a `src.`
prefix.

```text
src/
├── main.py                         # All CLI commands
├── config.py                       # Validated YAML/dataclass configuration
├── coding_rl/
│   ├── __init__.py                  # Lazy public environment API
│   ├── rollout/
│   │   ├── episode.py              # Public conversations, budgets and outcomes
│   │   ├── evaluate.py             # Scheduling, metrics and comparisons
│   │   └── probe.py                # Real-model read diagnostic
│   └── training/
│       ├── adapter.py              # Native four-tool environment adapter
│       ├── trainer.py              # TRL construction, admission and updates
│       ├── worker.py               # Parent/child training supervision
│       ├── feasibility.py          # Disposable model capacity measurements
│       ├── checkpoints.py          # Atomic full/LoRA artifact publication
│       └── evidence.py             # Development/baseline report validation
├── environments/
│   ├── coding.py                   # Reset, tools, freeze, grading and cleanup
│   ├── patches.py                  # Restricted exact unified diffs
│   ├── verifier.py                 # Protected assertions on worker results
│   └── execution/
│       ├── sandbox.py              # Immutable snapshots and Docker execution
│       └── worker.py               # Standalone isolated candidate executor
├── models/
│   ├── backend.py                  # Tokenizer and inference-process ownership
│   ├── profiles.py                 # Explicit Qwen/LFM model contracts
│   └── worker.py                   # Spawned model loading and generation
├── harnesses/
│   ├── scripted.py                 # Authored model-free lessons
│   ├── protocol.py                 # Tool schemas, parsing and dispatch
│   ├── bridge.py                   # Local chat/MCP protocol for external agents
│   └── external.py                 # OpenCode/custom harness process lifecycle
├── tasks/
│   ├── catalog.py                  # Immutable task metadata and public rows
│   ├── dataset.py                  # Atomic local repair-task importer
│   └── fixtures/                   # Public starters and protected grading data
└── utils/
    ├── errors.py                   # Shared action/infrastructure exceptions
    ├── ipc.py                      # Bounded JSON multiprocessing records
    ├── paths.py                    # Canonical relative path validation
    ├── process.py                  # Subprocess limits and RSS measurement
    ├── provenance.py               # Application-only nested source fingerprints
    ├── records.py                  # Atomic JSON writes and record digests
    ├── seeding.py                  # Stable sampling seeds
    └── trace.py                    # Ordered episode events and artifacts
```

Package initializers stay lightweight. The shared utilities do not import model,
environment, or training modules. Model weights and optional ML dependencies
continue to load only at runtime. Training and inference use the same IPC and
process helpers. Source provenance covers all application packages with relative
paths, including distinct workers with the same filename; it excludes candidate
fixture code and unrelated packages in `site-packages`.

## Install and run

From the repository root, refresh the editable installation after this move:

```bash
# Base environment and model-free lessons:
uv sync --locked
uv run coding-rl --help
uv run coding-rl smoke --sandbox trusted_local

# Model rollout and native training, including LoRA:
uv sync --locked --extra train-lora
source .venv/bin/activate
coding-rl --help
```

The console entry point is `main:main`. All existing subcommands/config paths
remain valid. With the virtual environment activated, the following are equivalent:

```bash
coding-rl tasks --split train
python -m main tasks --split train
python src/main.py tasks --split train
```

Examples for the reorganized application:

```bash
python src/main.py model-fetch --config configs/train-lfm2.5-350m.yaml
python src/main.py train --config configs/train-lfm2.5-350m.yaml --probe
python src/main.py train --config configs/train-lfm2.5-350m.yaml --feasibility
python src/main.py rollout --config configs/train-lfm2.5-350m.yaml --task zero-timeout
python src/main.py harness-rollout --config configs/opencode-lfm2.5-350m.yaml \
  --driver opencode --task zero-timeout
```

Use [the training guide](TRAINING_GUIDE.md) for the complete development/baseline,
training, checkpoint comparison, and custom-dataset sequence. Reorganization does
not change those evidence gates or enable external-harness gradient training.

Docker copies only `src/environments/execution/worker.py` to `/opt/worker.py`.
It does not copy controller packages, private cases, or model code:

```bash
docker build -f docker/Dockerfile -t coding-rl-sandbox:stage-a .
coding-rl smoke --config configs/default.yaml
```

The wheel includes every application package, `main.py`, `config.py`, fixtures,
and the standalone worker. Dependency versions and `uv.lock` stay unchanged.

## Import migration

Update custom Python integrations to these module paths. The old flat files are
removed rather than retained as forwarding modules.

| Previous module | Current module |
|---|---|
| `coding_rl.cli` | `main` |
| `coding_rl.config` | `config` |
| `coding_rl.environment` | `environments.coding` |
| `coding_rl.patches` | `environments.patches` |
| `coding_rl.verifier` | `environments.verifier` |
| `coding_rl.sandbox` | `environments.execution.sandbox` |
| `coding_rl.worker` | `environments.execution.worker` |
| `coding_rl.tasks` | `tasks.catalog` |
| `coding_rl.dataset` | `tasks.dataset` |
| `coding_rl.models` | `models.backend` |
| `coding_rl.profiles` | `models.profiles` |
| `coding_rl.inference_worker` | `models.worker` |
| `coding_rl.harness` | `harnesses.scripted` |
| `coding_rl.protocol` | `harnesses.protocol` |
| `coding_rl.bridge` | `harnesses.bridge` |
| `coding_rl.external_harness` | `harnesses.external` |
| `coding_rl.rollout` | `coding_rl.rollout.episode` |
| `coding_rl.evaluate` | `coding_rl.rollout.evaluate` |
| `coding_rl.probe` | `coding_rl.rollout.probe` |
| `coding_rl.trainer` | `coding_rl.training.trainer` |
| `coding_rl.trl_env` | `coding_rl.training.adapter` |
| `coding_rl.training_worker` | `coding_rl.training.worker` |
| `coding_rl.feasibility` | `coding_rl.training.feasibility` |
| `coding_rl.checkpoints` | `coding_rl.training.checkpoints` |
| `coding_rl.evidence` | `coding_rl.training.evidence` |
| `coding_rl.errors` | `utils.errors` |
| `coding_rl.trace` | `utils.trace` |

Shared helpers also have direct owners:

```python
from config import Config, load_config
from environments.coding import make_environment
from models.backend import TransformersBackend
from tasks.catalog import TaskCatalog
from utils.ipc import receive_record, send_record
from utils.paths import relative_path
from utils.process import ProcessResult, peak_rss_bytes, supervise
from utils.records import digest_record, write_record
from utils.seeding import derive_seed
```

The existing convenience API remains available without eager runtime imports:

```python
from coding_rl import Config, TaskCatalog, make_environment, run_model_episode
```

Checkpoint schemas and saved JSON reports are unchanged. Historical evidence
records retain the original source hashes/paths that they measured; regenerate a
baseline when changing the resolved Docker image or evaluation settings.

## Validation

The reorganization passed **187 tests** with native and Docker checks enabled,
Ruff, wheel/source builds, and a clean base-wheel install outside the checkout.
All three entry points found packaged tasks; the installed wheel ran nine
authored smoke cases without optional ML packages. The rebuilt Docker image also
passed nine smoke cases, and the relocated inference worker passed a genuine LFM
read probe. See [the validation record](evidence/module-layout-validation.json).
