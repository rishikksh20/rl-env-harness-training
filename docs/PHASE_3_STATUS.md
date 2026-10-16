# Phase 3 implementation and acceptance status

Updated **2026-10-16**. The native CPU training path is implemented. Installed TRL
tests exercise actual generation/tool-loop dispatch, rewards, tensors, backward,
optimizer callbacks, and full/LoRA save/reload using tiny fixture models and
scripted completions. These are integration tests, **not a Qwen3.5-0.8B learning
result**. No target-model training pilot or held-out before/after result is claimed.

The detailed implementation plan was written first in
[PHASE_3_IMPLEMENTATION_PLAN.md](PHASE_3_IMPLEMENTATION_PLAN.md). The acceptance
criteria remain [the original design](PHASE_3_NATIVE_TRAINING_DESIGN.md).

## Implementation

| Component | Responsibility |
|---|---|
| `config.py` | Separate strict native budgets and Hub/full/LoRA policy sources |
| `trl_env.py` | Exactly four typed tools, public train rows, fresh core episodes, latched failures |
| `trainer.py` | Installed compatibility probe, native trainer construction, complete-group reward admission, actual tensor/probability audits, optimizer evidence and checkpoint coordination |
| `training_worker.py` | One supervised model/optimizer process, RAM/deadline limits, exact owned-resource cleanup |
| `checkpoints.py` | Staged saves, verified reload, complete file digests and atomic completion manifest |
| `models.py`, `inference_worker.py` | Content-identified local full/LoRA evaluation using the pinned base tokenizer |
| `evaluate.py` | Policy-independent evaluation profile, coverage/cost metrics, matching before/after comparisons |

The trainer specializes public `compute_loss` and `TrainerCallback` extension
points and supplies a public reward function. It does not replace TRL generation,
its tool loop, advantage calculation, loss, or private collection methods. The
zero-valued admission reward contributes no learning signal; native environment
rewards supply the independently verified binary objective.

The [native environment contract](https://huggingface.co/docs/trl/en/grpo_trainer#environments)
is checked against the installed **TRL 1.16.0** implementation, rather than assumed
from moving documentation. The locked CPU stack is Torch **2.14.1+cpu** on Linux,
Transformers **5.19.0**, Hub **2.2.0**, Datasets **5.1.0**, Accelerate **1.15.0**,
and PEFT **0.21.2** for LoRA. Datasets 4.x conflicted with the retained Hub pin;
Datasets 5.1 resolves it. The three direct inference pins are unchanged; the shared
lock moves FSSpec from 2026.9.0 to 2026.7.0 to satisfy Datasets' upper bound. CUDA has no profile or
extra; changing `model.device` cannot enable it.

## Explicit native profile choices

The first profile uses one device, FP32, four attempts, **batch four and accumulation
one**, and one update per collection. Seeing the complete group in one public
loss call allows all members to be checked before backward. This changes the
design's proposed microbatch-one pilot and can require more memory. A full model
update is experimental; LoRA is the default. The target model's actual feasibility
measurement is required at runtime before collection.

The native profile permits **sequential calls within a turn**. It retains the core
call cap, including invalid calls reaching the core. Atomic single-call dispatch
and a proposed/unknown/malformed-call counter have no supported native interception
point in this release and are not advertised. The public native system prompt
declares sequential execution. The manual evaluator keeps its existing single-call
protocol for both baseline and trained policy.

The upstream Qwen response parser strips patch whitespace, including a required
final newline. A configured public `response_template` preserves payload bytes and
permits paired framing newlines. Reports fingerprint this parser configuration,
the original template, and the selected prefix-preserving native template. Genuine
recorded Phase 2 read IDs are also parsed by the installed native parser; multiline
patch probes are authored protocol fixtures, not new model generations.

| Configuration | Native owner / behavior |
|---|---|
| `training.completion_tokens` | TRL `max_completion_length`, including tool suffixes |
| `training.context_tokens` | Every initial prompt plus the complete region must fit |
| `training.observation_tokens` | Valid JSON bounded before insertion |
| `training.tool_iterations` | TRL `max_tool_calling_iterations` |
| `environment.max_tool_calls` | Existing entered-core call counter |
| `training.wall_seconds` | Parent deadline for the whole training run |
| `training.memory_limit_mb` | Parent process-tree RSS cap and pre-collection memory probe |
| `rollout.*` | Manual evaluation only; not translated to native budgets |

TRL can execute a tool and subsequently remove its result at a token/context
cutoff. The reward hook compares all executed observations with the retained
native conversation before grading/advantages. The loss hook checks the actual
trainer IDs and masks against admitted IDs and verified external suffix spans.
It also verifies advantage association after native shuffling and captures the
actual model forward's conditioning and token probabilities. Independent unpadded
fused scores must agree within FP32 tolerances. A disagreement rejects the entire
collection. Re-tokenized conversations are never substituted for learning tensors.

Infrastructure failures are latched before TRL can catch tool exceptions. Cleanup
failure leaves an attempt incomplete even if its private verdict is correct. A
final callback checks admission and finite task gradients immediately before
optimizer mutation. Informative updates require mixed genuine rewards, changed
intended parameters, unchanged frozen parameters, and generation from updated
weights. Equal-reward groups remain reported zero-contribution collections.

## Running the implementation

Install the optional LoRA profile and prepare the pinned tokenizer/configuration.
The compatibility probe downloads nothing and loads no model weights:

```bash
uv sync --locked --extra train-lora
uv run --extra train-lora hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --include '*.json' --include '*.jinja' --cache-dir outputs/model-cache
uv run --extra train-lora coding-rl train --config configs/train-cpu.yaml --probe
```

Before sampling held-out tasks, declare this evaluation profile. Prepare weights
and the normal Docker worker, then obtain a genuine development edit/test/verdict
and the fixed four-attempt baseline:

```bash
docker build -f docker/Dockerfile -t coding-rl-sandbox:stage-a .
uv run --extra train-lora hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --include '*.json' --include '*.jinja' --include '*.safetensors' \
  --cache-dir outputs/model-cache
uv run --extra train-lora coding-rl rollout --config configs/train-cpu.yaml --task zero-timeout
uv run --extra train-lora coding-rl eval --config configs/train-cpu.yaml
```

Set `training.development_outcome` to the completed rollout's `*-outcome.json`.
Its trace must contain visible test feedback and an accepted edit. A valid
unsuccessful repair is sufficient. `training.baseline_report` defaults to
`outputs/baseline.json`; it must have complete valid held-out coverage and the
matching evaluation profile. These are checked before allocating training weights.
An unsuccessful development attempt without an edit or test feedback does not
satisfy B-close; develop prompts/budgets on the training partition.

```bash
uv run --extra train-lora coding-rl train --config configs/train-cpu.yaml
```

The pilot stops after at most eight complete groups. `verified_update` means at
least one informative update produced a verified reloadable checkpoint;
`no_informative_update` leaves the learning gate incomplete and returns exit 2.
Infrastructure/admission/resource failures return `incomplete` and exit 2.
Cancellation returns 130. Probe-only success returns 0 but is not training evidence.

For full training, copy the config, set `training.mode: full`, and install/use
`--extra train-cpu`. Each mode has its own measured feasibility and real-update
gate. Neither mode is certified on the target model by fixture tests.

Run artifacts live in `outputs/training/<run-id>/`: compatibility/config records,
correlated group summaries, exact native/tensor/probability audits, update evidence,
and committed checkpoints. Temporary roots and Docker names are registered before
launch. The parent removes only those names/roots and incomplete checkpoint stages.
An interrupted run preserves previously completed checkpoints. Training resume is
intentionally unsupported; loading weights for evaluation does not restore an
optimizer.

For evaluation, copy the same config and change only the policy source and report
path, retaining the original model/base and tokenizer revisions:

```yaml
model:
  # Retain the base model/revisions/cache settings from the training config.
  source: lora  # or full
  checkpoint: outputs/training/<run-id>/checkpoints/step-1
evaluation:
  split: eval
  attempts_per_task: 4
  report_path: outputs/trained-evaluation.json
```

```bash
uv run --extra train-lora coding-rl eval --config configs/eval-trained.yaml \
  --baseline outputs/baseline.json
```

`configs/eval-trained.yaml` is the user-created copy described above. A profile
mismatch is rejected before evaluation sampling. Comparisons report both valid
denominators, infrastructure exclusions, costs, per-task verdicts, and the pass-rate
change when both denominators are nonzero. Local identity uses verified file
content, and LoRA includes its immutable base. Evaluation loads the original pinned
tokenizer rather than adopting the native training template.

## Executed evidence and remaining gates

The retained [compatibility report](evidence/phase3-native-compatibility.json)
records the installed signatures, source/template/parser hashes, schemas and native
configuration. `model_feasibility: not_run` and `runtime_certified: false` distinguish
that report from a target-model acceptance certificate.

Validation commands:

```bash
uv run --extra train-lora pytest
CODING_RL_NATIVE_TESTS=1 uv run --extra train-lora pytest -m native
CODING_RL_DOCKER_TESTS=1 uv run --extra train-lora pytest -m docker
uv run --extra train-lora ruff check .
uv build
```

The combined native/Docker suite passed **163 tests** with eight upstream warnings
that `environment_factory` is experimental. Ruff, the locked dependency check,
source/wheel builds, wheel fixture/model-free import checks, and the no-extra Docker
image inspection passed. See [the validation record](evidence/phase3-validation.json).

Native tests require the pinned tokenizer prepared above. The tiny model includes
both recurrent and full-attention layers. Full/LoRA tests use real native backward,
optimizer updates and artifact reload; scripted reference completions are confined
to tests and never enter the production training path. Parent tests cover forced
termination during simulated generation/tests/grading/save and removal of a live
registered Docker worker. The new no-extra worker image was built and inspected;
it contains pytest and no Torch, Transformers, TRL, Datasets, or PEFT.

| Gate | Status |
|---|---|
| B-close | Docker effective-limit tests passed; genuine target-model development episode/baseline not run |
| C1 | Locked stack, installed signatures, tokenizer/parser/template/batch checks passed; target-model RAM/forward/backward gate not run |
| C2 | Adapter, public rows, four-tool discovery, independent resets and split rejection tested |
| C3–C4 | Installed native fixture grouping, cutoff/failure rejection, exact tensors/probabilities and owned-resource cleanup tested; target-model runtime acceptance outstanding |
| C5 | Fixture optimizer and subsequent updated collection verified; genuine informative target-model update not run |
| C6–C7 | Full/LoRA fixture save/reload and inference loading tested; target-model reload and matched held-out comparison not run |

Only the tokenizer/configuration was downloaded during implementation. No target
weights, expensive CPU pilot, broad coding result, CUDA support, or training resume
are implied by this delivery.

| Device / parameter mode | Implementation | Target-model acceptance |
|---|---|---|
| CPU FP32 full | Native fixture update/save/reload tested | Not run; memory experiment |
| CPU FP32 LoRA | Native fixture update/save/reload tested | Not run; default pilot mode |
| CUDA / lower precision | No supported profile | Not tested |
