# Run coding-agent RL with LFM, Qwen, OpenCode, or your own harness

This guide uses the commands implemented in this repository. Start with CPU FP32
and LoRA. The runnable profiles are:

| Purpose | Config | Model |
|---|---|---|
| Native training and its manual baseline | `configs/train-lfm2.5-350m.yaml` | `LiquidAI/LFM2.5-350M` |
| OpenCode rollout/evaluation | `configs/opencode-lfm2.5-350m.yaml` | Same LFM policy; larger harness context reserve |
| Native training and its manual baseline | `configs/train-cpu.yaml` | `Qwen/Qwen3.5-0.8B` |
| Manual inference only | `configs/rollout-cpu.yaml` | Qwen, greedy by default |

Both model families have explicit loaders, tokenizer/token contracts, tool
parsers, and full/LoRA checkpoint reload paths. Changing models requires changing
the profile, model ID, model revision, and tokenizer revision together.

**OpenCode runs the agent loop for rollout and evaluation. Native TRL runs the
agent loop during gradient training.** Both use the same four environment tools
and independent verifier. OpenCode does not expose the exact native learning
tensors and policy synchronization needed by this trainer; its transcripts are
not accepted as GRPO batches or native training prerequisites. Your custom
external harness has the same evaluation capability. To customize training,
change the shared environment/task catalog and keep the native adapter contract.

LFM is a small pipeline-development model. Its publisher recommends it for tool
use but advises against programming; successful coding learning is an empirical
gate, not guaranteed by its size. See the
[model card](https://huggingface.co/LiquidAI/LFM2.5-350M). The measured LFM run here
passed LoRA capacity checks and model/tool integration, but did not establish a
genuine informative GRPO update. Consult [Phase 4 status](PHASE_4_STATUS.md) before
interpreting a test pass or an exit code as learning progress.

## 1. Prepare the Python environment and Docker worker

Run every command from the repository root. Use Python 3.11+; the validation host
uses Python 3.12 and Linux with Docker/cgroup v2.

```bash
uv sync --locked --extra train-lora
source .venv/bin/activate
python --version
coding-rl --help
docker info
docker build -f docker/Dockerfile -t coding-rl-sandbox:stage-a .
coding-rl smoke --config configs/default.yaml
```

The application entry point is now `src/main.py`; domain packages are described
in [the code layout guide](CODE_LAYOUT.md). Run `uv sync` again after pulling the
reorganization so `coding-rl` is refreshed. With the environment activated, the
following are equivalent:

```bash
coding-rl tasks --split train
python -m main tasks --split train
python src/main.py tasks --split train
```

The activation keeps the locked training dependencies available for the remaining
commands. If you prefer `uv run`, use `uv run --extra train-lora coding-rl ...`
throughout. `--extra train-cpu` installs full training without PEFT; the LoRA extra
supports both modes.

The sandbox image contains the Python worker and fixed visible-test dependencies.
It contains no model, reference repairs, or protected expected answers. Generated
code executes inside Docker with no network and bounded memory, CPU, processes,
output, and time. Do not use `trusted_local` for real model episodes.

## 2. Select and download the pinned model

For LFM:

```bash
cp configs/train-lfm2.5-350m.yaml configs/lfm-local.yaml
coding-rl model-fetch --config configs/lfm-local.yaml
coding-rl model-probe --config configs/lfm-local.yaml
```

`model-fetch` explicitly downloads the pinned model/tokenizer snapshots into
`outputs/model-cache`. Subsequent commands honor `model.offline: true`.
`model-probe` loads the actual policy, warms generation, asks it to read source,
and records raw tokens and the parsed call. It does not grade or train. Look for
`read_gate_passed: true`; output is also saved under the configured trace root.

The supplied immutable identities are:

```yaml
# LFM
model:
  profile: lfm2.5
  name: LiquidAI/LFM2.5-350M
  revision: 9e6c6ccf47cd318696e137d381a7ded8fe4df09f
  tokenizer_revision: 9e6c6ccf47cd318696e137d381a7ded8fe4df09f
```

```yaml
# Qwen
model:
  profile: qwen3.5
  name: Qwen/Qwen3.5-0.8B
  revision: 2fc06364715b967f1860aea9cf38778875588b17
  tokenizer_revision: 2fc06364715b967f1860aea9cf38778875588b17
```

Keep the rest of the copied profile. Both use CPU, float32, four threads, and
offline loading. CUDA, quantized gradient training, and vLLM are not implemented
profiles. No automatic model or sandbox fallback occurs.

LFM uses Python-style calls inside its tool markers. The adapter normalizes
wrapped tool definitions to LFM's flat format and decodes literal arguments
through a public parser extension. It preserves patch newlines and never executes
generated Python to parse a call. Qwen uses its XML tool-call format.

## 3. Check the native API and actual memory capacity

```bash
coding-rl train --config configs/lfm-local.yaml --probe
coding-rl train --config configs/lfm-local.yaml --feasibility
```

These commands do different jobs:

| Operation | Loads weights? | Updates weights? | Needs development/baseline? |
|---|---|---|---|
| `train --probe` | No | No | No |
| `model-probe` | Yes | No | No |
| `train --feasibility` | Yes | Disposable diagnostic Adam step | No |
| `train` | Yes | Admitted native GRPO steps | Yes |

The API probe checks the installed TRL interface, parser, patch round trip,
four-tool contract/defaults, public prompt budget, and complete-group batch layout.
The feasibility command runs in a supervised child. It checks estimated capacity
before loading, actual padded/independent behavior, a four-sequence forward and
backward at the longest public prompt plus the configured completion region,
every present gradient, optimizer allocation, and save/reload agreement. The child
then exits. Diagnostic weights, optimizer state, and staging files are discarded.
Training starts again from the pinned base.

Reports are under `training.output_dir/<run-id>/`, including `run.json` and
`compatibility.json`. Expect `feasibility_passed` for the capacity operation.
This is a capacity result for the declared batch/length/mode, not a correctness
or learning result.

The measured LFM LoRA layout used four sequences of 967 tokens and peaked around
10.4 GiB RSS; a larger layout exceeded a 10 GiB configured cap. Prompt changes
change the measured length, so run feasibility on your current configuration.
Keep RAM available for the OS and Docker too. The trainer's RSS limit is polled
process supervision; the Docker candidate worker has separate kernel limits.

Important native settings in the LFM profile:

```yaml
training:
  mode: lora
  group_size: 4
  max_groups: 8
  completion_tokens: 384
  context_tokens: 1536
  observation_tokens: 128
  tool_iterations: 6
  lora_rank: 8
  lora_alpha: 16
  lora_targets: [q_proj, v_proj]
  memory_limit_mb: 12288
  wall_seconds: 3600
```

The completion region includes native tool feedback and template suffixes.
Increasing it can help multi-turn repairs but increases memory. Recheck capacity
after changing it. `rollout.*` controls manual and external evaluation;
`training.*` controls native training. There is no implicit translation between
these budgets. The initial pilot supports exactly four attempts per group and at
most eight groups, with one group per loss call and one process.

## 4. Know which coding environment and dataset you are using

The default environment is **this project's stateful Python repository repair
environment**. Each episode gets a fresh tiny repository and these tools:

| Tool | Effect |
|---|---|
| `list_files` | List allowed public paths |
| `read_file` | Read bounded file contents |
| `apply_patch` | Apply an exact unified diff to existing editable source |
| `run_tests` | Run the fixed visible pytest suite in a fresh Docker snapshot |

The bundled dataset is authored and included in the wheel; no remote coding
dataset is downloaded automatically:

| Task | Split | Coding objective |
|---|---|---|
| `zero-timeout` | Train | Preserve an explicit zero timeout; distinguish it from missing/None |
| `empty-default` | Train | Preserve false, zero, and empty values instead of treating them as absent |
| `range-boundary` | Train | Implement the specified inclusive/exclusive interval boundaries |
| `chunk-tail` | Eval | Preserve the final short chunk |

```bash
coding-rl tasks --config configs/lfm-local.yaml --split train
coding-rl tasks --config configs/lfm-local.yaml --split eval
coding-rl tasks --config configs/lfm-local.yaml --split train \
  --output outputs/public-training-rows.jsonl
```

TRL receives public conversations and `task_id` only. Reset resolves the ID to an
immutable starter and its protected grading plan. The final files are frozen;
the verifier reconstructs protected files, runs visible tests, then checks
candidate function outputs against private JSON cases using fresh workers.

Reward is binary: 1 for independently verified correctness, 0 for a healthy
incorrect submission. Infrastructure failures have no reward. Passing visible
pytest alone is insufficient, and saying “fixed” does not change the submitted
files. These four tasks measure integration mechanics, not general coding-agent
ability. Source/test creation, dependency installation, arbitrary shell access,
and arbitrary repository test commands are outside this environment's contract.

## 5. Collect a genuine development episode

```bash
mkdir -p outputs
coding-rl rollout --config configs/lfm-local.yaml \
  --task zero-timeout --attempt 0 > outputs/lfm-development.json
python - <<'PY'
import json
r = json.load(open('outputs/lfm-development.json'))
print({k: r[k] for k in ('completed', 'reward', 'accepted_edits',
                         'finish_reason', 'trace_directory', 'episode_id')})
PY
```

Before training, the development evidence must have a completed, cleaned episode
on a **train** task, at least one accepted edit, healthy model-visible `run_tests`
feedback, an independent verdict, and a matching frozen-source/trace identity.
Reward 0 can qualify if the repair is incorrect but the edit/test/verifier path
actually ran. A read-only probe or a rejected patch cannot qualify.

Inspect the emitted trace directory. Relevant files are:

```text
<episode-id>.jsonl
<episode-id>-conversation.json
<episode-id>-submission.json
<episode-id>-outcome.json
```

Use another development sample or task if the small model fails to edit:

```bash
coding-rl rollout --config configs/lfm-local.yaml \
  --task zero-timeout --attempt 1 > outputs/lfm-development-attempt-1.json
coding-rl rollout --config configs/lfm-local.yaml \
  --task empty-default --attempt 0 > outputs/lfm-development-empty.json
coding-rl rollout --config configs/lfm-local.yaml \
  --task range-boundary --attempt 0 > outputs/lfm-development-range.json
```

`--attempt` deterministically selects a development sampling stream without
changing the held-out evaluation schedule. Choose a real qualifying outcome;
do not hand-edit its JSON or substitute a reference repair. The measured LFM
development sample attempted a patch, received a rejection, then claimed success;
private grading correctly gave 0 and the training gate refused that sample.
If your samples consistently fail this gate, use Qwen or improve the public tool
instructions on training tasks and collect fresh evidence.

Set the qualifying outcome path in your local config. For example, if the first
command above qualified:

```bash
python - <<'PY'
from pathlib import Path
import yaml
p = Path('configs/lfm-local.yaml')
c = yaml.safe_load(p.read_text())
c['training']['development_outcome'] = 'outputs/lfm-development.json'
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
```

## 6. Collect the fixed held-out baseline

The LFM training profile enables sampling and schedules four attempts on the
held-out task. Preserve these settings for after-training evaluation.

```bash
coding-rl eval --config configs/lfm-local.yaml
python - <<'PY'
import json
r = json.load(open('outputs/lfm-baseline.json'))
print(r['metrics'])
assert r['metrics']['planned'] == r['metrics']['valid'] == 4
assert r['metrics']['not_started'] == r['metrics']['incomplete'] == 0
PY
```

The report includes actual attempts, seeds, policy and tokenizer identities,
resolved sandbox image, task/grading identities, coverage, verified pass rate,
parser/call counts, per-task costs, and timings. Four valid failures form a valid
baseline. They do not demonstrate useful coding ability.

Training validates every attempt and recomputes coverage and metrics. An empty
report with a handwritten pass counter is rejected. Keep the base model, seed,
sampling, evaluation tasks, limits, tool template, and worker identity unchanged
for comparison. Regenerate the baseline if you change evaluator behavior.

```bash
python - <<'PY'
from pathlib import Path
import yaml
p = Path('configs/lfm-local.yaml')
c = yaml.safe_load(p.read_text())
c['training']['baseline_report'] = 'outputs/lfm-baseline.json'
p.write_text(yaml.safe_dump(c, sort_keys=False))
PY
```

## 7. Run the native RL pilot

After both evidence gates qualify:

```bash
coding-rl train --config configs/lfm-local.yaml | tee outputs/lfm-train-command.json
```

The default LoRA run trains only the requested attention projections. Native TRL
generates four independent attempts at one train task, invokes the four tools,
obtains independently verified rewards, computes group-relative advantages, and
updates the policy. The controller admits the whole group before backward and
checks actual token IDs, masks, probabilities, gradients, frozen weights, and
parameter changes. Training never imports a reference solution.

An informative binary-reward group contains both 0 and 1. All-zero/all-one groups
have zero task advantage. After earlier updates, optimizer momentum can still
affect weights on such a step; inspect recorded deltas. The implementation saves
a checkpoint only for an admitted informative update. It does not manufacture a
positive reward or a checkpoint when the model cannot solve anything.

The final command JSON prints the run directory. Inspect:

```text
outputs/lfm-training/<run-id>/
  config.json
  compatibility.json
  lora.json
  groups.json
  summary.json
  <collection-id>-collection.json
  <collection-id>-native.json
  <collection-id>-tensors.json
  <collection-id>-probabilities.json
  checkpoints/step-<n>/checkpoint.json
  run.json
```

Raw available collections and explicit rejection reasons are retained even when
admission fails. Native stop reasons unavailable through the public reward hook
are marked unavailable. `summary.json` reports statuses, informative groups,
rewards, and group reward variance. No private expected answers enter these files.

| Status | Meaning |
|---|---|
| `verified_update` | At least one admitted informative update produced a reload-validated checkpoint |
| `no_informative_update` | The bounded run finished without such a checkpoint; inspect rewards |
| `incomplete` | Gate, resource, audit, infrastructure, or save failure; inspect error fields |
| `cancelled` | Interrupted run; no unfinished group is admitted as usable evidence |

Exit codes are 0 for a verified update or successful explicit probe, 2 for an
incomplete/no-informative run, and 130 for cancellation. The run report is the
authoritative record; `tee` can hide the producer's exit code unless your shell
uses `set -o pipefail`.

Ctrl-C requests shutdown; the parent also enforces wall/RAM budgets and removes
registered containers, writable private roots, and partial checkpoint staging.
Earlier committed checkpoints survive. Checkpoints contain model or adapter
weights and evidence, not optimizer resume state. Resume is unsupported.

## 8. Evaluate a committed checkpoint against the baseline

Create an evaluation config from the completed run. This script refuses a run
without a committed checkpoint:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml
command = json.load(open('outputs/lfm-train-command.json'))
run = json.loads((Path(command['directory']) / 'run.json').read_text())
if not run.get('checkpoints'):
    raise SystemExit('No committed informative checkpoint. Inspect run.json and groups.json.')
checkpoint = run['checkpoints'][-1]
c = yaml.safe_load(Path('configs/lfm-local.yaml').read_text())
c['model'].update(source=checkpoint['source'], checkpoint=checkpoint['path'])
c['evaluation']['report_path'] = 'outputs/lfm-trained-eval.json'
Path('configs/lfm-trained-eval.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
coding-rl eval --config configs/lfm-trained-eval.yaml \
  --baseline outputs/lfm-baseline.json
```

Full checkpoints load their own weights; LoRA checkpoints reconstruct the pinned
base and attach the saved adapter. Both retain the pinned base tokenizer. New
checkpoint manifests use schema 2 and validate update/reload evidence and file
digests. Schema 1 artifacts lack the new validated proof contract and are rejected
with an explicit version error; an old directory cannot be relabeled as schema 2.

The after-report contains `comparison`, including coverage and pass-rate change.
Different profiles are rejected before sampling. Using the baseline as the
after-report output path is also rejected. A verified update may make held-out
performance worse; report the measured result.

For full training, create a separate profile and check it independently:

```bash
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('configs/lfm-local.yaml').read_text())
c['training'].update(mode='full', output_dir='outputs/lfm-full-training')
Path('configs/lfm-full.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
coding-rl train --config configs/lfm-full.yaml --feasibility
# Run this only after feasibility and development/baseline admission pass:
coding-rl train --config configs/lfm-full.yaml
```

Reuse base development/baseline evidence only when its actual evaluator
profile is unchanged. Full mode has its own capacity and learning gates; LoRA
capacity does not certify it.

## 9. Run OpenCode with the same local policy and environment

Install OpenCode if needed. Version 1.15.13 was exercised here. One project-local
installation option is:

```bash
npm install --prefix outputs/opencode-runtime opencode-ai@1.15.13
export PATH="$PWD/outputs/opencode-runtime/node_modules/.bin:$PATH"
opencode --version
coding-rl model-fetch --config configs/opencode-lfm2.5-350m.yaml
coding-rl harness-rollout --config configs/opencode-lfm2.5-350m.yaml \
  --driver opencode --task zero-timeout --attempt 0
coding-rl harness-eval --config configs/opencode-lfm2.5-350m.yaml --driver opencode
```

The controller starts a loopback model endpoint and an authenticated MCP endpoint
for one episode, then starts OpenCode in an empty temporary directory with
isolated configuration/data locations. It supplies a custom local provider and
only the four `coding_rl_*` MCP tools. Built-in filesystem, shell, network,
subagent, and other tools are disabled. No candidate workspace path or private
grading interface is given to OpenCode. The external harness itself is trusted
host code; this bridge does not sandbox the OpenCode executable.

OpenCode's supported
[custom providers](https://docs.opencode.ai/docs/providers/),
[MCP configuration](https://docs.opencode.ai/docs/mcp-servers/), and
[tool permissions](https://docs.opencode.ai/docs/tools/) provide the integration.
The local endpoint implements the text chat/tool subset needed here. Stream
responses are buffered until native generation finishes; this is a lab bridge,
not a general inference service. Model sampling comes from the lab config.

The controller keeps ownership of reset, budgets, freeze, grading, and cleanup.
Process stdout/stderr and exit status are retained in
`<episode-id>-harness-process.json`. A model's rejected patch is visible in tool
feedback and never becomes an accepted edit. An external controller failure has
no correctness reward.

The OpenCode baseline is `outputs/opencode-lfm-eval.json`. It is separate from the
manual baseline required by native training. To compare OpenCode before/after,
copy its config, set `model.source` and `model.checkpoint` to the committed
checkpoint as in step 8 and choose a different evaluation report path:

```bash
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('configs/opencode-lfm2.5-350m.yaml').read_text())
trained = yaml.safe_load(Path('configs/lfm-trained-eval.yaml').read_text())
c['model'] = trained['model']
c['evaluation']['report_path'] = 'outputs/opencode-lfm-trained-eval.json'
Path('configs/opencode-lfm-trained.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
coding-rl harness-eval --config configs/opencode-lfm-trained.yaml \
  --driver opencode --baseline outputs/opencode-lfm-eval.json
```

Use the same OpenCode version and harness contract on both sides. External
harness identity is included in the evaluation profile, so changing it invalidates
that comparison. This path evaluates the trained policy through OpenCode; it
does not use OpenCode as TRL's training sampler.

## 10. Replace OpenCode with your own harness, including Qwen

The working example is [examples/custom_harness.py](../examples/custom_harness.py).
It receives only these controller-provided environment variables:

| Variable | Meaning |
|---|---|
| `CODING_RL_BRIDGE_URL` | Per-episode loopback URL |
| `CODING_RL_BRIDGE_TOKEN` | Per-episode bearer capability |
| `CODING_RL_TASK` | JSON file with public messages, task ID, and turn limit |

It requests inference through `POST /v1/chat/completions`, discovers tools through
MCP `tools/list`, calls them through MCP `tools/call`, returns observations to the
model, and exits. The controller then freezes and independently grades. There
is no public reset/reward/workspace endpoint. Keep this script as trusted
controller code and replace its conversation strategy as needed.

```bash
coding-rl harness-rollout --config configs/lfm-local.yaml --driver custom \
  --task zero-timeout -- python "$PWD/examples/custom_harness.py"
```

For Qwen, start from its complete profile:

```bash
cp configs/train-cpu.yaml configs/qwen-local.yaml
coding-rl model-fetch --config configs/qwen-local.yaml
coding-rl model-probe --config configs/qwen-local.yaml
coding-rl train --config configs/qwen-local.yaml --probe
coding-rl train --config configs/qwen-local.yaml --feasibility
coding-rl harness-rollout --config configs/qwen-local.yaml --driver custom \
  --task zero-timeout -- python "$PWD/examples/custom_harness.py"
```

Create a separate custom-harness baseline config:

```bash
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('configs/qwen-local.yaml').read_text())
c['evaluation']['report_path'] = 'outputs/qwen-custom-baseline.json'
Path('configs/qwen-custom-eval.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
coding-rl harness-eval --config configs/qwen-custom-eval.yaml --driver custom \
  -- python "$PWD/examples/custom_harness.py"
```

To train Qwen, repeat steps 5–8 with `configs/qwen-local.yaml`, Qwen-specific
development/baseline/report paths, and the **manual** `rollout`/`eval` commands.
Retain a separate baseline for custom-harness comparison. The custom command and
hashes of file arguments form part of harness identity; use the same script and
absolute paths before/after. Arguments after `--` are executed as argv, without a
shell. The example uses only Python's standard library.

## 11. Add your own coding dataset

Use local Python repair tasks with JSON-serializable keyword inputs and expected
return values. The importer accepts one JSON object per line, with exactly:

```text
task_id, split, prompt, module, function, editable, files, private_cases
```

`files` maps allowed relative paths to exact starter text. Include
`test_visible.py`, source, and any small public support files. `private_cases`
contains `{kwargs, expected}` objects for the named function. Each task needs a
nonempty private case set. Reference repairs are unnecessary. Keep train and
evaluation families separate; exact duplicate starter/callable identities across
splits are rejected, while semantic split quality remains your responsibility.

Create a small example catalog:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [
    {'task_id': 'preserve-zero', 'split': 'train',
     'prompt': 'Fix choose(x=None): use 9 only for None and preserve zero.',
     'module': 'choice.py', 'function': 'choose', 'editable': ['choice.py'],
     'files': {'choice.py': 'def choose(x=None):\n    return x or 9\n',
               'test_visible.py': 'from choice import choose\ndef test_positive():\n    assert choose(2) == 2\n'},
     'private_cases': [{'kwargs': {'x': 0}, 'expected': 0},
                       {'kwargs': {'x': None}, 'expected': 9}]},
    {'task_id': 'keep-final-character', 'split': 'eval',
     'prompt': 'Fix copy_text(text): preserve the entire input string.',
     'module': 'text_copy.py', 'function': 'copy_text', 'editable': ['text_copy.py'],
     'files': {'text_copy.py': 'def copy_text(text):\n    return text[:-1]\n',
               'test_visible.py': 'from text_copy import copy_text\ndef test_empty():\n    assert copy_text("") == ""\n'},
     'private_cases': [{'kwargs': {'text': 'abc'}, 'expected': 'abc'},
                       {'kwargs': {'text': 'z'}, 'expected': 'z'}]},
]
Path('outputs/own-coding-tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
PY
coding-rl dataset-import --input outputs/own-coding-tasks.jsonl \
  --output outputs/own-coding-dataset
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('configs/train-lfm2.5-350m.yaml').read_text())
c['task_root'] = 'outputs/own-coding-dataset'
c['training']['output_dir'] = 'outputs/own-coding-training'
c['training']['baseline_report'] = 'outputs/own-coding-baseline.json'
c['evaluation']['report_path'] = 'outputs/own-coding-baseline.json'
Path('configs/own-coding.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY
coding-rl tasks --config configs/own-coding.yaml --split train \
  --output outputs/own-public-training-rows.jsonl
coding-rl train --config configs/own-coding.yaml --probe
coding-rl rollout --config configs/own-coding.yaml --task preserve-zero
```

Repeat feasibility, development selection, baseline, training, and checkpoint
comparison using this config. Changing `task_root` changes the actual train/eval
catalog and grading identities. Native training builds its public dataset from
that catalog automatically. **Do not feed the importer JSONL directly to a model
or trainer**: it contains protected expected values. The `tasks --output` command
exports only public rows. Keep the input and imported private files on the trusted
controller, outside candidate workspaces and public task prompts.

This functional profile is a starting curriculum for coding-agent RL. Add varied
small repair families with independent regression and edge cases, then larger
held-out families. Do not increase an eight-group plumbing pilot into a claimed
coding benchmark result.

For real repository issues, the next environment targets are
[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) for training environments and
[SWE-bench Verified](https://www.swebench.com/) for held-out evaluation. SWE-Gym's
[paper](https://arxiv.org/abs/2412.21139) describes executable training environments;
SWE-bench Verified contains 500 human-filtered issues. They require per-repository
images/dependencies, checkout identities, task-specific test commands and test
patch handling, richer file edits, and a repository test-result verifier. Raw
SWE-Gym/SWE-bench rows **cannot be loaded into this functional JSON importer**.
No such adapter or benchmark score is claimed here. HumanEval/MBPP-style
function synthesis also needs a deliberate conversion into buggy-starter repair
tasks before it fits this environment.

## 12. Troubleshooting and evidence interpretation

| Result | Next action |
|---|---|
| Missing pinned snapshot | Run `model-fetch` for the same profile/cache/revisions |
| EOS/template/profile error | Restore the complete supplied model profile; changing the model ID alone is insufficient |
| `B-close` prerequisite error | Inspect the real development trace and baseline attempts; ensure edit, healthy tests, verdict, identities, and complete coverage |
| Visible tests pass but reward is 0 | Inspect the frozen source and private verdict status; visible tests intentionally miss edge cases |
| Patch rejected | Use exact `--- a/file`, `+++ b/file`, hunk counts/context, and a final newline; no source-only replacement or Git/apply-patch preamble |
| Native cutoff rejection | Budget includes tool observations; increase native completion/context capacity and rerun feasibility |
| RAM/wall limit | Inspect parent error and measured RSS; reduce sequence budgets, use LoRA/smaller model, or move to a host with capacity |
| No informative groups | The samples have no mixed rewards; improve coding/tool competence or curriculum and collect a fresh bounded pilot |
| External process fails | Inspect the saved harness-process artifact and driver/version/script paths |
| Before/after profiles differ | Restore the original evaluator settings or collect a fresh baseline under the changed profile |

First-token timing is measured at the Transformers streamer callback after
prefill. Decode throughput uses subsequent token emissions; aggregate generation
throughput includes prefill. Peak RSS is a process high-water measurement.
Unavailable native hook counters are explicitly labeled. Test-generated
completions validate integration and optimizer mechanics; only genuine generated
episodes and admitted informative updates can establish learning evidence.
