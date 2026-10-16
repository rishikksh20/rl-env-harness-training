"""Native TRL setup and admission through public reward, loss, and callback hooks."""

import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from coding_rl.training.adapter import NativeEnvironment, environment_factory, training_rows
from coding_rl.training.checkpoints import publish_checkpoint
from config import Config
from environments.coding import TOOLS, tool_schemas
from environments.verifier import grading_fingerprint
from models.profiles import (
    configure_chat_template,
    model_class,
    model_profile,
    validate_tokenizer,
)
from tasks.catalog import TaskCatalog
from utils.errors import InfrastructureError
from utils.provenance import source_fingerprints
from utils.records import digest_record, write_record

TRAINING_PINS = {"trl": "1.16.0", "datasets": "5.1.0", "accelerate": "1.15.0"}


@dataclass
class Attempt:
    collection_id: str
    policy_version: int
    position: int
    task_id: str
    starter_hash: str
    grading_fingerprint: str
    episode_id: str
    observations: list[dict] = field(default_factory=list)
    reward: float | None = None
    submission_hash: str | None = None
    failure: str | None = None
    trace_directory: str | None = None


@dataclass
class GroupOutcome:
    collection_id: str
    policy_version: int
    attempts: list[Attempt]
    status: str = "collecting"
    informative: bool = False
    tensor_artifact: str | None = None
    update: dict = field(default_factory=dict)
    checkpoint: str | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class TrainingOutcome:
    status: str
    run_id: str
    groups: list[dict]
    checkpoints: list[dict]
    compatibility: dict
    error: str | None = None


def dependency_report(mode: str) -> dict:
    from models.backend import dependency_versions

    versions = dependency_versions()
    for name, expected in {
        **TRAINING_PINS,
        **({"peft": "0.21.2"} if mode == "lora" else {}),
    }.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InfrastructureError(
                "Missing native training packages; run uv sync --locked --extra "
                + ("train-lora" if mode == "lora" else "train-cpu")
            ) from exc
        if versions[name] != expected:
            raise InfrastructureError(f"Unsupported {name}; install the locked training extra")
    return versions


def native_arguments(config: Config, output_dir: Path) -> dict:
    """One complete group per loss call permits admission before any backward."""
    settings = config.training
    return {
        "output_dir": str(output_dir),
        "use_cpu": True,
        "per_device_train_batch_size": settings.group_size,
        "gradient_accumulation_steps": 1,
        "generation_batch_size": settings.group_size,
        "num_generations": settings.group_size,
        "num_iterations": 1,
        "max_steps": settings.max_groups,
        "max_completion_length": settings.completion_tokens,
        "max_tool_calling_iterations": settings.tool_iterations,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "beta": 0.0,
        "learning_rate": settings.learning_rate,
        "weight_decay": 0.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "constant",
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": False,
        "disable_dropout": True,
        "use_vllm": False,
        "torch_compile": False,
        "use_transformers_continuous_batching": False,
        "mask_truncated_completions": False,
        "pad_to_multiple_of": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "save_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "logging_steps": 1,
        "seed": config.seed,
        "data_seed": config.seed,
        "dataloader_pin_memory": False,
    }


def tool_methods(environment) -> list:
    return [
        getattr(environment, name)
        for name, _ in inspect.getmembers(type(environment), inspect.isfunction)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]


def configure_native_parser(tokenizer):
    """Use the supported response-template configuration to preserve patch whitespace."""
    from trl.chat_template_utils import add_response_schema

    if "<|tool_call_start|>" in tokenizer.chat_template:
        from transformers.utils.chat_parsing.content_parsers import CONTENT_PARSERS
        from trl.chat_template_utils import lfm2_2_5_template

        from harnesses.protocol import literal_tool_calls

        # The pinned Hub template is newer than TRL's exact-string registry.
        # Configure the public parser field explicitly and verify round trips below.
        tokenizer.response_template = copy.deepcopy(lfm2_2_5_template)
        # Upstream's lax JSON parser preserves literal \\n in single-quoted
        # patches. Decode the whole Python call list through the exported registry.
        CONTENT_PARSERS["coding_rl_python_calls"] = literal_tool_calls
        tokenizer.response_template["fields"]["tool_calls"] = {
            "open": "<|tool_call_start|>",
            "close": "<|tool_call_end|>",
            "content": "coding_rl_python_calls",
        }
        return tokenizer
    tokenizer = add_response_schema(tokenizer)
    template = copy.deepcopy(tokenizer.response_template)
    arguments = template["fields"]["tool_calls"]["content_args"]
    # The upstream Qwen pattern and text parser both strip all whitespace. Paired
    # framing newlines are optional; payload indentation and its final newline survive.
    arguments["tag_pattern"] = (
        r"<parameter=(?P<key>[^>\n]+)>(?P<frame>\n)?(?P<value>.*?)(?(frame)\n|)</parameter>"
    )
    arguments["value_parser"] = {"name": "text", "args": {"strip": False}}
    tokenizer.response_template = template
    return tokenizer


class NativePresenter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def compatibility_probe(config: Config, directory: Path) -> tuple[dict, object]:
    """Probe the installed release with the real pinned tokenizer, without weights."""
    versions = dependency_report(config.training.mode)
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema
    from trl import GRPOConfig, GRPOTrainer
    from trl.chat_template_utils import get_training_chat_template, parse_response

    model = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model.name,
        revision=model.tokenizer_revision,
        cache_dir=model.cache_dir,
        local_files_only=model.offline,
        trust_remote_code=False,
        token=False,
    )
    validate_tokenizer(model, tokenizer)
    hub_template = tokenizer.chat_template
    configure_chat_template(model, tokenizer)
    original = tokenizer.chat_template
    tokenizer = configure_native_parser(tokenizer)
    native_template = get_training_chat_template(tokenizer) or original
    tokenizer.chat_template = native_template
    tokenizer.padding_side = "left"
    # Schemas need bound methods; use an uninitialized instance without allocating an episode.
    instance = NativeEnvironment.__new__(NativeEnvironment)
    methods = tool_methods(instance)
    schemas = [get_json_schema(method) for method in methods]
    if {schema["function"]["name"] for schema in schemas} != set(TOOLS):
        raise InfrastructureError("Native tool discovery differs from the four-tool allowlist")
    for schema in schemas:
        function = schema["function"]
        expected = tool_schemas()[function["name"]]["parameters"]
        parameters = function["parameters"]
        if {name: value["type"] for name, value in parameters["properties"].items()} != {
            name: value["type"] for name, value in expected["properties"].items()
        } or set(parameters.get("required", [])) != set(expected["required"]):
            raise InfrastructureError("Discovered native argument schema differs from the core")
        signature = inspect.signature(getattr(instance, function["name"]))
        for name, argument in signature.parameters.items():
            expected_default = expected["properties"][name].get("default", inspect.Parameter.empty)
            if argument.default != expected_default:
                raise InfrastructureError(
                    "Discovered native argument default differs from the core"
                )
        if any(
            p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for p in signature.parameters.values()
        ):
            raise InfrastructureError("Native tools must reject extra arguments")
    rows = training_rows(TaskCatalog(config.task_root))
    row = rows[0]
    for item in rows:
        ids = tokenizer.apply_chat_template(
            item["prompt"],
            tools=methods,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=False,
        )
        if len(ids) + config.training.completion_tokens > config.training.context_tokens:
            raise InfrastructureError("Native public prompt exceeds the declared context reserve")
    prefix = tokenizer.apply_chat_template(
        row["prompt"],
        tools=methods,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    if len(prefix) + config.training.completion_tokens > config.training.context_tokens:
        raise InfrastructureError("Native public prompt exceeds the declared context reserve")
    patch = "--- a/timeout.py\n+++ b/timeout.py\n@@ -1 +1 @@\n-old\n+new\n"
    probes = {
        "read": "<tool_call>\n<function=read_file>\n<parameter=path>timeout.py</parameter>\n</function>\n</tool_call><|im_end|>",
        "patch": "<tool_call>\n<function=apply_patch>\n<parameter=patch>"
        + patch
        + "</parameter>\n</function>\n</tool_call><|im_end|>",
        "boolean": "<tool_call>\n<function=read_file>\n<parameter=path>timeout.py</parameter>\n<parameter=start_line>true</parameter>\n</function>\n</tool_call><|im_end|>",
        "final": "Finished.<|im_end|>",
    }
    if model.profile == "lfm2.5":
        probes.update(
            read="<|tool_call_start|>[read_file(path='timeout.py')]<|tool_call_end|><|im_end|>",
            patch="<|tool_call_start|>[apply_patch(patch="
            + repr(patch)
            + ")]<|tool_call_end|><|im_end|>",
            boolean="<|tool_call_start|>[read_file(path='timeout.py', start_line=True)]<|tool_call_end|><|im_end|>",
        )
    parsed = {
        key: parse_response(
            tokenizer,
            tokenizer.encode(text, add_special_tokens=False),
            prefix=prefix,
            tools=methods,
        )
        for key, text in probes.items()
    }
    if parsed["read"].get("tool_calls", [{}])[0].get("function", {}).get("name") != "read_file":
        raise InfrastructureError("Native Qwen read parser failed")
    arguments = parsed["patch"].get("tool_calls", [{}])[0].get("function", {}).get("arguments", {})
    if arguments.get("patch") != patch:
        raise InfrastructureError("Native parser changed multiline patch bytes")
    boolean = parsed["boolean"].get("tool_calls", [{}])[0].get("function", {}).get("arguments", {})
    if type(boolean.get("start_line")) is int:
        raise InfrastructureError("Native parser coerces a boolean into an admissible line integer")
    if parsed["final"].get("tool_calls"):
        raise InfrastructureError("Native parser misclassified a final response")
    # Check actual prefix token preservation around tool feedback.
    call = parsed["read"]
    before = tokenizer.apply_chat_template(
        row["prompt"] + [call],
        tools=methods,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    after = tokenizer.apply_chat_template(
        row["prompt"] + [call, {"role": "tool", "name": "read_file", "content": "{}"}],
        tools=methods,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if after[: len(before)] != before:
        raise InfrastructureError("Selected native Qwen template does not preserve prefixes")
    resolved = GRPOConfig(**native_arguments(config, directory / "native"))
    if resolved.steps_per_generation != 1 or resolved.generation_batch_size != 4:
        raise InfrastructureError(
            "Native batch layout is incompatible with complete-group admission"
        )
    source = Path(inspect.getfile(GRPOTrainer)).read_bytes()
    report = {
        "schema_version": 1,
        "versions": versions,
        "transitive_versions": {"fsspec": importlib.metadata.version("fsspec")},
        "dependency_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
        if Path("uv.lock").exists()
        else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "constructor": str(inspect.signature(GRPOTrainer)),
        "compute_loss": str(inspect.signature(GRPOTrainer.compute_loss)),
        "config": str(inspect.signature(GRPOConfig)),
        "trainer_source_sha256": hashlib.sha256(source).hexdigest(),
        "original_template_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "hub_template_sha256": hashlib.sha256(hub_template.encode()).hexdigest(),
        "native_template_sha256": hashlib.sha256(native_template.encode()).hexdigest(),
        "parser": "trl.parse_response/" + model.profile,
        "response_template_sha256": digest_record(tokenizer.response_template),
        "parser_results": parsed,
        "parser_evidence": "authored protocol probes, not generated model evidence",
        "generation_prefix": tokenizer.decode(prefix)[
            tokenizer.decode(prefix).rfind("<|im_start|>assistant\n") :
        ],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tool_schemas": schemas,
        "effective_tool_contract": tool_schemas(),
        "native_arguments": native_arguments(config, directory / "native"),
        "limits": {
            "entered_core_calls": config.environment.max_tool_calls,
            "proposed_unknown_malformed_calls": "unavailable; no native pre-dispatch hook",
            "calls_per_turn": "sequential native calls; atomic single-call enforcement unsupported",
            "tool_iterations": config.training.tool_iterations,
            "completion_tokens": "native region includes tool suffixes; cutoff groups rejected",
            "context_tokens": "initial prompt + completion region bounded before collection",
            "manual_rollout_settings": "not translated into native limits",
            "observation_tokens": config.training.observation_tokens,
            "wall_seconds": config.training.wall_seconds,
            "wall_owner": "parent process; whole training run, not a manual generation timeout",
        },
        "model_feasibility": "not_run",
        "runtime_certified": False,
        "sampling_seed": config.seed,
        "random_stream": "native seeded batched multinomial draws; positions recorded per collection",
    }
    write_record(directory / "compatibility.json", report)
    return report, tokenizer


def tool_suffix_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Tokenize only the external suffix; never rebuild a learning sequence from JSON."""
    dummy = [
        {"role": "user", "content": "dummy"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": messages[0]["name"], "arguments": {}}},
            ],
        },
    ]
    prefix = tokenizer.apply_chat_template(
        dummy,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        dummy + messages,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    end = max(i for i, token in enumerate(prefix) if token == tokenizer.eos_token_id) + 1
    if full[:end] != prefix[:end]:
        raise InfrastructureError("Native tool suffix is not prefix preserving")
    return full[end:]


def audit_sequence(
    tokenizer, ids: list[int], messages: list[dict], prompt: list[int], tools
) -> list[int]:
    """Annotate actual native IDs with independently checked external-token spans."""
    from trl.chat_template_utils import parse_response

    mask, cursor, index = [], 0, 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            raise InfrastructureError("Unexpected native message order")
        try:
            end = ids.index(tokenizer.eos_token_id, cursor) + 1
        except ValueError as exc:
            raise InfrastructureError("Partial native completion is ineligible") from exc
        parsed = parse_response(
            tokenizer, ids[cursor:end], prefix=prompt + ids[:cursor], tools=tools
        )
        if parsed.get("tool_calls") != message.get("tool_calls"):
            raise InfrastructureError("Native calls disagree with actual completion IDs")
        mask.extend([1] * (end - cursor))
        cursor, index = end, index + 1
        observations = []
        while index < len(messages) and messages[index].get("role") == "tool":
            observations.append(messages[index])
            index += 1
        if observations:
            suffix = tool_suffix_ids(tokenizer, observations)
            if ids[cursor : cursor + len(suffix)] != suffix:
                raise InfrastructureError(
                    "Executed tool result is absent from native completion IDs"
                )
            mask.extend([0] * len(suffix))
            cursor += len(suffix)
    if cursor != len(ids) or not mask or not any(mask):
        raise InfrastructureError("Native completion has unaccounted or empty learning positions")
    return mask


def validate_native_observations(env, messages: list[dict]) -> None:
    """Match effects once; allow only reproducible pre-dispatch native errors."""
    expected = []
    recorded = iter(env._attempt.observations)
    methods = {method.__name__: method for method in tool_methods(env)}
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            name, arguments = function.get("name"), function.get("arguments")
            method = methods.get(name)
            error = None
            if method is None:
                error = f"Tool {name} not found."
            else:
                try:
                    inspect.signature(method).bind(**arguments)
                except TypeError:
                    # Binding proved this cannot enter the adapter or cause effects.
                    try:
                        method(**arguments)
                    except TypeError as exc:
                        error = str(exc)
            observation = (
                {"name": name, "content": str({"error": error})} if error else next(recorded, None)
            )
            if observation is None or observation["name"] != name:
                raise InfrastructureError("Native cutoff/error removed an executed observation")
            expected.append(observation)
    observed = [
        {"name": m["name"], "content": m["content"]} for m in messages if m["role"] == "tool"
    ]
    if observed != expected or next(recorded, None) is not None:
        raise InfrastructureError("Native cutoff/error removed or changed an executed observation")


class RunOwner:
    def __init__(self, config: Config, run_id: str, directory: Path):
        self.config, self.run_id, self.directory = config, run_id, directory
        self.resource_root = directory / "private"
        self.resource_root.mkdir(exist_ok=True, mode=0o700)
        self.cores = []
        self.groups: list[GroupOutcome] = []
        self.pending: GroupOutcome | None = None
        self.policy_version = 0
        self.rows = []
        self.tokenizer = None
        self.tensor_checked = False
        self.probability_checked = False

    def register(self, core, task_id: str) -> Attempt:
        if self.pending is None:
            self.pending = GroupOutcome(uuid.uuid4().hex, self.policy_version, [])
            self.groups.append(self.pending)
            self.tensor_checked = self.probability_checked = False
        group = self.pending
        if group.status != "collecting" or len(group.attempts) >= self.config.training.group_size:
            raise InfrastructureError("Native collection reused stale or incomplete policy data")
        task = core.catalog.get(task_id)
        attempt = Attempt(
            group.collection_id,
            self.policy_version,
            len(group.attempts),
            task.task_id,
            task.starter_hash,
            grading_fingerprint(task),
            "",
            trace_directory=str(core.recorder.root),
        )
        group.attempts.append(attempt)
        core.recorder.policy = {
            "kind": "native_training",
            "run_id": self.run_id,
            "collection_id": group.collection_id,
            "policy_version": self.policy_version,
            "native_position": attempt.position,
        }
        return attempt

    def fail(self, attempt: Attempt, reason: str) -> None:
        attempt.failure = attempt.failure or reason
        if self.pending is not None:
            self.pending.status = "rejected"

    def check_failures(self) -> None:
        if any(attempt.failure for group in self.groups for attempt in group.attempts):
            raise InfrastructureError("Native collection has a latched infrastructure failure")

    def persist(self) -> None:
        write_record(self.directory / "groups.json", [asdict(group) for group in self.groups])
        write_record(
            self.directory / "summary.json",
            {
                "groups": len(self.groups),
                "informative_groups": sum(g.informative for g in self.groups),
                "statuses": {
                    s: sum(g.status == s for g in self.groups)
                    for s in sorted({g.status for g in self.groups})
                },
                "rewards": [a.reward for g in self.groups for a in g.attempts],
                "reward_variance": [
                    sum((a.reward - sum(x.reward for x in g.attempts) / 4) ** 2 for a in g.attempts)
                    / 4
                    if len(g.attempts) == 4 and all(a.reward is not None for a in g.attempts)
                    else None
                    for g in self.groups
                ],
            },
        )

    def admission_reward(self, prompts, completions, completion_ids, environments, **metadata):
        """A zero-weight admission source runs before the native environment rewards."""
        try:
            if self.pending is not None:
                write_record(
                    self.directory / f"{self.pending.collection_id}-collection.json",
                    {
                        "prompts": prompts,
                        "completions": completions,
                        "completion_ids": completion_ids,
                        "stop_reasons": "not exported by the pinned native reward hook",
                    },
                )
            self.check_failures()
            group = self.pending
            if group is None or len(group.attempts) != 4 or len(environments) != 4:
                raise InfrastructureError("Native collection is not a complete four-member group")
            if [env._attempt for env in environments] != group.attempts:
                raise InfrastructureError(
                    "Native completion positions differ from registered attempts"
                )
            if (
                any(env._core.workspace is None for env in environments)
                or len({env._core.workspace for env in environments}) != 4
            ):
                raise InfrastructureError("Native attempts must own four distinct workspaces")
            identities = {
                (a.task_id, a.starter_hash, a.grading_fingerprint, a.policy_version)
                for a in group.attempts
            }
            if len(identities) != 1 or len({a.episode_id for a in group.attempts}) != 4:
                raise InfrastructureError("Native group mixes task/policy identities or workspaces")
            self.rows = []
            for env, prompt, messages, ids in zip(
                environments,
                prompts,
                completions,
                completion_ids,
                strict=True,
            ):
                validate_native_observations(env, messages)
                initial = training_rows(env._core.catalog)
                public = next(
                    row["prompt"] for row in initial if row["task_id"] == env._attempt.task_id
                )
                if prompt[: len(public)] != public:
                    raise InfrastructureError(
                        "Native prompt differs from its public task projection"
                    )
                methods = tool_methods(env)
                prefix = self.tokenizer.apply_chat_template(
                    public,
                    tools=methods,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if (
                    len(prefix) + self.config.training.completion_tokens
                    > self.config.training.context_tokens
                ):
                    raise InfrastructureError("Native context reserve exceeded")
                mask = audit_sequence(self.tokenizer, ids, messages, prefix, methods)
                reward = env.get_reward()
                self.rows.append(
                    {
                        "prompt_ids": prefix,
                        "completion_ids": ids,
                        "tool_mask": mask,
                        "reward": reward,
                        "episode_id": env._attempt.episode_id,
                    }
                )
            self.check_failures()
            group.informative = {a.reward for a in group.attempts} == {0.0, 1.0}
            group.status = "admitted"
            write_record(self.directory / f"{group.collection_id}-native.json", self.rows)
            self.persist()
            return [0.0] * 4  # Only the native environment column supplies binary rewards.
        except BaseException as exc:
            if self.pending is not None:
                self.pending.status = "rejected"
                self.pending.rejection_reason = f"{type(exc).__name__}: {exc}"
                write_record(
                    self.directory / f"{self.pending.collection_id}-native.json", self.rows
                )
            self.persist()
            raise

    def audit_tensors(self, inputs) -> None:
        import torch

        self.check_failures()
        if self.pending is None or self.pending.status != "admitted":
            raise InfrastructureError("No admitted complete group before loss")
        if inputs["completion_ids"].shape[0] != 4:
            raise InfrastructureError("Loss must see the entire group before any backward")
        remaining = list(self.rows)
        rewards = [row["reward"] for row in self.rows]
        mean = sum(rewards) / 4
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 3)
        for index in range(4):
            pm, cm = inputs["prompt_mask"][index], inputs["completion_mask"][index]
            for mask in (pm, cm, inputs["tool_mask"][index]):
                if not bool(((mask == 0) | (mask == 1)).all()):
                    raise InfrastructureError("Non-binary native mask")
            p, c = int(pm.sum()), int(cm.sum())
            if (
                p == 0
                or c == 0
                or pm.tolist() != [0] * (len(pm) - p) + [1] * p
                or cm.tolist() != [1] * c + [0] * (len(cm) - c)
            ):
                raise InfrastructureError(
                    "Native padding/truncation mask changed sequence positions"
                )
            prompt = inputs["prompt_ids"][index][-p:].tolist()
            ids = inputs["completion_ids"][index][:c].tolist()
            if not bool(
                (inputs["prompt_ids"][index][:-p] == self.tokenizer.pad_token_id).all()
            ) or not bool(
                (inputs["completion_ids"][index][c:] == self.tokenizer.pad_token_id).all()
            ):
                raise InfrastructureError("Native padding IDs differ from the fixed tokenizer")
            matches = [
                row
                for row in remaining
                if row["prompt_ids"] == prompt and row["completion_ids"] == ids
            ]
            if not matches:
                raise InfrastructureError("Actual trainer tensors differ from admitted native IDs")
            candidates = [
                row
                for row in matches
                if torch.allclose(
                    inputs["advantages"][index],
                    torch.tensor(
                        (row["reward"] - mean) / (std + 1e-4),
                        dtype=inputs["advantages"].dtype,
                        device=inputs["advantages"].device,
                    ),
                    atol=1e-5,
                    rtol=1e-5,
                )
                and inputs["tool_mask"][index][:c].tolist() == row["tool_mask"]
            ]
            if not candidates:
                raise InfrastructureError(
                    "Native advantage or loss mask is associated with the wrong attempt"
                )
            row = candidates[0]
            remaining.remove(row)
            if inputs["tool_mask"][index][:c].tolist() != row["tool_mask"]:
                raise InfrastructureError(
                    "Actual native loss mask includes external tokens or excludes actions"
                )
            if not torch.allclose(
                inputs["advantages"][index],
                torch.tensor(
                    (row["reward"] - mean) / (std + 1e-4),
                    dtype=inputs["advantages"].dtype,
                    device=inputs["advantages"].device,
                ),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise InfrastructureError("Native advantage is associated with the wrong attempt")
        for key, value in inputs.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise InfrastructureError(f"Non-finite native tensor: {key}")
        artifact = f"{self.pending.collection_id}-tensors.json"
        write_record(
            self.directory / artifact,
            {
                key: value.detach().cpu().tolist()
                for key, value in inputs.items()
                if torch.is_tensor(value)
            },
        )
        self.pending.tensor_artifact = artifact
        self.tensor_checked = True

    def cleanup(self) -> None:
        errors = []
        for core in self.cores:
            try:
                try:
                    if core.phase == "active":
                        core.finish("infrastructure_failure")
                finally:
                    core.cleanup()
            except (OSError, InfrastructureError) as exc:
                errors.append(str(exc))
        if errors:
            raise InfrastructureError("Native core cleanup failed")


def parameter_digest(model, *, frozen: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == frozen:
            continue
        digest.update(name.encode())
        digest.update(memoryview(parameter.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def feasibility_probe(model, tokenizer, config: Config) -> dict:
    import psutil
    import torch

    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    minimum = parameters * 4 + trainable * 12
    limit = config.training.memory_limit_mb * 1048576
    if minimum > limit or minimum > psutil.virtual_memory().available:
        raise InfrastructureError(
            "FP32 weight/gradient/Adam minimum exceeds the measured host budget"
        )
    started = time.monotonic()
    model.eval()
    ids = torch.tensor([[1, 2, 3], [tokenizer.pad_token_id, 1, 2]])
    attention = torch.tensor([[1, 1, 1], [0, 1, 1]])
    with torch.no_grad():
        padded = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        for index, length in ((0, 3), (1, 2)):
            independent = model(
                input_ids=ids[index : index + 1, -length:],
                attention_mask=torch.ones(1, length, dtype=torch.long),
                use_cache=False,
            ).logits
            torch.testing.assert_close(
                padded[index, -length:], independent[0], atol=1e-4, rtol=1e-4
            )
    model.train()
    forward_started = time.monotonic()
    output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    loss = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, output.logits.size(-1)), torch.tensor([2, 3])
    )
    if not torch.isfinite(loss):
        raise InfrastructureError("Non-finite native feasibility forward")
    forward_seconds = time.monotonic() - forward_started
    backward_started = time.monotonic()
    loss.backward()
    backward_seconds = time.monotonic() - backward_started
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if (
        not gradients
        or not all(bool(torch.isfinite(g).all()) for g in gradients)
        or not any(bool(g.any()) for g in gradients)
    ):
        raise InfrastructureError("Native feasibility backward has no finite nonzero gradient")
    model.zero_grad(set_to_none=True)
    from utils.process import peak_rss_bytes

    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "fp32_adam_minimum_bytes": minimum,
        "forward_backward_seconds": time.monotonic() - started,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "rss_bytes": psutil.Process().memory_info().rss,
        "peak_rss_bytes": peak_rss_bytes(),
        "host_available_ram_bytes": psutil.virtual_memory().available,
        "independent_padding_tolerance": {"atol": 1e-4, "rtol": 1e-4},
        "passed": True,
    }


def validate_prerequisites(config: Config, catalog: TaskCatalog, worker: str, policy: dict) -> dict:
    from coding_rl.rollout.evaluate import evaluation_profile
    from coding_rl.training.evidence import (
        validate_development_trace,
        validate_evaluation,
        validate_outcome,
    )

    settings = config.training
    if not settings.baseline_report or not settings.development_outcome:
        raise InfrastructureError(
            "B-close requires training.baseline_report and training.development_outcome from real Docker model runs"
        )
    baseline = json.loads(Path(settings.baseline_report).read_text())
    development = json.loads(Path(settings.development_outcome).read_text())
    for record in (baseline, development):
        identity = record.get("policy", {})
        if (
            identity.get("kind") != "model"
            or identity.get("checkpoint") != config.model.name
            or identity.get("revision") != config.model.revision
            or identity.get("policy_source", {}).get("source") != "hub"
        ):
            raise InfrastructureError("B-close evidence must use the declared pinned base policy")
    if baseline.get("evaluation_profile") != evaluation_profile(config, catalog, worker, policy):
        raise InfrastructureError("Declared baseline does not match this evaluation profile")
    try:
        validate_evaluation(baseline, complete=True)
        for record in baseline["attempts"]:
            validate_outcome(record, catalog.get(record["task_id"]), baseline["policy"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InfrastructureError(f"Invalid B-close baseline: {exc}") from exc
    metrics = baseline["metrics"]
    if (
        baseline.get("split") != "eval"
        or metrics["valid"] != metrics["planned"]
        or metrics["not_started"]
    ):
        raise InfrastructureError("B-close baseline needs complete held-out coverage")
    if not config.rollout.do_sample or config.evaluation.attempts_per_task != 4:
        raise InfrastructureError("B-close requires four sampled attempts per held-out task")
    if not (
        development.get("driver_kind") == "model"
        and development.get("completed")
        and development.get("cleanup_complete")
        and development.get("proposed_calls", 0) > 0
        and development.get("accepted_edits", 0) > 0
        and development.get("verdict")
        and catalog.get(development["task_id"]).split == "train"
    ):
        raise InfrastructureError("B-close requires a completed genuine development edit/verdict")
    task = catalog.get(development["task_id"])
    if (
        development.get("infrastructure_error") is not None
        or development.get("starter_hash") != task.starter_hash
        or development.get("grading_fingerprint") != grading_fingerprint(task)
        or development["verdict"].get("submission_hash") != development.get("submission_hash")
        or any(development.get("policy", {}).get(key) != value for key, value in policy.items())
    ):
        raise InfrastructureError(
            "Development episode does not match the declared policy/source/verifier"
        )
    try:
        validate_outcome(development, task, policy)
        validate_development_trace(development, task, worker)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise InfrastructureError(f"Invalid B-close development evidence: {exc}") from exc
    return {
        "baseline_sha256": digest_record(baseline),
        "development_sha256": digest_record(development),
    }


def build_native_trainer(model, tokenizer, owner: RunOwner, catalog: TaskCatalog):
    """Specialize supported public hooks; generation and GRPO remain native."""
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    class AdmittedGRPOTrainer(GRPOTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            owner.audit_tensors(inputs)
            expected_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            attention = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            c = inputs["completion_ids"].shape[1]
            probabilities = []

            def record_forward(module, args, kwargs, output):
                if not torch.equal(kwargs.get("input_ids"), expected_ids) or not torch.equal(
                    kwargs.get("attention_mask"), attention
                ):
                    raise InfrastructureError(
                        "Native probability conditioning differs from admitted tensors"
                    )
                labels = expected_ids.masked_fill(attention == 0, -100)
                labels[:, :-c] = -100
                if (
                    not torch.equal(kwargs.get("labels"), labels)
                    or kwargs.get("past_key_values") is not None
                ):
                    raise InfrastructureError(
                        "Native probability labels/cache differ from admitted positions"
                    )
                logps = output.log_probs[:, -c:].detach().clone()
                if not bool(torch.isfinite(logps).all()):
                    raise InfrastructureError("Non-finite current policy probabilities")
                probabilities.append(logps)

            handle = model.register_forward_hook(record_forward, with_kwargs=True)
            try:
                loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            finally:
                handle.remove()
            if len(probabilities) != 1 or not bool(torch.isfinite(loss)):
                raise InfrastructureError("Native loss/probability export is incompatible")
            logps = probabilities[0]
            # Validate the actual native fused scores against independent unpadded execution.
            with torch.no_grad():
                for index in range(4):
                    live = attention[index].bool()
                    row_ids = expected_ids[index : index + 1, live]
                    row_labels = row_ids.clone()
                    length = int(inputs["completion_mask"][index].sum())
                    row_labels[:, :-length] = -100
                    independent = model(
                        input_ids=row_ids,
                        attention_mask=torch.ones_like(row_ids),
                        labels=row_labels,
                        fused_lm_head=True,
                        use_cache=False,
                    ).log_probs[:, -length:]
                    torch.testing.assert_close(
                        logps[index, :length], independent[0], atol=1e-4, rtol=1e-4
                    )
            old = inputs.get("old_per_token_logps")
            if old is not None:
                torch.testing.assert_close(old, logps, atol=1e-4, rtol=1e-4)
            if "ref_per_token_logps" in inputs:
                raise InfrastructureError(
                    "The beta-zero pilot must not create reference probabilities"
                )
            write_record(
                owner.directory / f"{owner.pending.collection_id}-probabilities.json",
                {
                    "current_logps": logps.cpu().tolist(),
                    "behavior": "current detached; one collection/update, no vLLM or reuse",
                    "independent_padding_atol": 1e-4,
                    "independent_padding_rtol": 1e-4,
                    "loss": float(loss.detach()),
                },
            )
            owner.probability_checked = True
            return loss

    peft_config = None
    settings = owner.config.training
    if settings.mode == "lora":
        from peft import LoraConfig

        names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in settings.lora_targets
        ]
        if not names or {name.rsplit(".", 1)[-1] for name in names} != set(settings.lora_targets):
            raise InfrastructureError(
                "LoRA targets do not resolve to every requested linear module type"
            )
        peft_config = LoraConfig(
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=0.0,
            target_modules=names,
            bias="none",
            task_type="CAUSAL_LM",
        )
        write_record(
            owner.directory / "lora.json",
            {
                "targets": names,
                "rank": settings.lora_rank,
                "alpha": settings.lora_alpha,
                "dropout": 0.0,
                "base": asdict(owner.config.model),
            },
        )
    owner.tokenizer = tokenizer
    trainer = AdmittedGRPOTrainer(
        model=model,
        args=GRPOConfig(**native_arguments(owner.config, owner.directory / "native")),
        reward_funcs=[owner.admission_reward],
        train_dataset=Dataset.from_list(training_rows(catalog)),
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=environment_factory(
            owner.config, catalog, owner, NativePresenter(tokenizer)
        ),
    )
    if trainer.accelerator.num_processes != 1:
        raise InfrastructureError("The native pilot supports exactly one training process")
    return trainer


def save_policy(trainer, owner: RunOwner, report: dict, prerequisites: dict) -> dict:
    import torch

    model = trainer.accelerator.unwrap_model(trainer.model)
    config = owner.config
    loader = model_class(config.model)
    model.eval()
    probe_ids = torch.tensor([[1, 2, 3]])
    with torch.random.fork_rng(), torch.no_grad():
        expected = model(input_ids=probe_ids, use_cache=False).logits.detach().cpu()
        generated = model.generate(
            input_ids=probe_ids,
            attention_mask=torch.ones_like(probe_ids),
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        ).tolist()
    evidence = owner.groups[-1].update
    evidence["updated_policy_generation"] = {
        "input_ids": probe_ids.tolist(),
        "output_ids": generated,
        "trainable_sha256": parameter_digest(model, frozen=False),
        "active_adapter": getattr(model, "active_adapter", None),
    }

    def reload_check(path: Path) -> dict:
        options = {"dtype": torch.float32, "trust_remote_code": False, "token": False}
        if config.training.mode == "full":
            loaded = loader.from_pretrained(path, local_files_only=True, **options)
        else:
            from peft import PeftModel

            base = loader.from_pretrained(
                config.model.name,
                revision=config.model.revision,
                cache_dir=config.model.cache_dir,
                local_files_only=True,
                **options,
            )
            loaded = PeftModel.from_pretrained(
                base, path, is_trainable=False, local_files_only=True
            )
        loaded.eval().requires_grad_(False)
        loaded.generation_config.eos_token_id = owner.tokenizer.eos_token_id
        loaded.generation_config.pad_token_id = owner.tokenizer.pad_token_id
        with torch.no_grad():
            actual = loaded(input_ids=probe_ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            output = loaded.generate(
                input_ids=probe_ids,
                attention_mask=torch.ones_like(probe_ids),
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            ).tolist()
        if output != generated:
            raise InfrastructureError("Reloaded deterministic generation differs from saved policy")
        del loaded
        return {
            "passed": True,
            "logits_atol": 1e-5,
            "logits_rtol": 1e-5,
            "input_ids": probe_ids.tolist(),
            "output_ids": output,
        }

    provenance = {
        "config_sha256": digest_record(asdict(config)),
        "source_sha256": digest_record(source_fingerprints()),
        "dependencies": report["versions"],
        "trainable_parameters": report.get("trainable_parameters", {}),
        "prerequisites": prerequisites,
        "native_template_sha256": report["native_template_sha256"],
        "response_template_sha256": report.get("response_template_sha256"),
        "evaluation_template_sha256": report["original_template_sha256"],
        "eos_token_id": owner.tokenizer.eos_token_id,
        "pad_token_id": owner.tokenizer.pad_token_id,
        "tasks": [
            {
                "task_id": task.task_id,
                "starter_hash": task.starter_hash,
                "grading_fingerprint": grading_fingerprint(task),
            }
            for task in TaskCatalog(config.task_root).tasks("train")
        ],
        "worker_identity": owner.cores[0].sandbox.identity,
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
        if Path("uv.lock").exists()
        else None,
    }
    audit_names = [
        evidence["tensor_artifact"],
        f"{evidence['collection_id']}-probabilities.json",
        f"{evidence['collection_id']}-native.json",
    ]
    provenance["audit_files"] = {
        "audit/" + name: hashlib.sha256((owner.directory / name).read_bytes()).hexdigest()
        for name in audit_names
    }

    def save(path: Path) -> None:
        model.save_pretrained(path, safe_serialization=True)
        (path / "audit").mkdir()
        for name in audit_names:
            shutil.copyfile(owner.directory / name, path / "audit" / name)

    try:
        with torch.random.fork_rng():
            checkpoint = publish_checkpoint(
                owner.directory / "checkpoints" / f"step-{trainer.state.global_step}",
                metadata={
                    "kind": config.training.mode,
                    "parent": {
                        "policy_version": owner.groups[-1].policy_version,
                        "trainable_sha256": evidence["before_sha256"],
                    },
                    "base": {
                        key: getattr(config.model, key)
                        for key in ("name", "revision", "tokenizer_revision")
                    },
                    "global_step": trainer.state.global_step,
                    "provenance": provenance,
                    "update_evidence": evidence,
                },
                save=save,
                validate_reload=reload_check,
            )
        return {"path": str(checkpoint.path), **checkpoint.identity}
    finally:
        model.train()


def update_callback(trainer, owner: RunOwner, report: dict, prerequisites: dict, checkpoints: list):
    import torch
    from transformers import TrainerCallback

    class VerifiedUpdate(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
            owner.check_failures()
            if (
                not owner.tensor_checked
                or not owner.probability_checked
                or owner.pending.status != "admitted"
            ):
                raise InfrastructureError(
                    "Optimizer admission requires a complete tensor/probability audit"
                )
            self.before = parameter_digest(model, frozen=False)
            self.frozen = parameter_digest(model, frozen=True)
            self.coordinates = []
            squared_norm = 0.0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if not bool(torch.isfinite(gradient).all()):
                    raise InfrastructureError("Non-finite task gradient before optimizer mutation")
                squared_norm += float(torch.linalg.vector_norm(gradient)) ** 2
                # Bound diagnostic copies even for the large embedding matrix.
                flat = gradient.reshape(-1)
                for start in range(0, flat.numel(), 1048576):
                    chunk = flat[start : start + 1048576]
                    position = start + int(chunk.abs().argmax())
                    if float(flat[position]) != 0:
                        self.coordinates.append(
                            (name, position, float(parameter.detach().reshape(-1)[position]))
                        )
                        break
            self.gradient_norm = math.sqrt(squared_norm)
            if owner.pending.informative and self.gradient_norm == 0:
                raise InfrastructureError("Informative group has no task-reward gradient")

        def on_optimizer_step(self, args, state, control, model, **kwargs):
            if parameter_digest(model, frozen=True) != self.frozen:
                raise InfrastructureError("Frozen base parameters changed during native update")
            parameters = dict(model.named_parameters())
            deltas = [
                abs(float(parameters[name].detach().reshape(-1)[index]) - value)
                for name, index, value in self.coordinates
            ]
            if any(not bool(torch.isfinite(p).all()) for p in parameters.values()):
                raise InfrastructureError("Non-finite policy after optimizer update")
            after = parameter_digest(model, frozen=False)
            changed = max(deltas, default=0.0)
            if owner.pending.informative and (after == self.before or changed <= 1e-12):
                raise InfrastructureError(
                    "Informative native update did not change intended parameters"
                )
            owner.pending.update = {
                "collection_id": owner.pending.collection_id,
                "policy_version": owner.pending.policy_version,
                "tensor_artifact": owner.pending.tensor_artifact,
                "global_step": state.global_step + 1,
                "task_gradient_norm": self.gradient_norm,
                "before_sha256": self.before,
                "after_sha256": after,
                "max_observed_parameter_delta": changed,
                "delta_tolerance": 1e-12,
                "frozen_sha256": self.frozen,
                "frozen_unchanged": True,
                "informative": owner.pending.informative,
            }
            owner.pending.status = "updated"
            owner.policy_version += 1

        def on_step_end(self, args, state, control, **kwargs):
            if owner.pending.status != "updated":
                raise InfrastructureError("Native step ended without optimizer evidence")
            owner.persist()
            if owner.pending.informative:
                checkpoint = save_policy(trainer, owner, report, prerequisites)
                checkpoints.append(checkpoint)
                owner.pending.checkpoint = checkpoint.get("path")
                owner.pending.status = "committed"
                owner.persist()
            else:
                owner.pending.status = "zero_contribution"
                owner.persist()
            owner.pending = None

    return VerifiedUpdate()


def execute_training(
    config: Config,
    run_id: str,
    directory: Path,
    *,
    probe_only: bool,
    feasibility_only: bool = False,
) -> TrainingOutcome:
    owner = RunOwner(config, run_id, directory)
    report, checkpoints = {}, []
    try:
        report, tokenizer = compatibility_probe(config, directory)
        if probe_only:
            return TrainingOutcome("probe_only", run_id, [], [], report)
        if feasibility_only:
            from coding_rl.training.feasibility import measure_capacity

            report["model_feasibility"] = measure_capacity(config, tokenizer, owner)
            write_record(directory / "compatibility.json", report)
            return TrainingOutcome("feasibility_passed", run_id, [], [], report)
        from environments.execution.sandbox import Sandbox
        from models.backend import PINNED_PACKAGES
        from models.worker import load_policy

        catalog = TaskCatalog(config.task_root)
        sandbox = Sandbox(config.sandbox, resource_root=owner.resource_root, run_id=run_id)
        sandbox.preflight()
        policy = {
            "template_sha256": report["original_template_sha256"],
            "protocol": model_profile(config.model).protocol,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "device": config.model.device,
            "dtype": config.model.dtype,
            "versions": {key: report["versions"][key] for key in PINNED_PACKAGES},
            "threads": config.model.threads,
        }
        prerequisites = validate_prerequisites(config, catalog, sandbox.identity, policy)
        import torch

        torch.set_num_threads(config.model.threads)
        torch.manual_seed(config.seed)
        from coding_rl.training.feasibility import preflight_capacity

        preflight_capacity(config)
        model, loading = load_policy(asdict(config.model), trainable=True)
        model.config.use_cache = False
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        trainer = build_native_trainer(model, tokenizer, owner, catalog)
        report["model_class"], report["loading"] = type(model).__name__, loading
        report["model_feasibility"] = feasibility_probe(trainer.model, tokenizer, config)
        report["trainable_parameters"] = {
            name: parameter.numel()
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }
        if not report["trainable_parameters"]:
            raise InfrastructureError("Native training has no intended trainable parameters")
        write_record(directory / "compatibility.json", report)
        trainer.add_callback(update_callback(trainer, owner, report, prerequisites, checkpoints))
        trainer.train()
        status = "verified_update" if checkpoints else "no_informative_update"
        return TrainingOutcome(
            status, run_id, [asdict(g) for g in owner.groups], checkpoints, report
        )
    except (Exception, KeyboardInterrupt) as exc:
        if owner.pending is not None:
            owner.pending.status = (
                "uncommitted" if owner.pending.status == "updated" else "rejected"
            )
            owner.pending.rejection_reason = f"{type(exc).__name__}: {exc}"
        owner.persist()
        return TrainingOutcome(
            "incomplete",
            run_id,
            [asdict(g) for g in owner.groups],
            checkpoints,
            report,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        owner.cleanup()
