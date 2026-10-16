"""Phase 4 regression contracts; authored data here never certifies real learning."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from coding_rl.training.adapter import environment_factory
from coding_rl.training.evidence import validate_evaluation, validate_outcome
from coding_rl.training.trainer import GroupOutcome, RunOwner
from config import ModelConfig, TrainingConfig
from harnesses.protocol import ProtocolError, parse_response
from models.backend import Generation
from tasks.catalog import TaskCatalog
from tasks.dataset import import_tasks


def test_lfm_literal_parser_preserves_patches_and_rejects_execution():
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-f('old')\n+f('new')\n"
    raw = "<|tool_call_start|>[apply_patch(patch=" + repr(patch) + ")]<|tool_call_end|><|im_end|>"
    parsed = parse_response(raw, generation_prefix="<|im_start|>assistant\n")
    assert parsed.call.arguments["patch"] == patch
    for body in (
        "[__import__('os')]",
        "[obj.method()]",
        "[read_file(path=str(1))]",
        "[run_tests(), run_tests()]",
        "[run_tests(**{})]",
    ):
        with pytest.raises(ProtocolError):
            parse_response(
                "<|tool_call_start|>" + body + "<|tool_call_end|>",
                generation_prefix="<|im_start|>assistant\n",
            )


def test_first_native_reset_and_reused_reset_have_current_identity(config, tmp_path):
    config = replace(config, model=ModelConfig(), training=TrainingConfig())
    owner = RunOwner(config, "a" * 32, tmp_path)
    env = environment_factory(config, TaskCatalog(), owner, None)()
    try:
        for position in range(2):
            env.reset("zero-timeout")
            record = json.loads(
                (env._core.recorder.root / f"{env._core.episode_id}.jsonl")
                .read_text()
                .splitlines()[0]
            )
            assert record["policy"]["kind"] == "native_training"
            assert record["policy"]["native_position"] == position
            assert env._attempt.trace_directory == str(env._core.recorder.root)
            env.get_reward()
    finally:
        owner.cleanup()


def test_duplicate_completions_correlate_advantages_once(config, tmp_path):
    torch = pytest.importorskip("torch")
    owner = RunOwner(replace(config, training=TrainingConfig()), "a" * 32, tmp_path)
    owner.pending = GroupOutcome("b" * 32, 0, [], status="admitted")
    owner.tokenizer = type("Tokens", (), {"pad_token_id": 0})()
    owner.rows = [
        {"prompt_ids": [1], "completion_ids": [2], "tool_mask": [1], "reward": r}
        for r in [1, 0, 1, 0]
    ]
    advantage = 0.5 / ((1 / 3) ** 0.5 + 1e-4)
    inputs = {
        k: torch.tensor(v)
        for k, v in {
            "prompt_ids": [[1]] * 4,
            "completion_ids": [[2]] * 4,
            "prompt_mask": [[1]] * 4,
            "completion_mask": [[1]] * 4,
            "tool_mask": [[1]] * 4,
            "advantages": [-advantage, advantage, -advantage, advantage],
        }.items()
    }
    owner.audit_tensors(inputs)
    assert owner.tensor_checked
    inputs["advantages"].fill_(advantage)
    with pytest.raises(Exception, match="advantage"):
        owner.audit_tensors(inputs)


def test_empty_baseline_and_rewardless_development_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        validate_evaluation(
            {
                "evaluation_profile": {"attempts_per_task": 4},
                "attempts": [],
                "metrics": {"valid": 0, "planned": 0},
            },
            complete=True,
        )
    with pytest.raises(ValueError, match="outcome"):
        validate_outcome({"reward": None, "completed": True}, TaskCatalog().get("zero-timeout"), {})


def task_row(task_id, split):
    return {
        "task_id": task_id,
        "split": split,
        "prompt": "Preserve the supplied value.",
        "module": "value.py",
        "function": "value",
        "editable": ["value.py"],
        "files": {
            "value.py": f"# {task_id}\ndef value(x):\n    return x or 1\n",
            "test_visible.py": "from value import value\ndef test_value():\n    assert value(2) == 2\n",
        },
        "private_cases": [{"kwargs": {"x": 0}, "expected": 0}],
    }


def test_custom_dataset_import_is_atomic_and_public_rows_hide_answers(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = [task_row("own-train", "train"), task_row("own-eval", "eval")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    destination = tmp_path / "dataset"
    assert import_tasks(source, destination)["train"] == 1
    catalog = TaskCatalog(destination)
    assert "expected" not in json.dumps(catalog.dataset_rows())
    rows[1]["files"] = rows[0]["files"]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="cross train/eval"):
        import_tasks(source, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
    assert not list(tmp_path.glob(".dataset-*"))


class FixtureBackend:
    identity = {"kind": "test", "fixture": "authored read/final"}
    startups = []

    def start(self):
        pass

    def close(self):
        pass

    def render(self, messages, *, tools=None):
        self.has_tool = any(m["role"] == "tool" for m in messages)
        return [1], "<|im_start|>assistant\n"

    def count_text(self, text):
        return len(text) // 4

    def generate(self, *args, **kwargs):
        text = (
            "Done.<|im_end|>"
            if self.has_tool
            else "<|tool_call_start|>[read_file(path='timeout.py')]<|tool_call_end|><|im_end|>"
        )
        return Generation("fixture", [2], text, "eos", 0.01)


def test_custom_subprocess_uses_bridge_and_preserves_grading(config, tmp_path, monkeypatch):
    from environments.coding import make_environment
    from harnesses.external import run_harness_episode

    monkeypatch.setattr("harnesses.external.validate_model_mode", lambda config: None)
    config = replace(config, model=ModelConfig())
    env = make_environment(config)
    script = Path("examples/custom_harness.py").resolve()
    try:
        outcome = run_harness_episode(
            env,
            FixtureBackend(),
            "zero-timeout",
            driver="custom",
            command=[sys.executable, str(script)],
        )
        assert outcome.completed and outcome.reward == 0
        assert outcome.core_calls == 1 and outcome.turns == 2
        assert outcome.driver_kind == "external_harness"
        assert env.workspace is None
    finally:
        env.cleanup()


def test_bridge_requires_capability_and_exposes_only_four_tools(config):
    import urllib.error
    import urllib.request

    from environments.coding import make_environment
    from harnesses.bridge import HarnessBridge

    env = make_environment(config)
    bridge = HarnessBridge(env, FixtureBackend(), "zero-timeout", 0)
    bridge.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bridge.url + "/v1/models")
        assert error.value.code == 401
        tools = bridge.mcp({"method": "tools/list"})["tools"]
        assert {t["name"] for t in tools} == {"list_files", "read_file", "apply_patch", "run_tests"}
        with pytest.raises(ValueError):
            bridge.mcp({"method": "get_reward"})
    finally:
        bridge.close()
        env.cleanup()
