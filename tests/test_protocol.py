import json
from dataclasses import replace

import pytest

from config import (
    EnvironmentConfig,
    ModelConfig,
    RolloutConfig,
    load_config,
    validate_model_mode,
)
from harnesses.protocol import (
    ProtocolError,
    ToolCall,
    function_tools,
    parse_response,
    validate_call,
)

PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def xml_call(name, **arguments):
    parameters = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def test_exact_public_tools_and_multiline_parameters():
    tools = function_tools()
    assert {t["function"]["name"] for t in tools} == {
        "read_file",
        "apply_patch",
        "run_tests",
        "list_files",
    }
    patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
    parsed = parse_response(
        xml_call("apply_patch", patch=patch) + "<|im_end|>", generation_prefix=PREFIX
    )
    assert parsed.call.arguments == {"patch": patch}
    assert parsed.call.raw_arguments == {"patch": patch}
    assert parsed.message()["tool_calls"][0]["function"]["name"] == "apply_patch"
    assert validate_call(parsed.call, EnvironmentConfig()) is None
    parsed = parse_response(
        xml_call("read_file", path="example.py", start_line="2"), generation_prefix=PREFIX
    )
    assert parsed.call.arguments["start_line"] == 2
    assert parsed.call.raw_arguments["start_line"] == "2"


@pytest.mark.parametrize(
    "raw",
    [
        "<tool_call>",
        "<function=run_tests></function>",
        xml_call("run_tests") + " suffix",
        xml_call("run_tests") + xml_call("list_files"),
        "<tool_call><function=read_file><parameter=path>a</parameter><parameter=path>b</parameter></function></tool_call>",
        "<tool_call><function=read_file><parameter=path>no end</function></tool_call>",
        "<think>" + xml_call("run_tests") + "</think>",
        xml_call("run_tests") + "<|im_start|>",
    ],
)
def test_malformed_and_multiple_calls_never_parse_as_final(raw):
    with pytest.raises(ProtocolError):
        parse_response(raw, generation_prefix=PREFIX)


@pytest.mark.parametrize(
    "name,args",
    [
        ("cleanup", {}),
        ("read_file", {}),
        ("run_tests", {"command": "anything"}),
        ("read_file", {"path": 1}),
        ("read_file", {"path": "a", "start_line": True}),
        ("read_file", {"path": "a", "max_lines": 0}),
        ("read_file", {"path": "a", "max_lines": 999999}),
        ("apply_patch", {"patch": "x" * 65537}),
        ("apply_patch", {"patch": "\ud800"}),
    ],
)
def test_typed_dispatch_rejection(name, args):
    assert validate_call(ToolCall(name, args, {}), EnvironmentConfig())


def test_boolean_xml_cannot_be_coerced_to_integer():
    parsed = parse_response(
        xml_call("read_file", path="a", start_line="true"), generation_prefix=PREFIX
    )
    assert parsed.call.arguments["start_line"] is True
    assert validate_call(parsed.call, EnvironmentConfig())


def test_final_text_and_generation_prefix():
    assert parse_response("Done.<|im_end|>", generation_prefix=PREFIX).call is None
    with pytest.raises(ValueError, match="prefix"):
        parse_response("Done", generation_prefix="")


def test_observed_checkpoint_end_of_text_suffix():
    raw = xml_call("read_file", path="timeout.py") + "<|im_end|>\n<|endoftext|>"
    parsed = parse_response(raw, generation_prefix=PREFIX)
    assert parsed.call.name == "read_file"
    assert parsed.call.arguments == {"path": "timeout.py"}
    with pytest.raises(ProtocolError):
        parse_response("<|endoftext|>" + xml_call("run_tests"), generation_prefix=PREFIX)


@pytest.mark.parametrize(
    "data",
    [
        {"revision": "main"},
        {"tokenizer_revision": None},
        {"device": "cuda"},
        {"dtype": "bfloat16"},
        {"offline": "yes"},
        {"threads": True},
    ],
)
def test_invalid_model_profiles(data):
    with pytest.raises(ValueError):
        ModelConfig(**data)


def test_phase1_and_phase2_configs(config):
    assert load_config("configs/default.yaml").model is None
    current = load_config("configs/rollout-cpu.yaml")
    assert current.model.name == "Qwen/Qwen3.5-0.8B"
    assert current.environment.episode_seconds == 900
    assert validate_model_mode(current) is None
    with pytest.raises(ValueError, match="model section"):
        validate_model_mode(config)
    with pytest.raises(ValueError, match="Docker"):
        validate_model_mode(replace(config, model=ModelConfig()))
    for kwargs in (
        {"max_turns": True},
        {"top_p": 1.1},
        {"do_sample": 1},
        {"observation_tokens": 1},
    ):
        with pytest.raises(ValueError):
            RolloutConfig(**kwargs)


def test_public_projection_excludes_protected_task_data(env):
    from dataclasses import asdict

    public = asdict(env.catalog.get("zero-timeout").public())
    assert set(public) == {"task_id", "prompt", "editable", "starter_hash"}
    assert "expected" not in json.dumps(public) and "references" not in json.dumps(public)
