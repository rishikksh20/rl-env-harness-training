import inspect
import json
import shutil
from pathlib import Path

import pytest

from coding_rl import TaskCatalog, load_config
from coding_rl.config import EnvironmentConfig, SandboxConfig
from coding_rl.environment import CodingEnvironment, tool_schemas
from coding_rl.errors import ActionError
from coding_rl.patches import apply_diff, make_diff


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "seed: [",
        "seed: true",
        "unknown: 1",
        "sandbox: []",
        "sandbox:\n  network: true",
        "environment:\n  max_tool_calls: -1",
        "sandbox:\n  kind: auto",
        "logging:\n  trace_dir: null",
    ],
)
def test_bad_config_is_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_default_config_and_numeric_validation():
    config = load_config("configs/default.yaml")
    assert config.sandbox.kind == "docker"
    assert config.environment.max_tool_calls == 6
    for value in (True, 0, -1, "1", float("inf"), float("nan")):
        with pytest.raises(ValueError):
            SandboxConfig(cpus=value)
    with pytest.raises(ValueError):
        EnvironmentConfig(verifier_retries=9)


def test_public_dataset_and_split():
    catalog = TaskCatalog()
    assert len(catalog.tasks("train")) == 3 and len(catalog.tasks("eval")) == 1
    rows = catalog.dataset_rows()
    assert all(set(row) == {"task_id", "prompt"} for row in rows)
    assert "expected" not in json.dumps(rows) and "references" not in json.dumps(rows)
    assert all(
        task.starter_hash == catalog.get(task.task_id).starter_hash for task in catalog.tasks()
    )


def test_only_allowed_tool_schemas_match_actual_methods():
    schemas = tool_schemas()
    assert set(schemas) == {"list_files", "read_file", "apply_patch", "run_tests"}
    for name, schema in schemas.items():
        parameters = inspect.signature(getattr(CodingEnvironment, name)).parameters
        assert set(schema["parameters"]["properties"]) == parameters.keys() - {"self"}
        assert schema["parameters"]["additionalProperties"] is False
    assert schemas["read_file"]["parameters"]["required"] == ["path"]


def test_fixture_hash_validation(tmp_path):
    source = Path(__file__).parents[1] / "src/coding_rl/fixtures"
    fixtures = tmp_path / "fixtures"
    shutil.copytree(source, fixtures)
    (fixtures / "zero-timeout/starter/timeout.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        TaskCatalog(fixtures)


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/../private\n+++ b/../private\n@@ -1 +1 @@\n-a\n+b\n",
        "--- a/main.py\n+++ b/renamed.py\n@@ -1 +1 @@\n-a\n+b\n",
        "diff --git a/main.py b/main.py\nold mode 100644\nnew mode 100755\n",
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+new\n",
        "--- a/main.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n",
        "--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-wrong\n+b\n",
        "--- a/main.py\n+++ b/main.py\n@@ -1,2 +1 @@\n-a\n+b\n",
    ],
)
def test_restricted_patch_rejections(patch):
    files = {"main.py": "a\n"}
    with pytest.raises(ActionError):
        apply_diff(files, ("main.py",), patch)
    assert files == {"main.py": "a\n"}


def test_multiple_hunks_and_files_round_trip():
    before = {"a.py": "".join(f"line{i}\n" for i in range(30)), "b.py": "b\n"}
    after = {
        "a.py": before["a.py"].replace("line1\n", "first\n").replace("line28\n", "last\n"),
        "b.py": "before\nb\nafter\n",
    }
    patch = make_diff(before, after)
    assert apply_diff(before, ("a.py", "b.py"), patch) == after
    assert before["b.py"] == "b\n"


def test_atomic_multi_file_rejection():
    before = {"a.py": "a\n", "b.py": "b\n"}
    after = {"a.py": "new-a\n", "b.py": "new-b\n"}
    patch = make_diff(before, after).replace("-b\n", "-incorrect\n")
    with pytest.raises(ActionError):
        apply_diff(before, ("a.py", "b.py"), patch)
    assert before == {"a.py": "a\n", "b.py": "b\n"}


def test_oversized_hunk_numbers_are_action_errors():
    patch = "--- a/main.py\n+++ b/main.py\n@@ -" + "9" * 5000 + " +1 @@\n-a\n+b\n"
    with pytest.raises(ActionError):
        apply_diff({"main.py": "a\n"}, ("main.py",), patch)
