"""The installed application must find nested code and its task assets anywhere."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from utils.provenance import source_fingerprints


@pytest.mark.parametrize("entry", ["console", "module", "script"])
def test_entry_points_find_packaged_tasks_outside_repository(tmp_path, entry):
    root = Path(__file__).resolve().parents[1]
    command = {
        "console": [str(Path(sys.executable).with_name("coding-rl"))],
        "module": [sys.executable, "-m", "main"],
        "script": [sys.executable, str(root / "src" / "main.py")],
    }[entry]
    result = subprocess.run(
        [*command, "tasks", "--split", "eval"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["tasks"] == ["chunk-tail"]


def test_provenance_covers_nested_packages_without_fixture_or_dependency_code(tmp_path):
    files = {
        "main.py": "# entry point\n",
        "config.py": "# configuration\n",
        "coding_rl/training/worker.py": "# training worker\n",
        "models/worker.py": "# inference worker\n",
        "environments/execution/worker.py": "# candidate worker\n",
        "utils/process.py": "# process supervision\n",
        "tasks/fixtures/sample/starter/code.py": "# candidate starter\n",
        "third_party/worker.py": "# dependency\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    before = source_fingerprints(tmp_path)
    assert set(before) == {
        name for name in files if "fixtures" not in name and "third_party" not in name
    }
    assert len({value for name, value in before.items() if name.endswith("worker.py")}) == 3
    path = tmp_path / "models/worker.py"
    path.write_text("# changed inference worker\n")
    after = source_fingerprints(tmp_path)
    assert {name for name in before if before[name] != after[name]} == {"models/worker.py"}
