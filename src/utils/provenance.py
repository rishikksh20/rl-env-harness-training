"""Fingerprint this application's nested packages without scanning dependencies."""

import hashlib
from pathlib import Path

SOURCE_ROOTS = (
    "main.py",
    "config.py",
    "coding_rl",
    "environments",
    "harnesses",
    "models",
    "tasks",
    "utils",
)


def source_fingerprints(root: Path | None = None) -> dict[str, str]:
    root = root or Path(__file__).resolve().parents[1]
    records = {}
    for name in SOURCE_ROOTS:
        source = root / name
        for path in [source] if source.is_file() else sorted(source.rglob("*.py")):
            relative = path.relative_to(root)
            if "fixtures" not in relative.parts:
                records[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records
