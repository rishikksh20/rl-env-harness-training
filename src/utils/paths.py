"""Canonical repository-relative path validation."""

from pathlib import PurePosixPath


def relative_path(path: str, *, root: bool = False) -> str:
    if not isinstance(path, str) or "\x00" in path or "\\" in path:
        raise ValueError("Use a repository-relative POSIX path")
    if root and path in {"", "."}:
        return "."
    parts = path.split("/")
    if not path or PurePosixPath(path).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("Use a canonical relative path without parent segments")
    return path
