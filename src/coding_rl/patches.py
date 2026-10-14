"""A deliberately narrow exact-context unified-diff parser, with no shell execution."""

import difflib
import re
from collections.abc import Mapping

from coding_rl.errors import ActionError
from coding_rl.tasks import relative_path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?\n?$")


def make_diff(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    return "".join(
        "".join(
            difflib.unified_diff(
                before[name].splitlines(True),
                after[name].splitlines(True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        for name in sorted(before)
        if before[name] != after[name]
    )


def apply_diff(files: Mapping[str, str], editable: tuple[str, ...], patch: str) -> dict[str, str]:
    """Validate all hunks before returning a new file map; input is never mutated."""
    if not isinstance(patch, str) or not patch or "\x00" in patch:
        raise ActionError("Provide a nonempty UTF-8 unified diff")
    lines = patch.splitlines(keepends=True)
    result = dict(files)
    seen = set()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- a/"):
            raise ActionError("Only --- a/path and +++ b/path headers are supported")
        old_path = lines[index][6:].rstrip("\n")
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ b/"):
            raise ActionError("Missing new-file header")
        new_path = lines[index][6:].rstrip("\n")
        index += 1
        try:
            relative_path(old_path)
            relative_path(new_path)
        except ValueError as exc:
            raise ActionError(str(exc)) from exc
        if old_path != new_path or old_path not in editable or old_path in seen:
            raise ActionError("Edit each allowed existing source file at most once; no renames")
        seen.add(old_path)
        original = files[old_path].splitlines(keepends=True)
        output: list[str] = []
        cursor = 0
        hunks = 0
        while index < len(lines) and not lines[index].startswith("--- "):
            match = HUNK.fullmatch(lines[index])
            if not match:
                raise ActionError("Malformed hunk header or unsupported patch directive")
            if any(value is not None and len(value) > 7 for value in match.groups()):
                raise ActionError("Hunk positions exceed supported bounds")
            old_start, old_count, new_start, new_count = (
                int(value) if value is not None else 1 for value in match.groups()
            )
            old_offset = old_start - 1 if old_count else old_start
            new_offset = new_start - 1 if new_count else new_start
            if old_offset < cursor or old_offset > len(original):
                raise ActionError("Hunks overlap or extend beyond the file")
            output.extend(original[cursor:old_offset])
            if new_offset != len(output):
                raise ActionError("New hunk position does not match the exact result")
            cursor = old_offset
            index += 1
            consumed = produced = 0
            while consumed < old_count or produced < new_count:
                if index >= len(lines) or not lines[index].startswith((" ", "+", "-")):
                    raise ActionError("Hunk line counts do not match")
                prefix, text = lines[index][0], lines[index][1:]
                if not text.endswith("\n"):
                    raise ActionError("Only newline-terminated text patches are supported")
                if prefix in {" ", "-"}:
                    if cursor >= len(original) or original[cursor] != text:
                        raise ActionError("Patch context does not exactly match")
                    cursor += 1
                    consumed += 1
                if prefix in {" ", "+"}:
                    output.append(text)
                    produced += 1
                if consumed > old_count or produced > new_count:
                    raise ActionError("Hunk line counts exceeded")
                index += 1
            hunks += 1
        if not hunks:
            raise ActionError("A file patch must contain a hunk")
        output.extend(original[cursor:])
        result[old_path] = "".join(output)
    return result
