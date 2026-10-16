"""Contained JSON callable worker. This file is executed, never imported by the grader."""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    # Bind trusted serialization before loading the untrusted module. This is a
    # value protocol, not a claim that Python can prevent in-process spoofing.
    dumps = json.dumps
    request = json.loads(sys.stdin.buffer.read(65537))
    module = Path.cwd() / request["module"]
    with contextlib.redirect_stdout(sys.stderr):
        try:
            spec = importlib.util.spec_from_file_location("candidate", module)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load callable")
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
            value = getattr(candidate, request["function"])(**request["kwargs"])
            response = dumps({"status": "ok", "value": value}, allow_nan=False)
        except Exception:
            response = dumps({"status": "candidate_error"})
    print(response)


if __name__ == "__main__":
    main()
