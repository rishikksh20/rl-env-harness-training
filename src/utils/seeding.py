"""Stable sampling seeds shared by all episode drivers."""

import hashlib
import json


def derive_seed(base: int, task_id: str, attempt: int, turn: int = 0) -> int:
    data = json.dumps([base, task_id, attempt, turn]).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")
