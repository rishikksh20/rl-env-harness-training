"""Bounded JSON records for trusted multiprocessing workers."""

import json
from multiprocessing.connection import Connection

MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def send_record(connection: Connection, value: dict) -> None:
    data = json.dumps(value, allow_nan=False).encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Inference message exceeds byte limit")
    connection.send_bytes(data)


def receive_record(connection: Connection) -> dict:
    value = json.loads(connection.recv_bytes(MAX_MESSAGE_BYTES))
    if type(value) is not dict:
        raise ValueError("Expected inference record")
    return value
