from dataclasses import replace

import pytest

from coding_rl import Config, make_environment
from coding_rl.config import LoggingConfig


@pytest.fixture
def config(tmp_path):
    base = Config(logging=LoggingConfig(str(tmp_path / "traces")))
    return replace(base, sandbox=replace(base.sandbox, kind="trusted_local"))


@pytest.fixture
def env(config):
    instance = make_environment(config)
    try:
        yield instance
    finally:
        instance.cleanup()
