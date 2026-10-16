"""Convenient public environment API; optional runtime packages load on demand."""

from importlib import import_module

_EXPORTS = {
    "Config": ("config", "Config"),
    "load_config": ("config", "load_config"),
    "CodingEnvironment": ("environments.coding", "CodingEnvironment"),
    "make_environment": ("environments.coding", "make_environment"),
    "TaskCatalog": ("tasks.catalog", "TaskCatalog"),
    "run_episode": ("harnesses.scripted", "run_episode"),
    "run_model_episode": ("coding_rl.rollout.episode", "run_model_episode"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
