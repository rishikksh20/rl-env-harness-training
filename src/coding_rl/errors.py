"""Separate bad actions from failures of trusted execution infrastructure."""


class ActionError(ValueError):
    """An invalid action or source snapshot; safe to report to the agent."""


class InfrastructureError(RuntimeError):
    """Correctness is unknown: never translate this into reward zero."""
