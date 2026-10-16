"""Explicit model compatibility diagnostic; never a graded task or baseline."""

import json
from dataclasses import asdict
from pathlib import Path

from config import Config
from environments.coding import make_environment
from harnesses.protocol import SYSTEM_PROMPT, parse_response
from models.backend import GenerationTimeout, TransformersBackend
from utils.seeding import derive_seed


def model_probe(config: Config) -> dict:
    if config.model is None:
        raise ValueError("model-probe requires a model configuration")
    backend = TransformersBackend(config.model)
    environment = None
    try:
        backend.start()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Use read_file to read timeout.py. Make exactly one tool call.",
            },
        ]
        ids, prefix = backend.render(messages)
        if len(ids) + config.rollout.max_new_tokens > config.rollout.context_tokens:
            raise ValueError("Probe prompt exceeds context budget")
        generation, parsed, observation, error = None, None, None, None
        try:
            generation = backend.generate(
                ids,
                max_new_tokens=config.rollout.max_new_tokens,
                seed=derive_seed(config.seed, "compatibility-probe", 0),
                timeout=config.rollout.generation_seconds,
                sampling=config.rollout,
            )
            if generation.stop_reason != "eos":
                raise ValueError("Probe generation truncated")
            parsed = parse_response(generation.raw_text, generation_prefix=prefix)
            if not parsed.call or parsed.call.name != "read_file":
                raise ValueError("Probe did not produce the requested read_file call")
            # Only a source read is allowed here; this diagnostic executes no code.
            from harnesses.protocol import dispatch

            environment = make_environment(config)
            environment.recorder.policy = backend.identity
            environment.reset("zero-timeout")
            observation = dispatch(environment, parsed.call)
        except (ValueError, GenerationTimeout) as exc:
            error = str(exc)
        report = {
            "kind": "compatibility_probe",
            "policy": backend.identity,
            "model_startups": backend.startups,
            "input_tokens": len(ids),
            "generation_prefix": prefix,
            "generation": asdict(generation) if generation else None,
            "generation_deadline_seconds": config.rollout.generation_seconds,
            "parsed": asdict(parsed) if parsed else None,
            "observation": observation,
            "error": error,
            "read_gate_passed": bool(observation and observation["ok"]),
            "limitation": "Diagnostic read only; no model repair, private grading, or held-out score",
        }
        path = Path(config.logging.trace_dir) / "model-probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return {**report, "report_path": str(path)}
    finally:
        backend.close()
        if environment:
            environment.cleanup()
