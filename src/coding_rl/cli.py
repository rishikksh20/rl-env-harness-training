"""Scripted lessons, CPU inference/evaluation, and supervised native training."""

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from coding_rl.config import load_config, validate_model_mode
from coding_rl.environment import make_environment
from coding_rl.errors import InfrastructureError
from coding_rl.harness import run_episode


def _print_event(event: dict) -> None:
    name, payload = event["event"], event["payload"]
    if name == "reset":
        detail = f"task={event['task_id']} episode={event['episode_id'][:8]} starter={payload['starter_hash'][:12]}"
    elif name == "action":
        detail = payload["tool"]
    elif name == "observation":
        result = payload["result"]
        detail = result.get("content") or result.get("stdout") or json.dumps(result)
    elif name == "verify":
        detail = (
            f"private checks: {payload['passed']}/{payload['total']} status={payload['status']}"
        )
    elif name == "reward":
        detail = str(payload["value"])
    else:
        detail = json.dumps(payload)
    print(f"[{name}] {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Coding-agent environment, CPU evaluation, and native TRL training"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("env-demo", "env-inspect", "smoke", "rollout", "eval", "model-probe", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--sandbox", choices=["docker", "trusted_local"])
        if name not in {"eval", "model-probe", "train"}:
            command.add_argument("--task", default="zero-timeout")
        if name == "train":
            command.add_argument(
                "--probe",
                action="store_true",
                help="Probe native API/tokenizer without model weights or updates",
            )
        if name == "eval":
            command.add_argument(
                "--baseline", type=Path, help="Require a matching before/after evaluation profile"
            )
    args = parser.parse_args()
    environment = None
    backend = None
    try:
        config = load_config(args.config)
        logging.basicConfig(level=config.logging.level, format="%(levelname)s %(message)s")
        if args.sandbox:
            config = replace(config, sandbox=replace(config.sandbox, kind=args.sandbox))
        if args.command == "train":
            from coding_rl.training_worker import train

            result = train(config, probe_only=args.probe)
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "status",
                            "directory",
                            "error",
                            "parent_error",
                            "peak_process_tree_rss_bytes",
                        )
                    },
                    indent=2,
                )
            )
            return (
                130
                if result["status"] == "cancelled"
                else (0 if result["status"] in {"verified_update", "probe_only"} else 2)
            )
        if args.command == "model-probe":
            from coding_rl.probe import model_probe

            result = model_probe(config)
            print(json.dumps(result, indent=2))
            return 0 if result["read_gate_passed"] else 1
        if args.command in {"rollout", "eval"}:
            validate_model_mode(config)
            from coding_rl.models import TransformersBackend, dependency_versions

            dependency_versions()
            environment = make_environment(config)
            if args.command == "rollout":
                environment.catalog.get(args.task)
            environment.sandbox.preflight()
            backend = TransformersBackend(config.model)
            if args.command == "rollout":
                from coding_rl.rollout import run_model_episode

                outcome = run_model_episode(environment, backend, args.task)
                print(json.dumps(asdict(outcome), indent=2))
                return (
                    130 if outcome.finish_reason == "cancelled" else 0 if outcome.completed else 2
                )
            from coding_rl.evaluate import evaluate

            baseline = json.loads(args.baseline.read_text()) if args.baseline else None
            result = evaluate(environment, backend, baseline=baseline)
            print(json.dumps(result["metrics"], indent=2))
            print(f"[report] {config.evaluation.report_path}")
            if result["metrics"]["finish_reasons"].get("cancelled"):
                return 130
            return 2 if result["metrics"]["incomplete"] else 0
        if config.sandbox.kind == "trusted_local":
            print("[backend] trusted_local: authored fixtures only; no untrusted-code isolation")
        environment = make_environment(
            config, observer=_print_event if args.command == "env-demo" else None
        )
        if args.command == "env-inspect":
            environment.reset(args.task)
            print(json.dumps(environment.inspect(), indent=2))
        elif args.command == "env-demo":
            first = run_episode(environment, args.task)
            second = run_episode(environment, args.task)
            negative = run_episode(environment, args.task, variant="baseline")
            if not (
                first.reward == second.reward == 1.0
                and negative.reward == 0.0
                and first.starter_hash == second.starter_hash
                and first.episode_id != second.episode_id
            ):
                raise RuntimeError("Demo reset/reward assertions failed")
            print(
                f"[isolation] two clean independent repairs verified; negative reward={negative.reward}"
            )
            print(f"[traces] {environment.recorder.root}")
        else:
            summaries = []
            for task in environment.catalog.tasks():
                for variant, expected in (("baseline", 0.0), ("correct", 1.0)):
                    summary = run_episode(environment, task.task_id, variant=variant)
                    if summary.reward != expected:
                        raise RuntimeError(f"Unexpected scripted reward: {task.task_id}/{variant}")
                    summaries.append(asdict(summary))
            partial = run_episode(environment, "range-boundary", variant="visible_only")
            if partial.reward != 0.0:
                raise RuntimeError("Visible-only repair incorrectly passed independent grading")
            summaries.append(asdict(partial))
            path = environment.recorder.root / "smoke.json"
            path.write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"[smoke] {len(summaries)} scripted cases passed; metrics={path}")
        return 0
    except KeyboardInterrupt:
        print("[cancelled] episode interrupted; no reward assigned", file=sys.stderr)
        return 130
    except InfrastructureError as exc:
        print(
            f"[infrastructure_error] {exc}; correctness unknown, no reward assigned",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            try:
                if backend:
                    backend.close()
            finally:
                if environment:
                    try:
                        if environment.phase == "active":
                            environment.finish("cancelled")
                    finally:
                        environment.cleanup()
        except (InfrastructureError, OSError) as exc:
            print(f"[infrastructure_error] Resource release incomplete: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
