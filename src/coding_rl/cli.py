"""Readable model-free lesson and assertions, using one validated configuration."""

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from coding_rl.config import load_config
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
        description="Stage A coding-agent environment lesson (no model)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("env-demo", "env-inspect", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--sandbox", choices=["docker", "trusted_local"])
        command.add_argument("--task", default="zero-timeout")
    args = parser.parse_args()
    environment = None
    try:
        config = load_config(args.config)
        logging.basicConfig(level=config.logging.level, format="%(levelname)s %(message)s")
        if args.sandbox:
            config = replace(config, sandbox=replace(config.sandbox, kind=args.sandbox))
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
        if environment:
            if environment.phase == "active":
                environment.finish("cancelled")
            environment.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
