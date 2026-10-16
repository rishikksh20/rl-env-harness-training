"""Parent-supervised native process and exact run-owned resource cleanup."""

import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from coding_rl.config import Config, validate_training_mode
from coding_rl.errors import InfrastructureError
from coding_rl.inference_worker import receive_record, send_record


def _worker_main(connection, config: Config, run_id: str, directory: str, probe_only: bool):
    try:
        os.setsid()
        from coding_rl.trainer import execute_training

        result = execute_training(config, run_id, Path(directory), probe_only=probe_only)
        # Large tensor/schema evidence is on disk; IPC carries only a small status.
        from coding_rl.trainer import write_record

        write_record(Path(directory) / "result.json", asdict(result))
        send_record(connection, {"status": result.status})
    except BaseException as exc:
        send_record(
            connection, {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"[:4000]}
        )
    finally:
        connection.close()


def cleanup_resources(directory: Path, run_id: str) -> None:
    """Only names registered before launch, and paths below this parent's run root."""
    private = directory / "private"
    journal = private / "containers.jsonl"
    names = []
    if journal.exists():
        for line in journal.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                # Registration flushes/fsyncs the newline before launch. An interrupted
                # partial registration cannot own a launched container.
                continue
            name = json.loads(line)
            prefix = f"coding-rl-{run_id}-"
            if (
                not isinstance(name, str)
                or not name.startswith(prefix)
                or len(name) != len(prefix) + 32
                or any(c not in "0123456789abcdef" for c in name[len(prefix) :])
            ):
                raise InfrastructureError("Invalid run-owned container registry")
            names.append(name)
    for name in names:
        try:
            result = subprocess.run(
                ["docker", "rm", "--force", name],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError("Parent could not release an owned Docker worker") from exc
        if result.returncode and "No such container" not in result.stderr:
            raise InfrastructureError("Parent could not release an owned Docker worker")
    for path in (directory / "checkpoints").glob(".checkpoint-*"):
        shutil.rmtree(path)
    if private.exists():
        shutil.rmtree(private)


def _stop(process) -> None:
    if process.is_alive():
        # Collect descendants before losing the parent relationship.
        import psutil

        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in descendants:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    process.join(timeout=2)


def train(config: Config, *, probe_only: bool = False) -> dict:
    validate_training_mode(config)
    from coding_rl.trainer import dependency_report, write_record

    dependency_report(config.training.mode)
    import psutil

    run_id = uuid.uuid4().hex
    directory = Path(config.training.output_dir).resolve() / run_id
    directory.mkdir(parents=True, mode=0o700)
    (directory / "private").mkdir(mode=0o700)
    write_record(directory / "config.json", asdict(config))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_main,
        args=(child, config, run_id, str(directory), probe_only),
        name="coding-rl-native-training",
    )
    status, error = "incomplete", None
    started = time.monotonic()
    peak = 0
    try:
        process.start()
        child.close()
        while True:
            if parent.poll(0.1):
                response = receive_record(parent)
                status, error = response["status"], response.get("error")
                break
            if not process.is_alive():
                error = "Native training process exited without a completed result"
                break
            try:
                live = psutil.Process(process.pid)
                rss = sum(
                    p.memory_info().rss
                    for p in [live, *live.children(recursive=True)]
                    if p.is_running()
                )
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                continue
            if rss > config.training.memory_limit_mb * 1048576:
                error = "Native process exceeded training.memory_limit_mb"
                break
            if time.monotonic() - started > config.training.wall_seconds:
                error = "Native process exceeded training.wall_seconds"
                break
    except KeyboardInterrupt:
        status, error = "cancelled", "Native training interrupted"
        if process.pid and process.is_alive():
            os.kill(process.pid, signal.SIGINT)
            process.join(timeout=2)
    except (OSError, EOFError, ValueError) as exc:
        error = f"Native process communication failed: {exc}"
    finally:
        try:
            if process.pid:
                _stop(process)
            cleanup_resources(directory, run_id)
        except (OSError, InfrastructureError) as exc:
            status, error = "incomplete", f"Parent cleanup failed: {exc}"
        finally:
            child.close()
            parent.close()
            process.close()
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if not result and (directory / "groups.json").exists():
        result["groups"] = json.loads((directory / "groups.json").read_text())
        for group in result["groups"]:
            if group["status"] in {"collecting", "admitted", "updated"}:
                group["status"] = "interrupted"
    from coding_rl.checkpoints import load_checkpoint

    committed = []
    for path in sorted((directory / "checkpoints").glob("step-*")):
        try:
            artifact = load_checkpoint(path)
        except ValueError:
            continue
        committed.append({"path": str(artifact.path), **artifact.identity})
        for group in result.get("groups", []):
            if group["collection_id"] == artifact.manifest["update_evidence"].get("collection_id"):
                group.update(status="committed", checkpoint=str(artifact.path))
    result["checkpoints"] = committed
    if error:
        status = "cancelled" if status == "cancelled" else "incomplete"
    result.update(
        {
            "status": status,
            "run_id": run_id,
            "directory": str(directory),
            "parent_error": error,
            "peak_process_tree_rss_bytes": peak,
            "seconds": time.monotonic() - started,
        }
    )
    write_record(directory / "run.json", result)
    return result
