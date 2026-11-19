"""Project-local identity, process accounting and deadline for the U/F study."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reasoning_rl.runtime import ROOT, json_write, jsonl_append

FAMILY = "desktopfilter_20261110"
EXP = ROOT / "experiments" / FAMILY
START = "checkpoints/desktoprepair_20261106_152712-S2"
START_HASH = "e8da01dae86ea2d7a95e19a645440510ec7dce2e4ceb546b26b3ff2dc9ac0bff"
OLD = ROOT / "experiments/desktoprepair_20261106_152712"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def remaining():
    return read(EXP / "session.json")["deadline_unix"] - time.time()


def run_phase(label, args, timeout=None, env=None):
    """Only terminate the child tree owned by this specific invocation."""
    log = ROOT / "logs" / f"{FAMILY}-{label}.log"
    if log.exists():
        raise FileExistsError(log)
    allowed = remaining() - 300
    if timeout is not None:
        allowed = min(allowed, timeout)
    if allowed <= 0:
        raise TimeoutError("Four-hour study deadline reached")
    command = [sys.executable, "-u", *args]
    started = time.time()
    jsonl_append(EXP / "commands.jsonl", [{"phase": label, "argv": command,
        "started_unix": started, "timeout_seconds": allowed}])
    code, error = None, None
    with log.open("x", encoding="utf-8") as out:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})})
        try:
            code = child.wait(timeout=allowed)
        except subprocess.TimeoutExpired:
            import psutil
            process = psutil.Process(child.pid)
            owned = process.children(recursive=True)
            for p in reversed(owned + [process]):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(owned + [process], timeout=5)
            for p in alive:
                p.kill()
            code, error = -1, "Owned child exceeded recorded time limit"
    elapsed = time.time() - started
    receipt = {"phase": label, "process_wall_seconds": elapsed, "exit_code": code,
        "error": error, "started_unix": started, "finished_unix": time.time(),
        "log": str(log.relative_to(ROOT)), "pid": child.pid}
    jsonl_append(EXP / "process_costs.jsonl", [receipt])
    if code:
        jsonl_append(EXP / "failures.jsonl", [receipt])
        raise RuntimeError(f"{label} failed: {error or code}; inspect {log}")
    return elapsed


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()
