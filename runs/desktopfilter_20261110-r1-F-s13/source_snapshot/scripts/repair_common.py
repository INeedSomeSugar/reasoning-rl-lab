"""Project-local process accounting and a six-hour hard stop for the repair study."""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from reasoning_rl.runtime import ROOT, json_write, jsonl_append

FAMILY = "desktoprepair_20261106_152712"
EXP = ROOT / "experiments" / FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deadline():
    return dt.datetime.fromisoformat(read(EXP / "session.json")["deadline_utc"]).timestamp()


def remaining():
    return deadline()-time.time()


def run_phase(label, args, state=None, status_path=None, reserve=900, timeout=None):
    """The child is this project's own command; kill its owned tree only at deadline."""
    log = ROOT / "logs" / f"{FAMILY}-{label}.log"
    if log.exists():
        raise FileExistsError(f"Preserve existing log {log}")
    allowed = remaining()-reserve
    if allowed <= 0:
        raise TimeoutError("Six-hour session cleanup boundary reached")
    if timeout is not None:
        allowed = min(allowed, timeout)
    command = [sys.executable, "-u", *args]
    if state is not None:
        state.update(phase=label)
        json_write(status_path, state)
    started = time.time()
    jsonl_append(EXP / "commands.jsonl", [{"phase": label, "argv": command, "started_unix": started}])
    status, error = None, None
    with log.open("x", encoding="utf-8") as out:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        try:
            status = child.wait(timeout=allowed)
        except subprocess.TimeoutExpired as exc:
            import psutil
            # All descendants are tied to the PID launched immediately above.
            process = psutil.Process(child.pid)
            descendants = process.children(recursive=True)
            for owned in reversed(descendants):
                try:
                    owned.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            psutil.wait_procs(descendants+[process], timeout=5)
            error = f"Deadline/phase timeout: {allowed:.1f} seconds"
            status = -1
    elapsed = time.time()-started
    jsonl_append(EXP / "process_costs.jsonl", [{"phase": label, "process_wall_seconds": elapsed,
        "exit_code": status, "error": error, "started_unix": started, "finished_unix": time.time()}])
    if status:
        jsonl_append(EXP / "failures.jsonl", [{"phase": label, "time_unix": time.time(),
            "exit_code": status, "error": error or f"Process exit {status}", "log": str(log.relative_to(ROOT))}])
        raise RuntimeError(f"{label} failed: {error or status}; raw evidence preserved")
    if state is not None:
        state.setdefault("completed", []).append(label)
        json_write(status_path, state)
    return elapsed
