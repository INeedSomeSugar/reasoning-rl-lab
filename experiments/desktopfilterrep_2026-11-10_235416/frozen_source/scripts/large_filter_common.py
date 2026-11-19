"""Persistent single-GPU child execution with append-only costs and heartbeat evidence."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reasoning_rl.runtime import ROOT, json_write, jsonl_append

START = "checkpoints/desktoprepair_20261106_152712-S2"
START_HASH = "e8da01dae86ea2d7a95e19a645440510ec7dce2e4ceb546b26b3ff2dc9ac0bff"
TRAIN_SEEDS = [13, 29, 47, 71, 97]
PROBE_SEEDS = [20261111, 20261112, 20261113]
RANDOM_SEEDS = [3101, 3102, 3103]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    json_write(path, value)


def matrix_order():
    """Seven units per training seed; rotate U and reverse the F/R order by seed/block."""
    order = []
    for s_idx, seed in enumerate(TRAIN_SEEDS):
        units = []
        for j_idx, probe in enumerate(PROBE_SEEDS):
            arms = ("F", "R") if (s_idx+j_idx) % 2 == 0 else ("R", "F")
            units.extend({"key": f"{a}-j{probe}-s{seed}", "arm": a, "probe_seed": probe, "seed": seed} for a in arms)
        units.insert((s_idx*3) % 7, {"key": f"U-s{seed}", "arm": "U", "probe_seed": None, "seed": seed})
        order.extend(units)
    assert len(order) == 35 and len({r["key"] for r in order}) == 35
    return order


def host_snapshot():
    import psutil
    return {"utc": utc_now(), "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": psutil.disk_usage(str(ROOT)).free,
        "cpu_percent_instantaneous": psutil.cpu_percent(),
        "process_pid": os.getpid(), "boot_time_unix": psutil.boot_time()}


def run_phase(family, label, args, timeout, *, gpu=True, budget_start_unix=None):
    import psutil
    exp = ROOT / "experiments" / family
    log = ROOT / "logs" / f"{family}-{label}.log"
    if log.exists():
        raise FileExistsError(log)
    session_path = exp / "session_limit_frozen.json"
    if session_path.exists():
        timeout = min(timeout, read(session_path)["deadline_unix"]-time.time())
    if timeout <= 0:
        raise TimeoutError("Frozen whole-session deadline reached")
    lock_path = ROOT / "environment" / "large_filter_gpu.lock.json"
    lock_fd = None
    child = None
    started = time.time()
    command = [sys.executable, "-u", *args]
    before = host_snapshot()
    try:
        if gpu:
            # Do not steal a stale lock or terminate unrelated model tasks.
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, json.dumps({"family": family, "label": label, "runner_pid": os.getpid(), "started_unix": started}).encode())
            os.close(lock_fd)
            lock_fd = None
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if proc.pid == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
                    continue
                cmd = " ".join(proc.info["cmdline"] or [])
                if str(ROOT).lower() in cmd.lower() and any(x in cmd for x in ("reasoning_rl", "large_filter_tasks.py", "run_filter_probe.py", "filter_preflight.py")):
                    raise RuntimeError(f"Another project model task exists, pid={proc.pid}")
        jsonl_append(exp / "commands.jsonl", [{"label": label, "argv": command,
            "started_unix": started, "outer_budget_start_unix": budget_start_unix,
            "timeout_seconds": timeout, "host_before": before}])
        with log.open("x", encoding="utf-8") as out:
            child_env = {**os.environ, "RRL_ACTIVE_FAMILY": family}
            if budget_start_unix is not None:
                child_env["RRL_BUDGET_START_UNIX"] = str(budget_start_unix)
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env=child_env)
            jsonl_append(exp / "process_events.jsonl", [{"event": "started", "label": label, "pid": child.pid,
                "runner_pid": os.getpid(), "time_unix": time.time()}])
            timed_out = False
            while child.poll() is None:
                elapsed = time.time()-started
                json_write(exp / "heartbeat.json", {"family": family, "runner_pid": os.getpid(),
                    "child_pid": child.pid, "phase": label, "elapsed_seconds": elapsed,
                    "timeout_seconds": timeout, "heartbeat_utc": utc_now(), "host": host_snapshot()})
                if elapsed >= timeout:
                    timed_out = True
                    owned = psutil.Process(child.pid)
                    children = owned.children(recursive=True)
                    for p in reversed(children + [owned]):
                        try:
                            p.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    _, alive = psutil.wait_procs(children+[owned], timeout=5)
                    for p in alive:
                        p.kill()
                    break
                try:
                    child.wait(timeout=min(15, max(0.1, timeout-elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
        result = {"label": label, "argv": command, "pid": child.pid, "runner_pid": os.getpid(),
            "started_unix": started, "finished_unix": time.time(), "process_wall_seconds": time.time()-started,
            "outer_budget_start_unix": budget_start_unix,
            "exit_code": code, "timeout": timed_out, "timeout_seconds": timeout,
            "log": log.relative_to(ROOT).as_posix(), "host_before": before, "host_after": host_snapshot()}
        jsonl_append(exp / "process_costs.jsonl", [result])
        if code or timed_out:
            jsonl_append(exp / "failures.jsonl", [result])
        return result
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if gpu and lock_path.exists():
            lock = read(lock_path)
            if lock.get("runner_pid") == os.getpid() and lock.get("label") == label:
                lock_path.unlink()


def write_state(family, **state):
    json_write(ROOT / "experiments" / family / "runner_state.json", {"updated_utc": utc_now(), "runner_pid": os.getpid(), **state})
