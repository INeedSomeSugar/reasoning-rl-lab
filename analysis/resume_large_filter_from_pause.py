"""Resume only between saved endpoints, using the unchanged frozen controller."""
import argparse
import ctypes
import datetime as dt
import os
from pathlib import Path
import sys
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, jsonl_append, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, write_new, write_state, utc_now
from run_large_filter import controlled, verify_frozen


def main(family, check_only=False):
    exp = ROOT / "experiments" / family
    pause_path = exp / "pause_after_endpoint_completed.json"
    paused = read(pause_path)
    assert paused["status"] == "paused_by_user_after_terminal_checkpoint"
    assert paused["current_endpoint_status"] == "first_success_fixed"
    assert not (exp / "fresh_development_opened.json").exists()
    assert not (exp / "all_training_endpoints_fixed.json").exists()
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    assert time.time() < plan["session_limit"]["deadline_unix"]
    units = read(exp / "training_units.json")
    completed = {k: u for k, u in units.items() if u["status"] == "first_success_fixed"}
    pending = [u["key"] for u in plan["order"] if units[u["key"]]["status"] == "pending"]
    assert len(completed) == 21 and len(pending) == 14
    assert pending[0] == "R-j20261111-s71"
    for key, item in completed.items():
        assert checkpoint_hash(ROOT / item["checkpoint"]) == item["checkpoint_sha256"]
    for key in pending:
        assert units[key]["attempts"] == []
        assert not (ROOT / "runs" / (family+"-"+key+"-a0")).exists()
        assert not (ROOT / "logs" / (family+"-"+key+"-a0.log")).exists()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or [])
        if str(ROOT).lower() in command.lower() and any(s in command for s in (
                "reasoning_rl", "large_filter_tasks.py", "run_filter_probe.py", "filter_preflight.py")):
            raise RuntimeError(f"Another project model task exists: pid{proc.pid}")
    if check_only:
        print({"status": "resume_checks_passed_without_model_or_state_mutation", "saved_endpoints_verified": 21,
            "pending": 14, "next": pending[0], "original_deadline_unchanged": plan["session_limit"]["deadline_unix"]})
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    directory = exp / ("controller_resume_"+stamp)
    directory.mkdir(exist_ok=False)
    write_new(directory / "resume_record.json", {"resumed_utc": utc_now(), "pid": os.getpid(),
        "user_authorization": "Latest explicit user instruction: continue",
        "resume_source_sha256": sha256_file(Path(__file__)), "prior_pause_receipt_sha256": sha256_file(pause_path),
        "protocol_sha256": sha256_file(exp / "protocol_frozen.json"),
        "saved_endpoints": {k: u["checkpoint_sha256"] for k, u in completed.items()},
        "pending_order": pending, "original_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"],
        "frozen_source_models_data_verified": True, "midunit_sampler_optimizer_resume": False,
        "completed_endpoints_and_probes_will_not_be_rerun": True})
    jsonl_append(exp / "process_events.jsonl", [{"event": "user_resumed_from_saved_endpoint_boundary",
        "label": "controller-resume", "runner_pid": os.getpid(), "time_unix": time.time(),
        "resume_record": directory.relative_to(ROOT).as_posix(), "saved_training_endpoints": 21}])
    awake = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else 0
    try:
        controlled(family)
    except BaseException as error:
        write_state(family, stage="controlled_resume", status="failed", error=repr(error))
        json_write(directory / "resume_error.json", {"recorded_utc": utc_now(), "error": repr(error),
            "traceback": traceback.format_exc(), "original_deadline_unchanged": True})
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    main(args.family, args.check_only)
