"""One audited controller recovery; retains frozen runtime and performs no exact resume."""
import argparse
import ctypes
import datetime as dt
import os
from pathlib import Path
import sys
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_append, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, write_new, write_state, utc_now
from run_large_filter import controlled, reconcile_interrupted_attempt, verify_frozen


def main(family, old_runner, old_child):
    exp = ROOT / "experiments" / family
    if (exp / "fresh_development_opened.json").exists() or (exp / "all_training_endpoints_fixed.json").exists():
        raise RuntimeError("This recovery is only for the pre-evaluation U47 interruption")
    for pid in (old_runner, old_child):
        if psutil.pid_exists(pid):
            raise RuntimeError(f"Recorded old pid{pid} exists; investigate before recovering")
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    if time.time() >= plan["session_limit"]["deadline_unix"]:
        raise TimeoutError("Original frozen session deadline reached")
    heartbeat = read(exp / "heartbeat.json")
    assert heartbeat["runner_pid"] == old_runner and heartbeat["child_pid"] == old_child
    assert heartbeat["phase"] == "U-s47-a0"
    assert psutil.boot_time() > heartbeat["host"]["boot_time_unix"]
    units_path = exp / "training_units.json"
    units = read(units_path)
    assert sum(u["status"] == "first_success_fixed" for u in units.values()) == 20
    interrupted = units["U-s47"]
    assert interrupted["status"] == "running_attempt" and len(interrupted["attempts"]) == 1
    assert interrupted["attempts"][0]["label"] == "U-s47-a0"
    assert not (ROOT / "runs" / (family+"-U-s47-a1")).exists()
    assert not (ROOT / "logs" / (family+"-U-s47-a1.log")).exists()
    lock_path = ROOT / "environment/large_filter_gpu.lock.json"
    stale = read(lock_path)
    assert stale["family"] == family and stale["runner_pid"] == old_runner and stale["label"] == "U-s47-a0"
    # Check every owned start event too. Do not terminate anything or steal a live lock.
    for event in jsonl_read(exp / "process_events.jsonl"):
        if event.get("label") == "U-s47-a0" and event.get("event") == "started":
            assert event["pid"] == old_child and not psutil.pid_exists(event["pid"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    recovery = exp / ("controller_recovery_"+stamp)
    recovery.mkdir(exist_ok=False)
    write_new(recovery / "before_recovery.json", {"family": family, "recorded_utc": utc_now(),
        "old_runner_pid": old_runner, "old_child_pid": old_child, "new_runner_pid": os.getpid(),
        "old_heartbeat": heartbeat, "units_before_sha256": sha256_file(units_path),
        "old_pid_receipt_preserved": "runner_controlled_pid.json", "boot_time_unix": psutil.boot_time(),
        "frozen_protocol_sha256": sha256_file(exp / "protocol_frozen.json"),
        "frozen_source_verified": True, "all20_successful_endpoints_preserved": True,
        "same_S2_config_seed_complete_retry_only": True, "exact_sampler_optimizer_resume": False,
        "original_session_deadline_unchanged": plan["session_limit"]["deadline_unix"]})
    archive_path = recovery / "stale_gpu_lock.json"
    assert ROOT.resolve() in lock_path.resolve().parents and ROOT.resolve() in archive_path.resolve().parents
    assert not archive_path.exists()
    lock_path.rename(archive_path)
    reconcile_interrupted_attempt(family, interrupted, "training")
    assert interrupted["attempts"][0]["status"] == "failed_interrupted"
    json_write(units_path, units)
    write_new(recovery / "runner_pid.json", {"pid": os.getpid(), "started_utc": utc_now(),
        "recovery_source_sha256": sha256_file(Path(__file__)), "status": "controlled_from_same_frozen_plan"})
    jsonl_append(exp / "process_events.jsonl", [{"event": "controller_recovery_after_confirmed_system_restart",
        "runner_pid": os.getpid(), "old_runner_pid": old_runner, "old_child_pid": old_child,
        "time_unix": time.time(), "recovery_receipt": recovery.relative_to(ROOT).as_posix(),
        "interrupted_attempt": "U-s47-a0", "only_remaining_training_attempt": "U-s47-a1"}])
    awake = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else 0
    try:
        controlled(family)
    except BaseException as error:
        write_state(family, stage="controlled_recovery", status="failed", error=repr(error))
        json_write(recovery / "recovery_failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "recorded_utc": utc_now(), "original_deadline_unchanged": True})
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--old-runner", type=int, required=True)
    parser.add_argument("--old-child", type=int, required=True)
    args = parser.parse_args()
    main(args.family, args.old_runner, args.old_child)
