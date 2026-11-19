"""Resume only pending evaluations after a verified user boundary pause.

The unchanged frozen controller sees the existing weight lock and skips training.
Every saved checkpoint and complete evaluation is verified before any new process.
"""
import argparse
import ctypes
import datetime as dt
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, jsonl_append, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, write_new, write_state, utc_now
from run_large_filter import controlled, verify_frozen


def checks(family):
    exp = ROOT / "experiments" / family
    pause_path = exp / "pause_after_evaluation_completed.json"
    paused = read(pause_path)
    assert paused["status"] == "paused_by_user_after_complete_evaluation"
    assert paused["no_next_model_started"] and paused["completed_training"] == 35
    assert read(exp / "runner_state.json")["status"] == "paused_by_user"
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    assert not (exp / "all_model_tasks_complete.json").exists()
    plan = read(exp / "protocol_frozen.json")
    assert time.time() < plan["session_limit"]["deadline_unix"]
    verify_frozen(family, plan)
    handoff = read(exp / "evaluation_pause_handoff_manifest_v2.json")
    for relative, digest in handoff["files"].items():
        assert sha256_file(ROOT / relative) == digest, relative
    lock_path = exp / "all_training_endpoints_fixed.json"
    lock = read(lock_path)
    opened = read(exp / "fresh_development_opened.json")
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    assert opened["endpoint_lock_sha256"] == sha256_file(lock_path)
    units = read(exp / "training_units.json")
    assert len(units) == 35 and all(u["status"] == "first_success_fixed" for u in units.values())
    for key, unit in units.items():
        assert checkpoint_hash(ROOT / unit["checkpoint"]) == unit["checkpoint_sha256"]
        assert lock["objects"][key]["checkpoint_sha256"] == unit["checkpoint_sha256"]
    evaluations = read(exp / "evaluation_units.json")
    completed = [key for key in plan["evaluation_order"] if evaluations[key]["status"] == "first_success_fixed"]
    pending = [key for key in plan["evaluation_order"] if evaluations[key]["status"] == "pending"]
    assert len(completed) == paused["completed_evaluation"] == 10
    assert len(completed)+len(pending) == 37 and len(pending) == 27
    assert pending == handoff["remaining_evaluations"] and pending[0] == "R-j20261113-s71"
    saved_hashes = {}
    durations = []
    trained_endpoint_durations = []
    for key in completed:
        item = evaluations[key]
        directory = ROOT / "runs" / item["run_name"]
        predictions = directory / "fresh_development/predictions.jsonl"
        assert sha256_file(predictions) == item["predictions_sha256"]
        assert item["n"] == plan["development_n"]
        assert sum(bool(line.strip()) for line in predictions.open(encoding="utf-8")) == plan["development_n"]
        metrics = read(directory / "fresh_development/metrics.json")
        assert metrics["n"] == plan["development_n"] and metrics["decoding"] == "greedy"
        assert read(directory / "evaluation_protocol.json")["adapter_sha256"] == lock["objects"][key]["checkpoint_sha256"]
        assert read(directory / "resolved_evaluation_ids.json")["ids"] == read(ROOT / plan["fresh_manifest_path"])["ids"]
        saved_hashes[key] = item["predictions_sha256"]
        durations.append(item["evaluation_process_seconds"])
        if key in units:
            trained_endpoint_durations.append(item["evaluation_process_seconds"])
    for key in pending:
        assert evaluations[key]["attempts"] == []
        name = family+"-eval-"+key+"-a0"
        assert not (ROOT / "runs" / name).exists() and not (ROOT / "logs" / (name+".log")).exists()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or [])
        if str(ROOT).lower() in command.lower() and any(marker in command for marker in (
                "reasoning_rl", "large_filter_tasks.py", "run_filter_probe.py", "filter_preflight.py",
                "run_large_filter.py", "resume_large_filter_from_pause.py", "pause_large_filter_after_evaluation.py")):
            raise RuntimeError(f"An existing project model/controller requires inspection: pid{proc.pid}")
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.cuda.get_device_name(0)
    assert "9070 XT" in device and torch.version.hip
    return plan, units, completed, pending, saved_hashes, {
        "torch": torch.__version__, "hip": torch.version.hip, "visible_device_count": torch.cuda.device_count(),
        "device": device, "platform": platform.platform(), "python": sys.executable,
        "completed_evaluation_process_seconds_mean": statistics.mean(durations),
        "completed_evaluation_process_seconds_min": min(durations),
        "completed_evaluation_process_seconds_max": max(durations),
        "completed_trained_endpoint_evaluation_seconds_mean": statistics.mean(trained_endpoint_durations),
        "remaining_model_hours_at_mean": len(pending)*statistics.mean(trained_endpoint_durations)/3600,
        "remaining_model_hours_observed_minmax_extrapolation": [len(pending)*min(trained_endpoint_durations)/3600, len(pending)*max(trained_endpoint_durations)/3600],
        "remaining_time_extrapolation_uses_trained_endpoints_excluding_A_S2": True,
        "timing_range_is_not_a_confidence_interval": True}


def main(family, check_only=False):
    exp = ROOT / "experiments" / family
    plan, units, completed, pending, saved_hashes, readiness = checks(family)
    if check_only:
        result = {"checked_utc": utc_now(), "status": "evaluation_resume_checks_passed",
            "training_permanently_closed": True, "saved_training_verified": 35,
            "complete_evaluations_verified": len(completed), "pending": len(pending), "next": pending[0],
            "readiness": readiness, "no_model_generation_or_training_run": True,
            "original_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"]}
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
        write_new(exp / ("evaluation_resume_precheck_"+stamp+".json"), result)
        print(result, flush=True)
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    directory = exp / ("evaluation_controller_resume_"+stamp)
    directory.mkdir(exist_ok=False)
    (directory / "resume_source.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    record = {"resumed_utc": utc_now(), "pid": os.getpid(), "user_authorization": "User explicitly instructed continue and requested remaining time",
        "resume_source_sha256": sha256_file(Path(__file__)), "archived_source_sha256": sha256_file(directory / "resume_source.py"),
        "prior_pause_receipt_sha256": sha256_file(exp / "pause_after_evaluation_completed.json"),
        "endpoint_lock_sha256": sha256_file(exp / "all_training_endpoints_fixed.json"),
        "protocol_sha256": sha256_file(exp / "protocol_frozen.json"),
        "all35_saved_training_hashes": {k: u["checkpoint_sha256"] for k, u in units.items()},
        "saved_complete_evaluation_hashes": saved_hashes, "pending_evaluation_order": pending,
        "readiness": readiness, "original_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"],
        "no_training_or_completed_evaluation_reruns": True, "partial_evaluation_resume": False,
        "frozen_runtime_data_models_verified_unchanged": True}
    write_new(directory / "resume_record.json", record)
    jsonl_append(exp / "process_events.jsonl", [{"event": "user_resumed_from_complete_evaluation_boundary",
        "label": "evaluation-controller-resume", "runner_pid": os.getpid(), "time_unix": time.time(),
        "resume_record": directory.relative_to(ROOT).as_posix(), "saved_training": 35, "saved_evaluations": len(completed)}])
    awake = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else 0
    try:
        controlled(family)
    except BaseException as error:
        write_state(family, stage="controlled_evaluation_resume", status="failed", error=repr(error))
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
