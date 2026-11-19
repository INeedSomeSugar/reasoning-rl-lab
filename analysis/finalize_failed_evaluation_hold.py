"""Preserve an actual failed child receipt and stop a diagnostic-held controller."""
import json
from pathlib import Path
import sys
import time
import psutil
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_append, sha256_file
from pause_large_filter_after_endpoint import open_handle, exit_info, K32
from large_filter_common import read, utc_now, host_snapshot

family = "desktopfilterrep_2026-11-10_235416"
exp = ROOT / "experiments" / family
parent = psutil.Process(34984)
assert "resume_large_filter_evaluation.py" in " ".join(parent.cmdline())
label = "eval-F-j20261112-s29-a0"
child_pid = 16816
command = [r for r in jsonl_read(exp / "commands.jsonl") if r["label"] == label]
assert len(command) == 1
command = command[0]
handle = open_handle(child_pid)
try:
    assert K32.WaitForSingleObject(handle, 0) == 0, "Current child still running; preserve it"
    info = exit_info(handle)
    assert info["exit_code"] != 0 and info["exit_code"] != 259
    directory = exp / ("runtime_incident_"+time.strftime("%Y-%m-%d_%H%M%S_UTC", time.gmtime()))
    directory.mkdir(exist_ok=False)
    for name in ("runner_state.json", "heartbeat.json", "evaluation_units.json", "commands.jsonl", "process_costs.jsonl"):
        (directory / ("before_"+name)).write_bytes((exp / name).read_bytes())
    json_write(directory / "actual_child_exit.json", info)
    receipt = {"label": label, "argv": command["argv"], "pid": child_pid, "runner_pid": parent.pid,
        "started_unix": command["started_unix"], "finished_unix": info["child_exit_unix"],
        "process_wall_seconds": info["child_exit_unix"]-command["started_unix"],
        "outer_budget_start_unix": command.get("outer_budget_start_unix"), "exit_code": info["exit_code"],
        "timeout": False, "timeout_seconds": command["timeout_seconds"],
        "log": "logs/"+family+"-"+label+".log", "host_before": command["host_before"], "host_after": host_snapshot(),
        "recorded_by_diagnostic_observer": True, "controller_suspension_not_added_to_child_process_seconds": True, **info}
    assert not [r for r in jsonl_read(exp / "process_costs.jsonl") if r["label"] == label]
    jsonl_append(exp / "process_costs.jsonl", [receipt])
    jsonl_append(exp / "failures.jsonl", [receipt])
    evaluations = read(exp / "evaluation_units.json")
    item = evaluations["F-j20261112-s29"]
    assert item["status"] == "running_attempt" and len(item["attempts"]) == 1
    attempt = item["attempts"][0]
    assert attempt["label"] == label
    attempt.update(receipt=receipt, status="failed", error="Independent evaluation failed with hipErrorInvalidImage; controller held to protect remaining attempts")
    item["status"] = "pending_retry"
    json_write(exp / "evaluation_units.json", evaluations)
    jsonl_append(exp / "attempt_validation_failures.jsonl", [{"key": "F-j20261112-s29", "attempt": 0,
        "stage": "evaluation", "error": attempt["error"], "diagnostic_observer": True}])
    parent.terminate()
    parent.wait(timeout=10)
    lock = ROOT / "environment/large_filter_gpu.lock.json"
    owned = read(lock)
    assert owned["family"] == family and owned["runner_pid"] == 34984 and owned["label"] == label
    target = directory / "gpu_lock_after_failed_child_and_controller_exit.json"
    assert ROOT.resolve() in target.resolve().parents and not target.exists()
    lock.rename(target)
    counts = {kind: sum(u["status"] == kind for u in evaluations.values()) for kind in (
        "first_success_fixed", "exhausted_failure", "pending_retry", "pending")}
    record = {"recorded_utc": utc_now(), "status": "diagnostic_hold_gpu_runtime_failure",
        "completed_training": 35, "completed_evaluation": counts["first_success_fixed"], "evaluation_status_counts": counts,
        "exhausted_evaluation_objects": [k for k,u in evaluations.items() if u["status"] == "exhausted_failure"],
        "current_failed_attempt": label, "current_attempt_actual_exit": info,
        "one_retry_still_available_for_current_object": True, "controller_terminated": 34984,
        "next_model_launch_stopped": True, "all_failures_and_costs_preserved": True,
        "frozen_models_data_config_scoring_unchanged": True, "training_permanently_closed": True,
        "source_sha256": sha256_file(Path(__file__)), "incident_directory": directory.relative_to(ROOT).as_posix()}
    json_write(directory / "diagnostic_hold.json", record)
    json_write(exp / "gpu_runtime_diagnostic_hold.json", record)
    json_write(exp / "runner_state.json", {"updated_utc": utc_now(), "runner_pid": None,
        "stage": "gpu_runtime_diagnostic_hold", "status": "diagnostic_hold", "successful_training": 35,
        "successful_evaluation": counts["first_success_fixed"], "error": "hipErrorInvalidImage after native evaluation abort; remaining attempts protected",
        "incident_directory": directory.relative_to(ROOT).as_posix()})
    print(json.dumps(record, ensure_ascii=False))
finally:
    K32.CloseHandle(handle)
