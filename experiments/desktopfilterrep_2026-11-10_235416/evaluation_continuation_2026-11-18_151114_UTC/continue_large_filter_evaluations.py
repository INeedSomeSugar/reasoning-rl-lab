"""Continue the frozen evaluation order from the November 13 saved boundary."""
import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, jsonl_append, jsonl_read, sha256_file

from resume_large_filter_after_gpu_recovery import Guard, failed_labels, host_gpu, selfcheck
from large_filter_common import read, run_phase, utc_now, write_new
import run_large_filter as controller


class UserPause(Exception):
    pass


def checked_state(exp, family, previous):
    state = read(exp / "runner_state.json")
    assert state["status"] in {"paused_by_user", "diagnostic_hold"}
    assert not (exp / "all_model_tasks_complete.json").exists()
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    plan = read(exp / "protocol_frozen.json")
    assert time.time() < plan["session_limit"]["deadline_unix"]
    controller.verify_frozen(family, plan)
    lock_path = exp / "all_training_endpoints_fixed.json"
    lock = read(lock_path)
    opened = read(exp / "fresh_development_opened.json")
    assert opened["endpoint_lock_sha256"] == sha256_file(lock_path) and opened["training_permanently_closed_for_this_family"]
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    training = read(exp / "training_units.json")
    assert len(training) == 35 and all(row["status"] == "first_success_fixed" for row in training.values())
    for key, obj in lock["objects"].items():
        if obj["checkpoint"]:
            assert checkpoint_hash(ROOT / obj["checkpoint"]) == obj["checkpoint_sha256"], key
        if key in training:
            assert training[key]["checkpoint_sha256"] == obj["checkpoint_sha256"], key
    units = read(exp / "evaluation_units.json")
    assert set(units) == set(plan["evaluation_order"]) == set(lock["objects"])
    assert all(row["status"] in {"pending", "pending_retry", "exhausted_failure", "first_success_fixed"} for row in units.values())
    completed = [k for k in plan["evaluation_order"] if units[k]["status"] == "first_success_fixed"]
    exhausted = [k for k in plan["evaluation_order"] if units[k]["status"] == "exhausted_failure"]
    pending = [k for k in plan["evaluation_order"] if units[k]["status"] in {"pending", "pending_retry"}]
    assert set(previous["exhausted"]) <= set(exhausted)
    assert set(previous["saved_complete_prediction_hashes"]) <= set(completed)
    ids = read(ROOT / plan["fresh_manifest_path"])["ids"]
    saved = {}
    for key in completed:
        row = units[key]
        directory = ROOT / "runs" / row["run_name"]
        path = directory / "fresh_development/predictions.jsonl"
        digest = sha256_file(path)
        assert digest == row["predictions_sha256"] == previous["saved_complete_prediction_hashes"].get(key, digest)
        assert [r["id"] for r in jsonl_read(path)] == ids
        assert read(directory / "evaluation_protocol.json")["adapter_sha256"] == lock["objects"][key]["checkpoint_sha256"]
        assert read(directory / "fresh_development/metrics.json")["n"] == len(ids)
        saved[key] = digest
    labels = []
    for key, row in units.items():
        assert len(row["attempts"]) <= 2
        if key in exhausted:
            assert len(row["attempts"]) == 2 and all(a["status"].startswith("failed") for a in row["attempts"])
    for key in pending:
        count = len(units[key]["attempts"])
        assert count < 2 and all(a["status"].startswith("failed") for a in units[key]["attempts"])
        for attempt in range(count, 2):
            label = f"eval-{key}-a{attempt}"
            assert not (ROOT / "runs" / f"{family}-{label}").exists(), label
            assert not (ROOT / "logs" / f"{family}-{label}.log").exists(), label
            labels.append(label)
    return plan, units, saved, exhausted, pending, labels


def main(family, prior_receipt, user_message):
    exp = ROOT / "experiments" / family
    previous_path = ROOT / prior_receipt
    previous = read(previous_path)
    previous_host = read(exp / "driver_recovery_paused_host_verification.json")
    installed_driver = previous_host["display"]["DriverVersion"]
    assert selfcheck()["new_failure_blocks_next_attempt_before_verification"]
    plan, units, saved, exhausted, pending, labels = checked_state(exp, family, previous)
    if read(exp / "runner_state.json")["status"] == "paused_by_user":
        relative = (exp / "evaluation_units.json").relative_to(ROOT).as_posix()
        assert sha256_file(ROOT / relative) == previous["files"][relative]
        assert saved == previous["saved_complete_prediction_hashes"]
        assert exhausted == previous["exhausted"] and pending == previous["pending_order"]
    ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        if proc.pid in ancestors or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or [])
        local = str(ROOT).casefold() in command.casefold() or str(proc.info["cwd"]).casefold() == str(ROOT).casefold()
        if local and any(marker in command for marker in ("reasoning_rl", "large_filter_tasks", "run_large_filter", "resume_large_filter", "continue_large_filter_evaluations", "check_large_filter_driver_recovery")):
            raise RuntimeError(f"An existing project model/controller requires inspection: PID {proc.pid}")
    gpu = host_gpu()
    assert gpu["driver"]["DriverVersion"] == installed_driver
    prior_recovery = sorted(exp.glob("gpu_recovery_evaluation_controller_*/resume_record.json"))[-1]
    assert read(prior_recovery)["GPU"]["pnp"]["InstanceId"] == gpu["pnp"]["InstanceId"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    directory = exp / ("evaluation_continuation_"+stamp)
    directory.mkdir(exist_ok=False)
    for source in (Path(__file__), ROOT / "analysis/resume_large_filter_after_gpu_recovery.py",
            ROOT / "analysis/check_large_filter_driver_recovery.py"):
        (directory / source.name).write_bytes(source.read_bytes())
    for name in ("runner_state.json", "heartbeat.json", "evaluation_units.json"):
        (directory / ("before_"+name)).write_bytes((exp / name).read_bytes())
    pause_request = directory / "pause_requested.json"
    awake = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    original_verify, original_phase = controller.verify_frozen, controller.run_phase
    try:
        write_new(directory / "authorization.json", {"recorded_utc": utc_now(), "runner_pid": os.getpid(),
            "user_message": user_message, "previous_handoff_sha256": sha256_file(previous_path),
            "training_locked_and_closed": True, "original_deadline_unix": plan["session_limit"]["deadline_unix"]})
        health_label = "continuation-tensor-"+stamp
        health = run_phase(family, health_label, ["analysis/check_large_filter_driver_recovery.py", "--tensor-child",
            "--record", (directory / "tensor.json").relative_to(ROOT).as_posix()], 180)
        if health["exit_code"] or health["timeout"]:
            raise RuntimeError("Same-device tensor readiness failed before any new evaluation")
        readiness = read(directory / "tensor.json")
        assert readiness["status"] == "tensor_BF16_SDPA_forward_backward_and_cast_passed"
        prior_readiness = read(ROOT / read(prior_recovery)["readiness_path"])["tensor"]
        for field in ("device", "visible_device_count", "torch", "hip"):
            assert readiness[field] == prior_readiness[field], field
        duration = statistics.mean(units[key]["evaluation_process_seconds"] for key in saved if key not in {"A", "S2"})
        record = {"recorded_utc": utc_now(), "runner_pid": os.getpid(), "GPU": gpu, "boot_time_unix": psutil.boot_time(),
            "readiness_process": health, "readiness": readiness, "complete_prediction_sha256": saved,
            "exhausted_objects": exhausted, "remaining_order": pending, "allowed_attempt_labels": labels,
            "remaining_expected_model_hours": len(pending)*duration/3600,
            "original_deadline_unix": plan["session_limit"]["deadline_unix"],
            "weights_lock_sha256": sha256_file(exp / "all_training_endpoints_fixed.json"),
            "protocol_sha256": sha256_file(exp / "protocol_frozen.json"),
            "source_sha256": sha256_file(Path(__file__)), "prior_handoff_sha256": sha256_file(previous_path),
            "no_additional_training": True, "one_new_failure_stops_further_launches": True}
        write_new(directory / "resume_record.json", record)
        jsonl_append(exp / "process_events.jsonl", [{"event": "explicit_user_continue_after_driver_recovery_pause",
            "runner_pid": os.getpid(), "time_unix": time.time(), "record": directory.relative_to(ROOT).as_posix(),
            "driver": installed_driver, "saved_evaluations": len(saved), "eligible_evaluations": len(pending)}])
        print(json.dumps({"state": "ready", "saved": len(saved), "exhausted": len(exhausted), "pending": len(pending),
            "expected_hours": record["remaining_expected_model_hours"], "next": pending[0] if pending else None,
            "receipt": str(directory / "resume_record.json")}, ensure_ascii=False), flush=True)
        guard = Guard(family, failed_labels(units), installed_driver, labels, original_verify, original_phase)

        def verify_before_scheduling(name, frozen):
            if pause_request.exists():
                raise UserPause("User requested pause at the next saved evaluation boundary")
            guard.verify(name, frozen)

        controller.verify_frozen, controller.run_phase = verify_before_scheduling, guard.phase
        controller.controlled(family)
        write_new(directory / "completed.json", {"recorded_utc": utc_now(), "state": "eligible_model_work_terminal",
            "model_completion": read(exp / "all_model_tasks_complete.json"), "initial_exhausted_objects": exhausted})
    except BaseException as error:
        current = read(exp / "evaluation_units.json")
        paused = isinstance(error, UserPause)
        status = "paused_by_user" if paused else "diagnostic_hold"
        snapshot = {"updated_utc": utc_now(), "runner_pid": None, "stage": "evaluation_continuation", "status": status,
            "successful_training": 35, "successful_evaluation": sum(u["status"] == "first_success_fixed" for u in current.values()),
            "error": str(error), "continuation": directory.relative_to(ROOT).as_posix()}
        write_new(directory / ("paused.json" if paused else "stopped.json"), {**snapshot,
            "traceback": traceback.format_exc(), "results_and_retry_counts_preserved": True})
        json_write(exp / "runner_state.json", snapshot)
        if not paused:
            raise
    finally:
        controller.verify_frozen, controller.run_phase = original_verify, original_phase
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--prior-receipt", required=True)
    parser.add_argument("--user-message", default="继续")
    args = parser.parse_args()
    main(args.family, args.prior_receipt, args.user_message)
