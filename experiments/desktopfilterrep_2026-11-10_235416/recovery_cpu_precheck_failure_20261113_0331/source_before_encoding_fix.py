"""Evaluation-only recovery with a failure circuit breaker around the frozen controller.

The overlay changes failure scheduling, not model arguments, scores, budgets or seeds.
It stops before another attempt can be appended after any new validation/process failure.
"""
import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, jsonl_append, jsonl_read, sha256_file

sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, write_new, write_state, utc_now
import run_large_filter as controller


class RecoveryHold(RuntimeError):
    pass


def host_gpu():
    command = "$r=Get-PnpDevice -Class Display | Where-Object {$_.FriendlyName -eq 'AMD Radeon RX 9070 XT'} | Select-Object FriendlyName,InstanceId,Status,Problem,Present; $d=Get-CimInstance Win32_VideoController | Where-Object {$_.Name -eq 'AMD Radeon RX 9070 XT'} | Select-Object Name,DriverVersion; [pscustomobject]@{pnp=$r;driver=$d} | ConvertTo-Json -Depth 4 -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RecoveryHold("Cannot verify registered GPU presence: "+result.stderr[-300:])
    value = json.loads(result.stdout)
    pnp = value.get("pnp")
    if not pnp or not pnp.get("Present") or pnp.get("Problem") != 0 or pnp.get("Status") != "OK":
        raise RecoveryHold("Registered9070XT is not present/healthy; no next model launch")
    return value


def failed_labels(evaluations):
    return {a["label"] for unit in evaluations.values() for a in unit.get("attempts", []) if a.get("status", "").startswith("failed")}


class Guard:
    def __init__(self, family, baseline_failures, driver, eligible_labels, verify, phase, gpu_query=host_gpu):
        self.family, self.exp = family, ROOT / "experiments" / family
        self.baseline = set(baseline_failures)
        self.driver = driver
        self.eligible_labels = set(eligible_labels)
        self.original_verify, self.original_phase, self.gpu_query = verify, phase, gpu_query
        self.failure = None
        self.last_gpu = None

    def verify(self, family, plan):
        if self.failure:
            raise RecoveryHold(self.failure)
        new_failures = failed_labels(read(self.exp / "evaluation_units.json"))-self.baseline
        if new_failures:
            raise RecoveryHold("New evaluation validation failure; stop before another attempt: "+",".join(sorted(new_failures)))
        self.last_gpu = self.gpu_query()
        if self.last_gpu["driver"]["DriverVersion"] != self.driver:
            raise RecoveryHold("Driver changed again during recovery segment")
        self.original_verify(family, plan)

    def phase(self, family, label, args, timeout, **kwargs):
        assert family == self.family and label in self.eligible_labels and label.startswith("eval-")
        assert args[:3] == ["-m", "reasoning_rl", "evaluate"] and "fresh_development" in args
        if self.failure:
            raise RecoveryHold(self.failure)
        receipt = self.original_phase(family, label, args, timeout, **kwargs)
        jsonl_append(self.exp / "evaluation_driver_segments.jsonl", [{"recorded_utc": utc_now(), "label": label,
            "segment": "after_user_driver_reinstall_2026-11-13", "driver": self.driver,
            "device_before_launch": self.last_gpu, "process_receipt_finished_unix": receipt["finished_unix"],
            "model_arguments_and_original_budget_not_changed": True}])
        if receipt["exit_code"] or receipt["timeout"]:
            self.failure = "New evaluation process failure; stop before next attempt: "+label
        return receipt


def selfcheck():
    # Check the scheduler boundary with fake receipts, not any model or evaluation.
    called = []
    guard = Guard("unused", set(), "unchanged", {"eval-unit-a0"}, lambda *_: called.append("verify"),
        lambda *_a, **_k: {"exit_code": 1, "timeout": False, "finished_unix": 1}, lambda: {})
    guard.failure = "synthetic process failure"
    try:
        guard.verify("unused", {})
    except RecoveryHold:
        pass
    else:
        raise AssertionError("Failure must fence the next pre-attempt verification")
    assert not called
    assert failed_labels({"x": {"attempts": [{"label": "old-a0", "status": "failed"}, {"label": "ok-a1", "status": "success"}]}}) == {"old-a0"}
    return {"new_failure_blocks_next_attempt_before_verification": True,
        "baseline_failures_can_be_distinguished_from_new_failures": True, "no_model_calls": True}


def checks(family):
    exp = ROOT / "experiments" / family
    assert read(exp / "runner_state.json")["status"] == "diagnostic_hold"
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    assert not (exp / "all_model_tasks_complete.json").exists()
    plan = read(exp / "protocol_frozen.json")
    assert time.time() < plan["session_limit"]["deadline_unix"]
    controller.verify_frozen(family, plan)
    lock_path = exp / "all_training_endpoints_fixed.json"
    lock = read(lock_path)
    assert read(exp / "fresh_development_opened.json")["endpoint_lock_sha256"] == sha256_file(lock_path)
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    training = read(exp / "training_units.json")
    assert len(training) == 35 and all(u["status"] == "first_success_fixed" for u in training.values())
    for key, obj in lock["objects"].items():
        if obj["checkpoint"]:
            assert checkpoint_hash(ROOT / obj["checkpoint"]) == obj["checkpoint_sha256"], key
    units = read(exp / "evaluation_units.json")
    completed = [k for k in plan["evaluation_order"] if units[k]["status"] == "first_success_fixed"]
    exhausted = [k for k in plan["evaluation_order"] if units[k]["status"] == "exhausted_failure"]
    pending = [k for k in plan["evaluation_order"] if units[k]["status"] in ("pending", "pending_retry")]
    audit = read(exp / "gpu_hold_audit_2026-11-13/summary.json")
    assert len(completed) == 22 and len(exhausted) == 5 and len(pending) == 10
    assert exhausted == audit["exhausted_evaluation_objects"] and pending == audit["eligible_remaining_evaluation_order"]
    ids = read(ROOT / plan["fresh_manifest_path"])["ids"]
    saved = {}
    durations = []
    for key in completed:
        unit = units[key]
        directory = ROOT / "runs" / unit["run_name"]
        path = directory / "fresh_development/predictions.jsonl"
        assert sha256_file(path) == unit["predictions_sha256"]
        assert [r["id"] for r in jsonl_read(path)] == ids
        assert read(directory / "evaluation_protocol.json")["adapter_sha256"] == lock["objects"][key]["checkpoint_sha256"]
        saved[key] = unit["predictions_sha256"]
        if key in training:
            durations.append(unit["evaluation_process_seconds"])
    labels = []
    for key in pending:
        assert len(units[key]["attempts"]) < 2
        for attempt in range(len(units[key]["attempts"]), 2):
            label = f"eval-{key}-a{attempt}"
            labels.append(label)
            assert not (ROOT / "runs" / (family+"-"+label)).exists()
            assert not (ROOT / "logs" / (family+"-"+label+".log")).exists()
    readiness_paths = sorted(exp.glob("driver_recovery_readiness_*/completed.json"))
    assert readiness_paths
    readiness_path = readiness_paths[-1]
    readiness = read(readiness_path)
    assert readiness["status"] == "passed" and readiness["old64_parity"]["n"] == 64
    assert readiness["fresh_development_test_SVAMP_not_accessed"]
    assert dt.datetime.fromisoformat(readiness["completed_utc"]).timestamp() > psutil.boot_time()
    gpu = host_gpu()
    assert gpu["driver"]["DriverVersion"] == "32.0.32015.16001"
    ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid in ancestors or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or [])
        if str(ROOT).lower() in command.lower() and any(m in command for m in ("reasoning_rl", "large_filter_tasks.py", "run_large_filter.py", "resume_large_filter", "check_large_filter_driver_recovery.py")):
            raise RuntimeError(f"Existing project model/controller requires inspection: pid{proc.pid}")
    return {"plan": plan, "training": training, "units": units, "completed": completed, "exhausted": exhausted,
        "pending": pending, "eligible_labels": labels, "saved": saved, "GPU": gpu,
        "readiness_path": readiness_path.relative_to(ROOT).as_posix(), "readiness_sha256": sha256_file(readiness_path),
        "process_hours_at_observed_mean": len(pending)*statistics.mean(durations)/3600}


def main(family, check_only=False):
    test = selfcheck()
    ctx = checks(family)
    exp = ROOT / "experiments" / family
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    public = {"recorded_utc": utc_now(), "status": "recovered_driver_evaluation_precheck_passed",
        "training_fixed": 35, "completed_evaluations_verified": 22, "exhausted_preserved": ctx["exhausted"],
        "remaining_evaluation_order": ctx["pending"], "readiness_path": ctx["readiness_path"],
        "readiness_sha256": ctx["readiness_sha256"], "GPU": ctx["GPU"], "selfcheck": test,
        "source_sha256": sha256_file(Path(__file__)), "original_deadline_unix": ctx["plan"]["session_limit"]["deadline_unix"],
        "remaining_process_hours_at_observed_mean": ctx["process_hours_at_observed_mean"],
        "driver_changed_from": "32.0.32015.12012", "driver_change_is_environment_recovery_stratum": True,
        "no_training_completed_evaluation_rerun_or_third_attempt": True,
        "failure_circuit_breaker_is_operational_overlay_not_numerical_change": True}
    if check_only:
        write_new(exp / ("gpu_recovery_precheck_"+stamp+".json"), public)
        print(json.dumps(public, ensure_ascii=False), flush=True)
        return
    directory = exp / ("gpu_recovery_evaluation_controller_"+stamp)
    directory.mkdir(exist_ok=False)
    (directory / "source.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    write_new(directory / "resume_record.json", {**public, "runner_pid": os.getpid(),
        "user_authorization": "Prior continue-to-completion remains active; user reports cleaned/reinstalled driver",
        "saved_training_hashes": {k: u["checkpoint_sha256"] for k,u in ctx["training"].items()},
        "saved_completed_evaluation_hashes": ctx["saved"],
        "endpoint_lock_sha256": sha256_file(exp / "all_training_endpoints_fixed.json"),
        "protocol_sha256": sha256_file(exp / "protocol_frozen.json")})
    for name in ("runner_state.json", "heartbeat.json", "evaluation_units.json"):
        (directory / ("before_"+name)).write_bytes((exp / name).read_bytes())
    original_verify, original_phase = controller.verify_frozen, controller.run_phase
    guard = Guard(family, failed_labels(ctx["units"]), ctx["GPU"]["driver"]["DriverVersion"], ctx["eligible_labels"], original_verify, original_phase)
    controller.verify_frozen, controller.run_phase = guard.verify, guard.phase
    jsonl_append(exp / "process_events.jsonl", [{"event": "resumed_after_user_driver_reinstall",
        "runner_pid": os.getpid(), "time_unix": time.time(), "driver": ctx["GPU"]["driver"]["DriverVersion"],
        "record": directory.relative_to(ROOT).as_posix(), "new_failure_stops_further_attempts": True}])
    awake = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else 0
    try:
        controller.controlled(family)
        write_new(directory / "completed.json", {"completed_utc": utc_now(), "all_remaining_eligible_units_terminal": True,
            "missing_exhausted_objects_preserved": ctx["exhausted"], "no_claim_all37_successful": True})
    except BaseException as error:
        write_state(family, stage="GPU_recovery_evaluation_hold", status="diagnostic_hold", error=repr(error),
            no_automatic_next_attempt=True, recovery_record=directory.relative_to(ROOT).as_posix())
        json_write(directory / "stopped.json", {"recorded_utc": utc_now(), "error": repr(error), "traceback": traceback.format_exc(),
            "new_attempts_not_automatically_retried": True, "frozen_runtime_unchanged": True})
        raise
    finally:
        controller.verify_frozen, controller.run_phase = original_verify, original_phase
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--selfcheck-only", action="store_true")
    args = parser.parse_args()
    if args.selfcheck_only:
        print(json.dumps(selfcheck()))
    else:
        main(args.family, args.check_only)
