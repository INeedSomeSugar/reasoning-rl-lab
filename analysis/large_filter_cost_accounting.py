"""All-attempt physical ledger and bounded failure costs; never reads predictions."""
import argparse
import datetime as dt
import math
import sys
import time

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read


def accounting(exp, training_units, evaluations):
    processes = jsonl_read(exp / "process_costs.jsonl")
    exact = {r["label"]: r for r in processes}
    assert len(exact) == len(processes), "Duplicate process receipt labels would double-charge"
    commands = jsonl_read(exp / "commands.jsonl")
    starts = sorted(r["started_unix"] for r in commands)
    complete_path = exp / "all_model_tasks_complete.json"
    final_clock = dt.datetime.fromisoformat(read(complete_path)["completed_utc"]).timestamp() if complete_path.exists() else time.time()
    diagnostic_hold = exp / "gpu_runtime_diagnostic_hold.json"
    validated_stop = {}
    if diagnostic_hold.exists():
        hold = read(diagnostic_hold)
        if hold["next_model_launch_stopped"]:
            validated_stop[hold["current_failed_attempt"]] = dt.datetime.fromisoformat(hold["recorded_utc"]).timestamp()
    interrupted = []
    bounds_path = exp / "interruption_cost_bounds.jsonl"
    if bounds_path.exists():
        audit_path = exp / "interruption_cost_audit.json"
        if not audit_path.exists():
            raise RuntimeError("Audit interrupted-process bounds before claiming complete costs")
        audit = read(audit_path)
        assert audit["source_bounds_sha256"] == sha256_file(bounds_path), "Refresh interruption audit after any new interruption"
        interrupted = audit["interrupted_processes"]
        raw = jsonl_read(bounds_path)
        assert len({r["label"] for r in raw}) == len(raw)
        assert {r["label"] for r in interrupted} == {r["label"] for r in raw}
        assert not set(exact) & {r["label"] for r in interrupted}, "Interrupted costs must not duplicate an exact receipt"
        for row in interrupted:
            assert 0 <= row["process_seconds_lower_bound"] <= row["process_seconds_upper_bound"]
            assert 0 <= row["failed_endpoint_path_seconds_lower_bound"] <= row["failed_endpoint_path_seconds_upper_bound"]
    interrupted_by_label = {r["label"]: r for r in interrupted}
    failed_by_endpoint = {}
    failed_labels = set()
    for collection in (training_units, evaluations):
        for key, unit in collection.items():
            target = failed_by_endpoint.setdefault(key, {"failed_attempts": 0, "lower_seconds": 0.0,
                "upper_seconds": 0.0, "labels": [], "bounds_notes": []})
            for attempt in unit.get("attempts", []):
                if not attempt.get("status", "").startswith("failed"):
                    continue
                label = attempt["label"]
                assert label not in failed_labels
                failed_labels.add(label)
                if label in interrupted_by_label:
                    lost = interrupted_by_label[label]
                    lower = lost["failed_endpoint_path_seconds_lower_bound"]
                    upper = lost["failed_endpoint_path_seconds_upper_bound"]
                    note = "Known outer preparation plus live-child activity/next-OS-boot bounds; exact exit unknown"
                else:
                    if label not in exact:
                        raise RuntimeError(f"Failed attempt {label} has neither exact nor bounded cost evidence")
                    receipt = exact[label]
                    outer = receipt.get("outer_budget_start_unix") or receipt["started_unix"]
                    lower = receipt["finished_unix"]-outer
                    # Failed parent validation duration was not separately logged.
                    # Bound it by the next process start or controller completion;
                    # do not invent an exact zero-overhead failure duration.
                    next_start = next((s for s in starts if s > receipt["finished_unix"]), final_clock)
                    upper = max(lower, min(next_start, validated_stop.get(label, final_clock))-outer)
                    note = "Exact child/known outer start; unlogged parent-failure overhead bounded by next process start, verified stop, or controller completion"
                assert 0 <= lower <= upper
                target["failed_attempts"] += 1
                target["lower_seconds"] += lower
                target["upper_seconds"] += upper
                target["labels"].append(label)
                target["bounds_notes"].append(note)
    assert set(interrupted_by_label) <= failed_labels, "Interrupted attempt must remain explicitly failed in unit state"
    fields = ("label", "pid", "runner_pid", "outer_budget_start_unix", "started_unix", "finished_unix",
        "process_wall_seconds", "exit_code", "timeout", "timeout_seconds", "log")
    ledger = [{**{k: r.get(k) for k in fields}, "receipt_status": "exact_completed_process",
        "process_seconds_lower_bound": r["process_wall_seconds"], "process_seconds_upper_bound": r["process_wall_seconds"],
        "finished_unix_lower_bound": r["finished_unix"], "finished_unix_upper_bound": r["finished_unix"]} for r in processes]
    events = jsonl_read(exp / "process_events.jsonl")
    for lost in interrupted:
        owned = [e for e in events if e.get("label") == lost["label"] and e.get("event") == "started"]
        assert len(owned) == 1
        cmd = [c for c in commands if c["label"] == lost["label"]]
        assert len(cmd) == 1
        ledger.append({"label": lost["label"], "pid": owned[0]["pid"], "runner_pid": owned[0]["runner_pid"],
            "outer_budget_start_unix": lost["outer_budget_start_unix"], "started_unix": lost["started_unix"],
            "finished_unix": None, "process_wall_seconds": None, "exit_code": None, "timeout": None,
            "timeout_seconds": cmd[0]["timeout_seconds"], "log": "logs/"+exp.name+"-"+lost["label"]+".log",
            "receipt_status": "interrupted_bounded_no_exact_exit",
            "process_seconds_lower_bound": lost["process_seconds_lower_bound"],
            "process_seconds_upper_bound": lost["process_seconds_upper_bound"],
            "finished_unix_lower_bound": lost["last_raw_activity_unix"],
            "finished_unix_upper_bound": lost["next_OS_boot_event_unix"]})
    ledger.sort(key=lambda r: r["started_unix"])
    assert all(set(r) == set(ledger[0]) for r in ledger), "Every exact or interrupted row must fit one auditable ledger schema"
    complete_sum = sum(r["process_wall_seconds"] for r in processes)
    failed_exact = [r for r in processes if r["exit_code"] or r["timeout"] or r["label"] in failed_labels]
    interrupted_lower = sum(r["process_seconds_lower_bound"] for r in interrupted)
    interrupted_upper = sum(r["process_seconds_upper_bound"] for r in interrupted)
    summary = {"measured_owned_model_process_seconds": complete_sum,
        "exact_completed_process_receipt_count": len(processes), "interrupted_process_count": len(interrupted),
        "interrupted_model_process_seconds_lower_bound": interrupted_lower,
        "interrupted_model_process_seconds_upper_bound": interrupted_upper,
        "all_attempt_model_process_seconds_lower_bound": complete_sum+interrupted_lower,
        "all_attempt_model_process_seconds_upper_bound": complete_sum+interrupted_upper,
        "measured_owned_model_process_seconds_scope": "Exact completed process receipts only; interrupted model costs are included in all-attempt bounds",
        "completed_failed_process_seconds": sum(r["process_wall_seconds"] for r in failed_exact),
        "failed_process_seconds": None if interrupted else sum(r["process_wall_seconds"] for r in failed_exact),
        "failed_process_seconds_scope": "Unknown exact total when an interruption exists; use failed-model cost bounds, never impute zero",
        "completed_failed_process_count": len(failed_exact),
        "failed_model_process_seconds_lower_bound": sum(r["process_wall_seconds"] for r in failed_exact)+interrupted_lower,
        "failed_model_process_seconds_upper_bound": sum(r["process_wall_seconds"] for r in failed_exact)+interrupted_upper,
        "controller_preparation_failures_with_preserved_source": len(list(exp.glob("controller_recovery_*/failed_recovery_wrapper_source.py"))),
        "CPU_preparation_and_user_pause_cost_scope": "Included in registered session wall time; not falsely labeled GPU-active process time",
        "all_attempts_in_physical_ledger": True, "unknown_exit_codes_not_imputed": True}
    return summary, ledger, failed_by_endpoint, interrupted


def endpoint_failure_columns(row, failed):
    lower, upper = failed.get("lower_seconds", 0), failed.get("upper_seconds", 0)
    charge = row.get("charged_end_to_end_seconds")
    low_total = None if charge is None else charge+lower
    high_total = None if charge is None else charge+upper
    signal = row.get("effective_completions")
    gradients = row.get("nonzero_gradient_steps")
    return {"failed_attempt_count": failed.get("failed_attempts", 0),
        "failed_attempt_own_path_seconds_lower_bound": lower, "failed_attempt_own_path_seconds_upper_bound": upper,
        "charged_end_to_end_including_failed_attempts_seconds_lower_bound": low_total,
        "charged_end_to_end_including_failed_attempts_seconds_upper_bound": high_total,
        "failure_inclusive_C_overshoot_seconds_lower_bound": max(0, low_total-row["C_seconds"]) if low_total is not None and row.get("C_seconds") is not None else None,
        "failure_inclusive_C_overshoot_seconds_upper_bound": max(0, high_total-row["C_seconds"]) if high_total is not None and row.get("C_seconds") is not None else None,
        "effective_completions_per_failure_inclusive_second_lower_bound": signal/high_total if signal is not None and high_total else None,
        "effective_completions_per_failure_inclusive_second_upper_bound": signal/low_total if signal is not None and low_total else None,
        "nonzero_gradient_steps_per_failure_inclusive_second_lower_bound": gradients/high_total if gradients is not None and high_total else None,
        "nonzero_gradient_steps_per_failure_inclusive_second_upper_bound": gradients/low_total if gradients is not None and low_total else None}


def operational_rate_bounds(endpoints, arm, probe=None):
    rows = [r for r in endpoints if r["arm"] == arm and (probe is None or r["probe_seed"] == probe)]
    assert rows and all(r["charged_end_to_end_seconds"] is not None for r in rows)
    low = sum(r["charged_end_to_end_including_failed_attempts_seconds_lower_bound"] for r in rows)
    high = sum(r["charged_end_to_end_including_failed_attempts_seconds_upper_bound"] for r in rows)
    effective = sum(r["effective_completions"] for r in rows)
    gradients = sum(r["nonzero_gradient_steps"] for r in rows)
    return {"endpoint_count": len(rows), "failure_inclusive_charged_seconds_lower_bound": low,
        "failure_inclusive_charged_seconds_upper_bound": high,
        "effective_completions_per_second_lower_bound": effective/high,
        "effective_completions_per_second_upper_bound": effective/low,
        "nonzero_gradient_steps_per_second_lower_bound": gradients/high,
        "nonzero_gradient_steps_per_second_upper_bound": gradients/low,
        "scope": "Cost-bound sensitivity for retained endpoint signals; not a confidence interval or extra independent sample"}


def selfcheck():
    row = {"charged_end_to_end_seconds": 100, "effective_completions": 20, "nonzero_gradient_steps": 5}
    d = endpoint_failure_columns(row, {"lower_seconds": 10, "upper_seconds": 20, "failed_attempts": 1})
    assert d["failed_attempt_count"] == 1
    assert d["charged_end_to_end_including_failed_attempts_seconds_lower_bound"] == 110
    assert d["charged_end_to_end_including_failed_attempts_seconds_upper_bound"] == 120
    assert math.isclose(d["effective_completions_per_failure_inclusive_second_lower_bound"], 20/120)
    assert math.isclose(d["effective_completions_per_failure_inclusive_second_upper_bound"], 20/110)
    assert d["effective_completions_per_failure_inclusive_second_upper_bound"] < 20/100
    return {"status": "passed", "failed_time_added_without_adding_discarded_training_signal": True,
        "inverse_rate_bounds_in_correct_order": True, "synthetic_only": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    exp = ROOT / "experiments" / args.family
    test = selfcheck()
    physical, ledger, failures, interrupted = accounting(exp, read(exp / "training_units.json"), {})
    assert physical["interrupted_process_count"] == 1 and physical["failed_model_process_seconds_lower_bound"] > 0
    assert failures["U-s47"]["failed_attempts"] == 1
    assert any(r["label"] == "U-s47-a0" and r["process_wall_seconds"] is None for r in ledger)
    json_write(exp / "failure_cost_accounting_pre_fresh_selfcheck.json", {"synthetic": test,
        "live_training_cost_only": physical, "U47_failed_path_bounds": failures["U-s47"],
        "fresh_predictions_read": False, "source_sha256": sha256_file(ROOT / "analysis/large_filter_cost_accounting.py")})
    print({"selfcheck": "passed", "interrupted_attempts": len(interrupted),
        "failure_model_seconds_bounds": [physical["failed_model_process_seconds_lower_bound"], physical["failed_model_process_seconds_upper_bound"]],
        "no_fresh_predictions_or_model_calls": True})
