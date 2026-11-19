"""Persistent staged controller for the full five-by-three fixed-filter/random study."""
import argparse
import json
import os
import time
import traceback

from reasoning_rl.runtime import ROOT, json_write, sha256_file, checkpoint_hash, code_hash, jsonl_read, jsonl_append
from large_filter_common import START, START_HASH, PROBE_SEEDS, read, run_phase, write_new, write_state, utc_now


def calibration(family):
    exp = ROOT / "experiments" / family
    if (exp / "calibration_complete.json").exists():
        raise FileExistsError("Calibration already completed")
    write_state(family, stage="calibration", status="running")
    phases = [
        ("preflight-a0", ["scripts/large_filter_tasks.py", "--family", family, "--task", "preflight"], 600),
        ("old-development64-a0", ["-m", "reasoning_rl", "evaluate", "--name", family+"-old-development64-a0",
            "--config", f"configs/{family}_calibration.yaml", "--hardware", "rocm", "--adapter", START], 1200),
        ("calibration-train-a0", ["-m", "reasoning_rl", "grpo", "--name", family+"-calibration-train-a0",
            "--config", f"configs/{family}_calibration_train.yaml", "--hardware", "rocm", "--strategy", "uniform",
            "--sft-checkpoint", START], 400),
    ]
    costs = {}
    for label, args, limit in phases:
        write_state(family, stage="calibration", phase=label, status="running")
        receipt = run_phase(family, label, args, limit)
        costs[label] = receipt
        if receipt["exit_code"]:
            raise RuntimeError(f"Calibration task failed: {label}")
    result = read(ROOT / "runs" / (family+"-calibration-train-a0") / "training_result.json")
    assert result["completed_optimizer_steps"] == 8
    write_new(exp / "calibration_complete.json", {"completed_utc": utc_now(), "process_costs": costs,
        "old64_evaluation_process_seconds": costs["old-development64-a0"]["process_wall_seconds"],
        "optimizer_steps": 8, "fresh_development_accessed": False, "accuracy_not_used_for_budget": True})
    write_state(family, stage="calibration", status="completed")


def probes(family):
    exp = ROOT / "experiments" / family
    assert (exp / "data_frozen.json").exists() and (exp / "calibration_complete.json").exists()
    files = ["src/reasoning_rl/"+n+".py" for n in ("modeling", "evaluation", "scoring", "data", "fresh_development", "prefiltering")]
    files.extend(["scripts/large_filter_tasks.py", f"configs/{family}_calibration.yaml"])
    frozen_inputs = {p: sha256_file(ROOT / p) for p in files}
    write_new(exp / "probe_inputs_frozen.json", {"frozen_utc": utc_now(), "order": PROBE_SEEDS,
        "files": frozen_inputs, "candidate_pool_sha256": sha256_file(exp / "candidate_pool.json"),
        "common_start_sha256": START_HASH, "attempt_limit": 2, "full_process_timeout_seconds": 3600})
    completed = {}
    for seed in PROBE_SEEDS:
        for p, digest in frozen_inputs.items():
            assert sha256_file(ROOT / p) == digest
        assert checkpoint_hash(ROOT / START) == START_HASH
        attempts = []
        for attempt in (0, 1):
            label = f"probe-j{seed}-a{attempt}"
            write_state(family, stage="probes", phase=label, status="running", completed_probe_seeds=list(completed))
            receipt = run_phase(family, label, ["scripts/large_filter_tasks.py", "--family", family,
                "--task", "probe", "--probe-seed", str(seed), "--attempt", str(attempt)], 3600)
            attempts.append(receipt)
            if receipt["exit_code"] == 0:
                directory = ROOT / "runs" / f"{family}-{label}"
                result = read(directory / "probe_result.json")
                assert len(result["candidate_ids"]) == 256
                completed[str(seed)] = {"run": directory.name, "attempt": attempt, "attempts": attempts,
                    "P_seconds": receipt["process_wall_seconds"], "retained_n": len(result["retained_ids"]),
                    "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
                    "probe_result_sha256": sha256_file(directory / "probe_result.json")}
                json_write(exp / "probe_process_results.json", completed)
                break
        else:
            raise RuntimeError(f"Probe seed{seed} exhausted its one full retry")
    write_new(exp / "all_probes_complete.json", {"completed_utc": utc_now(), "probes": completed,
        "new_development_inference_started": False})
    write_state(family, stage="probes", status="completed", completed_probe_seeds=list(completed))


def verify_frozen(family, plan):
    exp = ROOT / "experiments" / family
    if code_hash() != plan["code_sha256"]:
        raise ValueError("Source/config changed after main freeze")
    for relative, digest in read(exp / "frozen_source_manifest.json").items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen executable changed: {relative}")
    for relative, digest in plan["artifacts"].items():
        if sha256_file(exp / relative) != digest:
            raise ValueError(f"Frozen evidence changed: {relative}")
    if checkpoint_hash(ROOT / START) != START_HASH:
        raise ValueError("Shared real S2 changed")
    if sha256_file(ROOT / plan["fresh_manifest_path"]) != plan["fresh_manifest_sha256"]:
        raise ValueError("Fresh development manifest changed")
    if sha256_file(ROOT / "data/fresh_development/registry.json") != plan["reservation_registry_sha256"]:
        raise ValueError("Fresh development reservations changed")
    for item in read(exp / "preflight.json")["public_model_files"]:
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Public model/tokenizer cache changed before an endpoint")


def expected_pool(family, unit):
    exp = ROOT / "experiments" / family
    if unit["arm"] == "U":
        return read(exp / "candidate_pool.json")["ids"]
    name = ("prefilter" if unit["arm"] == "F" else "random")+f"_j{unit['probe_seed']}.json"
    item = read(exp / name)
    return item["retained_ids"] if unit["arm"] == "F" else item["ids"]


def audit_training(family, unit, run_name, debit):
    run_dir = ROOT / "runs" / run_name
    contract = read(run_dir / "comparison_contract.json")
    result = read(run_dir / "training_result.json")
    assert contract["initial_checkpoint_sha256"] == START_HASH
    assert contract["pool_ids"] == expected_pool(family, unit)
    assert contract["candidate_pool_ids"] == read(ROOT / "experiments" / family / "candidate_pool.json")["ids"]
    assert contract["initial_probe_standalone_cost_debit_seconds"] == debit
    assert result["initial_checkpoint_sha256"] == START_HASH
    assert result["groups"] == result["completed_optimizer_steps"] == result["actual_signal_checked_groups"] > 0
    costs = jsonl_read(run_dir / "costs.jsonl")
    grads = [r for r in costs if r["phase"] == "gradient_diagnostic"]
    updates = [r for r in costs if r["phase"] == "optimizer_weight_update"]
    assert len(grads) == len(updates) == result["completed_optimizer_steps"]
    assert all(r["finite_gradients"] and r["frozen_parameter_gradients"] == 0 for r in grads)
    assert checkpoint_hash(ROOT / "checkpoints" / run_name) == result["checkpoint_sha256"]
    assert (ROOT / "checkpoints" / run_name / "optimizer_diagnostic_state.pt").exists()
    assert not list((run_dir / "fresh_development").glob("predictions.jsonl"))
    return result


def reconcile_interrupted_attempt(family, item, stage):
    """Recover only execution status. Never resume optimizer/sampler position."""
    import psutil
    exp = ROOT / "experiments" / family
    if item.get("status") != "running_attempt":
        return
    attempt = item["attempts"][-1]
    pid = attempt.get("pid")
    if pid and psutil.pid_exists(pid):
        raise RuntimeError(f"Recorded {stage} child pid{pid} still exists; do not launch a duplicate")
    # Parent may have died before recording the child PID; inspect its append-only start event.
    events_path = exp / "process_events.jsonl"
    matching = [e for e in jsonl_read(events_path) if e["label"] == attempt["label"]] if events_path.exists() else []
    if matching and psutil.pid_exists(matching[-1]["pid"]):
        raise RuntimeError("Prior owned child still exists; wait for it or investigate, do not duplicate")
    lower = max(0, (ROOT / "logs" / f"{family}-{attempt['label']}.log").stat().st_mtime-attempt["started_unix"]) if (ROOT / "logs" / f"{family}-{attempt['label']}.log").exists() else 0
    receipt = {"label": attempt["label"], "status": "interrupted_no_final_receipt",
        "started_unix": attempt["started_unix"], "confirmed_dead_unix": time.time(),
        "observed_process_seconds_lower_bound": lower,
        "process_seconds_upper_bound": time.time()-attempt["started_unix"],
        "exact_exit_code": None, "exact_resume_supported": False}
    attempt.update(receipt=receipt, status="failed_interrupted")
    item["status"] = "pending_retry"
    jsonl_append(exp / "interruption_cost_bounds.jsonl", [receipt])


def controlled(family):
    exp = ROOT / "experiments" / family
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    units_path = exp / "training_units.json"
    units = read(units_path) if units_path.exists() else {
        unit["key"]: {**unit, "status": "pending", "attempts": []} for unit in plan["order"]}
    budget = plan["budget"]
    lock_path = exp / "all_training_endpoints_fixed.json"
    if not lock_path.exists():
        if (exp / "fresh_development_opened.json").exists():
            raise ValueError("Fresh evaluation already opened; additional training prohibited")
        for unit in plan["order"]:
            key = unit["key"]
            item = units[key]
            reconcile_interrupted_attempt(family, item, "training")
            if item["status"] in {"first_success_fixed", "exhausted_failure"}:
                continue
            debit = 0 if unit["arm"] == "U" else budget["P_seconds"][str(unit["probe_seed"])]
            if debit >= budget["B_seconds"]:
                raise ValueError("Probe exhausts frozen training allocation")
            for attempt in range(len(item["attempts"]), 2):
                prep_started_unix = time.time()
                prep_start = time.perf_counter()
                verify_frozen(family, plan)
                cfg = plan["configs"][key]["path"]
                run_name = f"{family}-{key}-a{attempt}"
                label = key+f"-a{attempt}"
                record = {"attempt": attempt, "run_name": run_name, "label": label,
                    "started_unix": time.time(), "initial_checkpoint_sha256": START_HASH,
                    "config_sha256": plan["configs"][key]["sha256"]}
                item["attempts"].append(record)
                item["status"] = "running_attempt"
                json_write(units_path, units)
                write_state(family, stage="main_training", phase=label, status="running",
                    completed_training=[k for k, v in units.items() if v["status"] == "first_success_fixed"])
                parent_preparation = time.perf_counter()-prep_start
                receipt = run_phase(family, label, ["-m", "reasoning_rl", "grpo", "--name", run_name,
                    "--hardware", "rocm", "--config", cfg, "--strategy", "uniform", "--sft-checkpoint", START],
                    budget["B_seconds"]-debit+budget["O_seconds"]/2-parent_preparation,
                    budget_start_unix=prep_started_unix)
                record["receipt"] = receipt
                audit_start = time.perf_counter()
                try:
                    if receipt["exit_code"] or receipt["timeout"]:
                        raise RuntimeError("Training process failed or timed out")
                    result = audit_training(family, unit, run_name, debit)
                except Exception as error:
                    record.update(status="failed", error=repr(error))
                    item["status"] = "pending_retry"
                    jsonl_append(exp / "attempt_validation_failures.jsonl", [{"key": key, "attempt": attempt,
                        "error": repr(error), "traceback": traceback.format_exc()}])
                else:
                    record["status"] = "success"
                    item.update(status="first_success_fixed", selected_attempt=attempt, run_name=run_name,
                        checkpoint="checkpoints/"+run_name, checkpoint_sha256=result["checkpoint_sha256"],
                        training_process_seconds=receipt["process_wall_seconds"],
                        parent_training_preparation_and_audit_seconds=parent_preparation+time.perf_counter()-audit_start,
                        probe_algorithm_debit_seconds=debit if unit["arm"] == "F" else 0,
                        R_experimental_matching_debit_seconds=debit if unit["arm"] == "R" else 0,
                        total_probe_or_matching_debit_seconds=debit,
                        completed_optimizer_steps=result["completed_optimizer_steps"])
                json_write(units_path, units)
                if item["status"] == "first_success_fixed":
                    break
            else:
                item["status"] = "exhausted_failure"
                json_write(units_path, units)
        assert all(x["status"] in {"first_success_fixed", "exhausted_failure"} for x in units.values())
        verify_frozen(family, plan)
        objects = {"A": {"status": "fixed", "checkpoint": None, "checkpoint_sha256": None},
            "S2": {"status": "fixed", "checkpoint": START, "checkpoint_sha256": START_HASH}}
        objects.update({k: {"status": "fixed" if v["status"] == "first_success_fixed" else "missing",
            "checkpoint": v.get("checkpoint"), "checkpoint_sha256": v.get("checkpoint_sha256")}
            for k, v in units.items()})
        write_new(lock_path, {"stage": "all_training_endpoints_fixed_before_fresh_evaluation", "frozen_utc": utc_now(),
            "family": family, "registered_training_keys": list(units), "training_units": units, "objects": objects,
            "fresh_development_manifest_sha256": plan["fresh_manifest_sha256"], "protocol_sha256": sha256_file(exp / "protocol_frozen.json"),
            "public_model_files": read(exp / "preflight.json")["public_model_files"],
            "all35_successful": all(v["status"] == "first_success_fixed" for v in units.values()),
            "no_fresh_predictions_before_lock": not (exp / "fresh_development_opened.json").exists()})
    # There is deliberately no training path below this immutable gate.
    lock = read(lock_path)
    assert lock["protocol_sha256"] == sha256_file(exp / "protocol_frozen.json")
    evaluations_path = exp / "evaluation_units.json"
    evaluations = read(evaluations_path) if evaluations_path.exists() else {
        key: {"status": "pending" if lock["objects"][key]["status"] == "fixed" else "missing_training_endpoint", "attempts": []}
        for key in plan["evaluation_order"]}
    if not (exp / "fresh_development_opened.json").exists():
        write_new(exp / "fresh_development_opened.json", {"opened_utc": utc_now(),
            "endpoint_lock_sha256": sha256_file(lock_path), "training_permanently_closed_for_this_family": True,
            "all35_terminal": True, "all35_successful": lock["all35_successful"]})
    for key in plan["evaluation_order"]:
        item = evaluations[key]
        reconcile_interrupted_attempt(family, item, "evaluation")
        if item["status"] in {"first_success_fixed", "exhausted_failure", "missing_training_endpoint"}:
            continue
        obj = lock["objects"][key]
        for attempt in range(len(item["attempts"]), 2):
            prep_start = time.perf_counter()
            verify_frozen(family, plan)
            name = f"{family}-eval-{key}-a{attempt}"
            label = f"eval-{key}-a{attempt}"
            record = {"attempt": attempt, "run_name": name, "label": label, "started_unix": time.time()}
            item["attempts"].append(record)
            item["status"] = "running_attempt"
            json_write(evaluations_path, evaluations)
            write_state(family, stage="fresh_development_evaluation", phase=label, status="running",
                completed_training=[k for k, v in units.items() if v["status"] == "first_success_fixed"],
                completed_evaluation=[k for k, v in evaluations.items() if v["status"] == "first_success_fixed"])
            already = 0
            if key in units and units[key]["status"] == "first_success_fixed":
                u = units[key]
                already = u["training_process_seconds"]+u["parent_training_preparation_and_audit_seconds"]+u["total_probe_or_matching_debit_seconds"]
            parent_prep = time.perf_counter()-prep_start
            allowed = min(budget["E_seconds"]+budget["O_seconds"]/2, budget["C_seconds"]-already-parent_prep)
            args = ["-m", "reasoning_rl", "evaluate", "--name", name, "--hardware", "rocm",
                "--config", plan["evaluation_config"]["path"], "--split", "fresh_development",
                "--endpoint-lock", lock_path.relative_to(ROOT).as_posix(), "--endpoint-key", key]
            if obj["checkpoint"]:
                args.extend(["--adapter", obj["checkpoint"]])
            receipt = run_phase(family, label, args, allowed)
            record["receipt"] = receipt
            audit_start = time.perf_counter()
            try:
                if receipt["exit_code"] or receipt["timeout"]:
                    raise RuntimeError("Independent evaluation failed or timed out")
                directory = ROOT / "runs" / name
                metrics = read(directory / "fresh_development/metrics.json")
                assert metrics["n"] == plan["development_n"] and metrics["decoding"] == "greedy"
                ids = read(directory / "resolved_evaluation_ids.json")["ids"]
                assert ids == read(ROOT / plan["fresh_manifest_path"])["ids"]
                protocol = read(directory / "evaluation_protocol.json")
                assert protocol["adapter_sha256"] == obj["checkpoint_sha256"]
            except Exception as error:
                record.update(status="failed", error=repr(error))
                item["status"] = "pending_retry"
                jsonl_append(exp / "attempt_validation_failures.jsonl", [{"key": key, "attempt": attempt,
                    "stage": "evaluation", "error": repr(error), "traceback": traceback.format_exc()}])
            else:
                record["status"] = "success"
                total = already+parent_prep+receipt["process_wall_seconds"]+time.perf_counter()-audit_start
                item.update(status="first_success_fixed", selected_attempt=attempt, run_name=name,
                    evaluation_process_seconds=receipt["process_wall_seconds"],
                    parent_evaluation_preparation_and_audit_seconds=parent_prep+time.perf_counter()-audit_start,
                    correct=metrics["correct"], n=metrics["n"],
                    charged_end_to_end_seconds=total, C_seconds=budget["C_seconds"],
                    common_cap_overshoot_seconds=max(0, total-budget["C_seconds"]),
                    predictions_sha256=sha256_file(directory / "fresh_development/predictions.jsonl"))
            json_write(evaluations_path, evaluations)
            if item["status"] == "first_success_fixed":
                break
        else:
            item["status"] = "exhausted_failure"
            json_write(evaluations_path, evaluations)
    verify_frozen(family, plan)
    write_new(exp / "all_model_tasks_complete.json", {"completed_utc": utc_now(),
        "successful_training": sum(v["status"] == "first_success_fixed" for v in units.values()),
        "registered_training": 35, "successful_evaluation": sum(v["status"] == "first_success_fixed" for v in evaluations.values()),
        "registered_evaluation": 37, "failed_units_preserved": True, "no_further_training": True})
    write_state(family, stage="all_model_tasks", status="completed", successful_training=sum(v["status"] == "first_success_fixed" for v in units.values()),
        successful_evaluation=sum(v["status"] == "first_success_fixed" for v in evaluations.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", choices=["calibration", "probes", "controlled"], required=True)
    args = parser.parse_args()
    exp = ROOT / "experiments" / args.family
    write_new(exp / f"runner_{args.stage}_pid.json", {"pid": os.getpid(), "started_utc": utc_now()})
    awake = None
    if os.name == "nt":
        import ctypes
        # A process-lifetime idle-sleep request, not a persistent power-plan change.
        awake = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        jsonl_append(exp / "process_events.jsonl", [{"event": "idle_sleep_prevention_request",
            "stage": args.stage, "runner_pid": os.getpid(), "return_value": awake, "time_unix": time.time(),
            "scope": "This runner lifetime only; does not prevent explicit user sleep/shutdown"}])
    try:
        globals()[args.stage](args.family)
    except BaseException as error:
        write_state(args.family, stage=args.stage, status="failed", error=repr(error))
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
