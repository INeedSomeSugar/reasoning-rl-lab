"""Six immutable training endpoints, followed by six independent reload evaluations."""
import json
import time
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file, checkpoint_hash, code_hash
from filter_common import EXP, FAMILY, START, START_HASH, read, run_phase, remaining


def verify(plan):
    if code_hash() != plan["code_sha256"]:
        raise AssertionError("Source/config changed after freeze")
    for relative, digest in read(EXP / "frozen_source_manifest.json").items():
        if sha256_file(ROOT / relative) != digest:
            raise AssertionError(f"Frozen executable changed: {relative}")
    for path, key in (("candidate_pool.json", "candidate_pool_sha256"),
        ("prefilter_manifest.json", "prefilter_manifest_sha256"), ("development_ids.json", "development_ids_sha256")):
        assert sha256_file(EXP / path) == plan[key]
    assert checkpoint_hash(ROOT / START) == START_HASH == plan["common_start_sha256"]
    assert sha256_file(ROOT / "data/manifest.json") == plan["data_manifest_sha256"]


def main():
    plan = read(EXP / "protocol_frozen.json")
    status_path = ROOT / "environment" / (FAMILY+"-status.json")
    if status_path.exists():
        raise FileExistsError(status_path)
    state = {"status": "running_training", "started_unix": time.time(), "completed_training": [],
        "completed_evaluation": [], "holdout_opened": False}
    json_write(status_path, state)
    endpoints = {}
    cap = plan["budget"]["end_to_end_common_cap_seconds"]
    train_allocation = plan["budget"]["selected"]["train_allocation_seconds"]
    probe_cost = plan["probe_cost"]["primary_standalone_debit_per_F_endpoint_seconds"]
    try:
        for seed, arm in plan["order"]:
            label = f"{arm}-s{seed}"
            prep_started = time.perf_counter()
            verify(plan)
            debit = probe_cost if arm == "F" else 0.0
            if debit >= train_allocation:
                raise RuntimeError("Measured probe cost exhausts frozen F training allocation")
            name = FAMILY+"-"+label
            cfg = plan["configs"][label]["path"]
            preparation = time.perf_counter()-prep_started
            state.update(phase=label)
            json_write(status_path, state)
            training = run_phase(label, ["-m", "reasoning_rl", "grpo", "--name", name,
                "--hardware", "rocm", "--config", cfg, "--strategy", "uniform", "--sft-checkpoint", START],
                timeout=train_allocation-debit+40)
            check_started = time.perf_counter()
            run_dir = ROOT / "runs" / name
            contract = read(run_dir / "comparison_contract.json")
            result = read(run_dir / "training_result.json")
            expected = plan["candidate_pool_ids"] if arm == "U" else plan["retained_ids"]
            assert contract["pool_ids"] == expected
            assert contract["candidate_pool_ids"] == plan["candidate_pool_ids"]
            assert contract["initial_checkpoint_sha256"] == START_HASH
            assert contract["initial_probe_standalone_cost_debit_seconds"] == debit
            assert result["actual_signal_checked_groups"] == result["groups"] == result["completed_optimizer_steps"]
            assert result["actual_nonzero_advantage_completions"] == result["nonzero_advantage_completions"]
            assert checkpoint_hash(ROOT / "checkpoints" / name) == result["checkpoint_sha256"]
            costs = jsonl_read(run_dir / "costs.jsonl")
            updates = [r for r in costs if r["phase"] == "optimizer_weight_update"]
            grads = [r for r in costs if r["phase"] == "gradient_diagnostic"]
            assert len(updates) == len(grads) == result["completed_optimizer_steps"]
            assert all(g["finite_gradients"] and g["frozen_parameter_gradients"] == 0 for g in grads)
            checking = time.perf_counter()-check_started
            endpoints[label] = {"arm": arm, "seed": seed, "training_run": name, "config": cfg,
                "checkpoint_sha256": result["checkpoint_sha256"], "training_process_seconds": training,
                "initial_probe_standalone_debit_seconds": debit,
                "parent_preparation_and_training_audit_seconds": preparation+checking}
            state["completed_training"].append(label)
            json_write(EXP / "endpoint_costs.json", endpoints)
            json_write(status_path, state)
            print(json.dumps({"completed_training": label, "steps": result["completed_optimizer_steps"],
                "charged_process_seconds": training+debit, "remaining_session_seconds": remaining()}), flush=True)
        json_write(EXP / "all_training_endpoints_fixed.json", {"fixed_unix": time.time(),
            "before_any_main_endpoint_evaluation": True, "endpoints": endpoints,
            "protocol_sha256": sha256_file(EXP / "protocol_frozen.json")})
        state["status"] = "running_independent_evaluation"
        json_write(status_path, state)
        for seed, arm in plan["order"]:
            label = f"{arm}-s{seed}"
            endpoint = endpoints[label]
            prep_started = time.perf_counter()
            verify(plan)
            assert checkpoint_hash(ROOT / "checkpoints" / endpoint["training_run"]) == endpoint["checkpoint_sha256"]
            prep = time.perf_counter()-prep_started
            already = (endpoint["training_process_seconds"]+endpoint["initial_probe_standalone_debit_seconds"]
                +endpoint["parent_preparation_and_training_audit_seconds"]+prep)
            allowed = cap-already
            if allowed <= 0:
                raise TimeoutError(f"No independent evaluation budget remains for {label}")
            state["phase"] = label+"-development"
            json_write(status_path, state)
            evaluated = run_phase(label+"-development", ["-m", "reasoning_rl", "evaluate", "--name",
                endpoint["training_run"]+"-development", "--hardware", "rocm", "--config", endpoint["config"],
                "--adapter", "checkpoints/"+endpoint["training_run"]], timeout=allowed)
            finish_started = time.perf_counter()
            eval_path = ROOT / "runs" / (endpoint["training_run"]+"-development")
            metrics = read(eval_path / "validation/metrics.json")
            assert metrics["n"] == 128 and metrics["decoding"] == "greedy"
            assert checkpoint_hash(ROOT / "checkpoints" / endpoint["training_run"]) == endpoint["checkpoint_sha256"]
            finish = time.perf_counter()-finish_started
            endpoint.update(evaluation_process_seconds=evaluated,
                parent_evaluation_preparation_and_audit_seconds=prep+finish,
                charged_end_to_end_seconds=already+evaluated+finish, common_cap_seconds=cap,
                correct=metrics["correct"], n=metrics["n"])
            endpoint["unused_common_cap_seconds"] = cap-endpoint["charged_end_to_end_seconds"]
            assert endpoint["charged_end_to_end_seconds"] <= cap
            state["completed_evaluation"].append(label)
            json_write(EXP / "endpoint_costs.json", endpoints)
            json_write(status_path, state)
            print(json.dumps({"completed_independent_evaluation": label, "correct": metrics["correct"],
                "n": 128, "charged_end_to_end_seconds": endpoint["charged_end_to_end_seconds"]}), flush=True)
        verify(plan)
        state.update(status="completed_three_paired_seed_development", finished_unix=time.time())
        json_write(status_path, state)
    except BaseException as error:
        state.update(status="failed_or_stopped", error=repr(error), finished_unix=time.time())
        json_write(status_path, state)
        raise


if __name__ == "__main__":
    main()
