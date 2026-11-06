"""Measure actual full probing and training costs before freezing final C/D/E."""
import copy
import json
import time
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file
from repair_common import FAMILY, EXP, read, run_phase


def main():
    status_path = ROOT / "environment" / (FAMILY + "-calibration-status.json")
    if status_path.exists():
        raise FileExistsError(status_path)
    state = {"status": "running", "completed": [], "started_unix": time.time(), "holdout_opened": False}
    json_write(status_path, state)
    try:
        run_phase("initialize-start", ["scripts/initialize_repair_start.py"], state, status_path)
        start = read(EXP / "common_start.json")
        base = yaml.safe_load((ROOT / f"configs/{FAMILY}_eval.yaml").read_text(encoding="utf-8"))
        base["stage"] = "desktop_rollout_cache_calibration"
        base["grpo"].update(pool_size=64, pool_offset=128, probe_samples=4, steps=12,
            sampling_mechanism="rollout_cache_fifo_v1", cache_max_trials=16, cache_ttl_steps=32,
            max_wall_seconds=900, final_validation_reserve_seconds=0, standalone_endpoints_only=True,
            validation_interval=100000, max_accounted_tokens=10000000)
        configs = {}
        for arm, steps in (("C", 8), ("D", 12), ("E", 1)):
            config = copy.deepcopy(base)
            config["grpo"]["steps"] = steps
            relative = f"configs/{FAMILY}_calibration_{arm}.yaml"
            (ROOT / relative).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
            configs[arm] = relative
        json_write(EXP / "calibration_preregistration.json", {"created_unix": time.time(),
            "configs": {k: {"path": p, "sha256": sha256_file(ROOT / p)} for k, p in configs.items()},
            "initial_checkpoint": start, "scope": "cost and learning-signal diagnostics only, not scientific endpoints",
            "pool": "64 train candidates after first128, four previously flagged train items excluded =60",
            "criteria": "D must finish full probe, consume real rollouts, change weights, have finite nonzero gradient and nonzero adapter delta; D/E initial probes and weights must match",
            "no_parameter_search": "one fixed C8/D12/E1 diagnostic; failure stops main rather than repeatedly tuning to obtain signal",
            "pool_selection": "At final freeze choose largest of60,44,28 with linearly estimated initial K4 probe <=35% of training budget; estimate from measured60-item full probe. Training cap selected from600/480/360 by remaining session time, reserving comparison+confirmation and900s handoff."})
        source_flag = "--sft-checkpoint" if start["kind"] == "sft" else "--public-initialization"
        for arm, strategy in (("C", "uniform"), ("D", "dynamic"), ("E", "static")):
            name = FAMILY + "-calibration-" + arm
            run_phase("calibration-"+arm, ["-m", "reasoning_rl", "grpo", "--name", name,
                "--hardware", "rocm", "--config", configs[arm], "--strategy", strategy,
                source_flag, start["checkpoint"]], state, status_path)
            run_phase("calibration-"+arm+"-reloadregression", ["-m", "reasoning_rl", "evaluate", "--name", name+"-reloadregression",
                "--hardware", "rocm", "--config", f"configs/{FAMILY}_regression.yaml", "--adapter", "checkpoints/"+name], state, status_path)
        cdir = ROOT / "runs" / (FAMILY+"-calibration-C")
        ddir = ROOT / "runs" / (FAMILY+"-calibration-D")
        edir = ROOT / "runs" / (FAMILY+"-calibration-E")
        dc, ec = [read(p / "probes/step_00000_state.json") for p in (ddir, edir)]
        dp, ep = [jsonl_read(p / "probes/step_00000.jsonl") for p in (ddir, edir)]
        assert dc["weights"] == ec["weights"]
        assert [(r["id"],r["prediction"]) for r in dp] == [(r["id"],r["prediction"]) for r in ep]
        result = read(ddir / "training_result.json")
        gradients = [r for r in jsonl_read(ddir / "costs.jsonl") if r["phase"] == "gradient_diagnostic"]
        assert result["adapter_updated"] and result["cache_rollout_update_events"] > 0
        assert result["sampler_weight_change_events"] > 0
        assert all(r["finite_gradients"] and not r["frozen_parameter_gradients"] for r in gradients)
        assert any(r["nonzero_gradient_elements"] for r in gradients)
        costs = {arm: jsonl_read(p / "costs.jsonl") for arm, p in (("C",cdir),("D",ddir),("E",edir))}
        measured = {}
        for arm, values in costs.items():
            measured[arm] = {"grpo_seconds": sum(r.get("wall_seconds",0) for r in values if r["phase"]=="grpo_total"),
                "full_probe_seconds": sum(r.get("wall_seconds",0) for r in values if r["phase"]=="difficulty_probe"),
                "model_load_seconds": sum(r.get("wall_seconds",0) for r in values if r["phase"]=="load_model")}
            actual = read(ROOT / "runs" / (FAMILY+"-calibration-"+arm) / "training_result.json")
            measured[arm].update(steps=actual["completed_optimizer_steps"], rollout_tokens=actual["generated_tokens"],
                cache_maintenance_seconds=actual["sampler_maintenance_seconds"])
        eval_seconds = {}
        for suffix in ("A-comparison","S1-comparison","S2-comparison","S3-comparison","S4-comparison"):
            values = jsonl_read(ROOT / "runs" / (FAMILY+"-"+suffix) / "costs.jsonl")
            eval_seconds[suffix] = sum(r.get("wall_seconds",0) for r in values if r["phase"] in {"validation","load_model"})
        json_write(EXP / "calibration_result.json", {"status":"pass", "actual_costs":measured,
            "comparison128_evaluation_seconds":eval_seconds, "D_E_initial_weights_exactly_equal":True,
            "D_E_initial_predictions_exactly_equal":True, "D_nonzero_gradient_steps":sum(r["nonzero_gradient_elements"]>0 for r in gradients),
            "D_adapter_delta_l2":result["adapter_delta_l2"], "candidate_pool_size_after_exclusion":60,
            "initial_effective_pool_size":dc["effective_pool_size"], "final_effective_pool_size":read(ddir/"sampler_final.json")["effective_pool_size"],
            "full_probe_cost_ratio":measured["D"]["full_probe_seconds"]/measured["D"]["grpo_seconds"]})
        state.update(status="completed_calibration",finished_unix=time.time())
        json_write(status_path,state)
    except BaseException as error:
        state.update(status="failed",error=repr(error),finished_unix=time.time())
        json_write(status_path,state)
        raise


if __name__=="__main__":
    main()
