"""Cost-based capacity calibration, then immutable three-seed C/D/E registration."""
import copy
import hashlib
import json
import math
import time
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file, checkpoint_hash, code_hash
from reasoning_rl.data import load_split
from reasoning_rl.modeling import render_prompt
from repair_common import FAMILY, EXP, read, remaining


def main():
    target = EXP / "sampling_frozen.json"
    if target.exists():
        raise FileExistsError(target)
    calibration = read(EXP / "calibration_result.json")
    assert calibration["status"] == "pass"
    start = read(EXP / "common_start.json")
    assert checkpoint_hash(ROOT / start["checkpoint"]) == start["checkpoint_sha256"]
    decision = read(EXP / "sft_decision.json")
    comparisons = calibration["comparison128_evaluation_seconds"]
    # Use all four measured candidates, including the slow degraded endpoint.
    eval_reserve = math.ceil(max(comparisons.values())/30)*30 + 60
    confirmation_n = 11 if start["kind"] == "sft" else 10
    candidates = []
    for seconds in (600,480,360):
        estimated = 9*(seconds+eval_reserve+30) + confirmation_n*(eval_reserve+30) + 900
        candidates.append({"training_cap":seconds, "reserved_remaining_seconds":estimated,
                           "fits_current_session":estimated <= remaining()})
    usable = next((r["training_cap"] for r in candidates if r["fits_current_session"]), None)
    if usable is None:
        json_write(EXP / "sampling_not_started.json", {"reason":"Insufficient session time for a complete balanced comparison and one-stage confirmation",
            "remaining_seconds":remaining(),"capacity_candidates":candidates})
        raise TimeoutError("Preserve diagnostics: not enough time to preregister the complete intended experiment")
    full_probe = calibration["actual_costs"]["D"]["full_probe_seconds"]
    pool_candidates = [{"retained_size":n,"candidate_size":n+4,"estimated_probe_seconds":full_probe*n/60,
                        "estimated_probe_fraction":full_probe*n/60/usable} for n in (60,44,28)]
    chosen = next((r for r in pool_candidates if r["estimated_probe_fraction"]<=0.35), pool_candidates[-1])
    if chosen["estimated_probe_fraction"]>0.60:
        json_write(EXP/"sampling_not_started.json",{"reason":"Minimum28-item initial probe estimated above60% of common cap",
            "pool_candidates":pool_candidates,"training_cap":usable})
        raise TimeoutError("Initial probing leaves insufficient optimization time")
    base = yaml.safe_load((ROOT / f"configs/{FAMILY}_eval.yaml").read_text(encoding="utf-8"))
    if start["kind"] == "sft":
        source_cfg = yaml.safe_load((ROOT / f"configs/{FAMILY}_{decision['selected_candidate']}.yaml").read_text(encoding="utf-8"))
        base["sft"] = source_cfg["sft"]
    base["stage"] = "desktop_rollout_cache_frozen_development"
    base["grpo"].update(pool_size=chosen["candidate_size"],pool_offset=128,
        steps=100000,probe_samples=4,sampling_mechanism="rollout_cache_fifo_v1",
        cache_max_trials=16,cache_ttl_steps=32,max_wall_seconds=usable,
        max_accounted_tokens=10000000,standalone_endpoints_only=True,
        final_validation_reserve_seconds=0,validation_interval=100000,
        endpoint_eval_reserve_seconds=eval_reserve,common_end_to_end_cap_seconds=usable+eval_reserve+30)
    configurations = {}
    for seed in (13,29,47):
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        relative = f"configs/{FAMILY}_grpo_s{seed}.yaml"
        (ROOT / relative).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8",newline="\n")
        configurations[str(seed)] = {"path":relative,"sha256":sha256_file(ROOT / relative)}
    # Replace the old unmeasured 2048-example/NF4 capacity draft with measured local capacity.
    main_cfg = copy.deepcopy(base)
    main_cfg["stage"] = "main_capacity_calibrated_development_only"
    main_cfg["capacity_calibration"] = {"family":FAMILY,"formal_test_protocol_ready":False,
        "hardware":"RX9070XT single-device ROCm10 BF16 nonquantized SDPA",
        "actual_calibration":str((EXP / "calibration_result.json").relative_to(ROOT)),
        "initialization_kind":start["kind"],"common_initial_checkpoint":start["checkpoint"],
        "remaining_formal_requirements":"Review development/confirmation, register final checkpoint plan and independent evaluation budget; holdout remains sealed"}
    (ROOT / "configs/main.yaml").write_text(yaml.safe_dump(main_cfg,sort_keys=False,allow_unicode=True),encoding="utf-8",newline="\n")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen3_17b",local_files_only=True)
    pool = load_split("train","select",limit=128+chosen["candidate_size"],seed=base["data_seed"])[128:]
    excluded = set(base["grpo"]["excluded_train_ids"])
    assert excluded <= {r["id"] for r in pool}
    pool = [r for r in pool if r["id"] not in excluded and len(tokenizer.encode(
        render_prompt(tokenizer,r["question"]),add_special_tokens=False))<=base["generation"]["max_prompt_length"]]
    meta = read(ROOT / start["checkpoint"] / "rrl_checkpoint.json")
    assert not set(meta["training_ids"]) & {r["id"] for r in pool}
    source_files = {}
    for folder in ("src","configs","scripts","tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py",".ps1",".json",".yaml"}:
                source_files[str(path.relative_to(ROOT)).replace("\\","/")] = sha256_file(path)
    json_write(EXP / "frozen_source_manifest.json",source_files)
    plan = {"registered_unix":time.time(),"family":FAMILY,"status":"frozen_development_only",
        "common_start":start,"data_manifest_sha256":sha256_file(ROOT / "data/manifest.json"),
        "development_ids_sha256":sha256_file(EXP / "development_ids.json"),
        "public_model_verification_sha256":sha256_file(EXP / "public_model_verification.json"),
        "configs":configurations,"code_sha256":code_hash(),
        "frozen_source_manifest_sha256":sha256_file(EXP / "frozen_source_manifest.json"),
        "pool_ids":[r["id"] for r in pool],"audited_excluded_train_ids":sorted(excluded),
        "pool_quality":"Four historical flagged items excluded before sampling; other training labels not exhaustively human-reviewed",
        "cost_based_capacity":{"actual60_item_probe_seconds":full_probe,"pool_candidates":pool_candidates,"selected_pool":chosen,
            "session_remaining_seconds_at_freeze":remaining(),"training_cap_candidates":candidates,
            "evaluation_reserve_rule":"ceil(max(all A/S1/S2/S3/S4 comparison128 load+generation seconds)/30)*30+60",
            "evaluation_reserve_seconds":eval_reserve,"training_including_loading_and_sampling_cap_seconds":usable,
            "end_to_end_common_cap_seconds":usable+eval_reserve+30,
            "unused_eval_reserve":"No after-evaluation training; report real underuse and natural-boundary overshoot",
            "confirmation_count":confirmation_n},
        "arms":{"C":"uniform_with_replacement","D":"rollout_cache_fifo_v1_dynamic","E":"initial_probe_static_ablation"},
        "seeds":[13,29,47],"order":[[13,"C"],[13,"D"],[13,"E"],[29,"E"],[29,"D"],[29,"C"],[47,"D"],[47,"C"],[47,"E"]],
        "sampler":{"initial_probe":"Each D/E independently generates K4 for every train-pool question with identical paired-seed forked RNG; actual cost paid in each arm",
            "success_estimator":"p_i=(sum(last up to16 binary rewards)+0.5)/(n_i+1); initial observations=ordered K4 probe",
            "sampling_formula":"q_i=0.25/N +0.75*[4*p_i*(1-p_i)]/sum_j[4*p_j*(1-p_j)]",
            "dynamic_refresh":"After each real training rollout group, append its rewards and keep last16 for that question; no extra generated probe",
            "cache_expiry":"At selection/update time, 32 optimizer steps since last observation clears that question's cache and sets p=0.5; static E never updates or expires",
            "coverage":"Uniform floor 0.25/N plus prior reversion for stale items; finite-budget full coverage is measured, not guaranteed",
            "maintenance_cost":"Age checks, weight computation, state persistence and update logs run inside the common clock; CPU maintenance logged separately without double-counting",
            "initial_weights":"D and E per seed must have byte-equivalent probe predictions and identical initial weights; verification is mandatory"},
        "common_reward_loss_generation_optimizer":"Exact shared config per seed; only sampling mechanism and necessary cost differ",
        "learning_signal_gate":"Passed preregistered C8/D12/E1 diagnostics, not a performance selection",
        "comparison":"Independent fresh-process reload on fixed comparison128 at one fixed endpoint per arm; no interpolation/no checkpoint selection",
        "confirmation":"After all nine checkpoints fixed, one opening of disjoint128 for A, selectedSFT if any, and all nine. No subsequent tuning.",
        "claim_limit":"No new-protocol full-pool dynamic baseline is run, so no claim that cache sampling improves model outcomes over original full-pool dynamic. Historical timing is descriptive only.",
        "stop":"Six hours from session start, reserve900s for handoff; retain every negative/failed/incomplete run",
        "holdout":"GSM8K test and SVAMP sealed"}
    json_write(target,plan)
    print(json.dumps({"training_cap":usable,"eval_reserve":eval_reserve,"pool_n":len(pool),"code_sha256":plan["code_sha256"]},indent=2))


if __name__=="__main__":
    main()
