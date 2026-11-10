"""Read-only verification of frozen inputs, paid endpoint costs and mechanism drift."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import time

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, checkpoint_hash, sha256_file
from reasoning_rl.data import load_split
from reasoning_rl.scoring import group_advantages
sys.path.insert(0, str(ROOT / "scripts"))
from filter_common import EXP, FAMILY, START, START_HASH, read
from run_filter_controlled import verify

OUT = ROOT / "reports/comparisons" / FAMILY


def window(rs):
    groups=list(rs)
    n=len(groups)
    rewards=[r["reward"] for group in groups for r in group]
    mixed=sum(0 < sum(r["reward"] for r in group) < 4 for group in groups)
    effective=sum(sum(a != 0 and not r["truncated"] for a,r in zip(group_advantages([r["reward"] for r in group]),group,strict=True)) for group in groups)
    return {"groups":n,"mixed_groups":mixed,"mixed_group_fraction":mixed/n if n else None,
        "zero_variance_group_fraction":1-mixed/n if n else None,
        "training_stochastic_success_fraction":sum(rewards)/len(rewards) if rewards else None,
        "effective_advantage_completions":effective}


def main():
    plan=read(EXP / "protocol_frozen.json")
    verify(plan)
    prereg=read(EXP / "candidate_preregistration.json")
    train=load_split("train","select",seed=20261105)
    val=load_split("validation","evaluate_validation",seed=20261105)
    candidates=set(plan["candidate_pool_ids"])
    sft=set(prereg["nominal_sft_pool_ids"])
    dev={r["id"] for r in val}
    assert candidates <= {r["id"] for r in train}
    assert not candidates & (sft | dev | set(prereg["audited_excluded_train_ids"]))
    pairs=jsonl_read(ROOT/"data/audit/near_pairs.jsonl")
    crossing=[p for p in pairs if (p["a"] in candidates and p["b"] in sft|dev) or (p["b"] in candidates and p["a"] in sft|dev)]
    assert not crossing
    manifest=read(ROOT/"data/manifest.json")
    holdout_hashes={name:sha256_file(ROOT/manifest["splits"][name]["path"]) for name in ("test","external")}
    assert all(holdout_hashes[name] == manifest["splits"][name]["sha256"] for name in holdout_hashes)
    probe=read(EXP/"prefilter_manifest.json")
    assert checkpoint_hash(ROOT/START) == START_HASH
    status=read(ROOT/"environment"/(FAMILY+"-status.json"))
    assert status["status"] == "completed_three_paired_seed_development"
    costs=read(EXP/"endpoint_costs.json")
    process=jsonl_read(EXP/"process_costs.jsonl")
    commands=jsonl_read(EXP/"commands.jsonl")
    assert all(r["exit_code"] == 0 for r in process)
    assert all(abs(r["finished_unix"]-r["started_unix"]-r["process_wall_seconds"]) < .01 for r in process)
    fixed=read(EXP/"all_training_endpoints_fixed.json")
    eval_starts=[r["started_unix"] for r in commands if r["phase"].endswith("-development") and r["phase"] != "S2-development"]
    assert fixed["fixed_unix"] <= min(eval_starts)
    drift=[]
    excluded_signal=[]
    checkpoints=[]
    optimizer_checks=[]
    for label,c in costs.items():
        directory=ROOT/"runs"/c["training_run"]
        contract=read(directory/"comparison_contract.json")
        chosen=plan["candidate_pool_ids"] if c["arm"] == "U" else plan["retained_ids"]
        selections=jsonl_read(directory/"selections.jsonl")
        for r in selections:
            assert list(r["weights"]) == chosen
            assert all(w == 1/len(chosen) for w in r["weights"].values())
            assert set(r["ids"]) <= set(chosen) and len(r["ids"]) == 1
            assert r["sampler_updates"] == 0
        groups=defaultdict(list)
        for r in jsonl_read(directory/"rollouts.jsonl"):
            groups[r["generation_batch"]].append(r)
        ordered=[groups[i] for i in sorted(groups)]
        size=max(1,len(ordered)//4)
        drift.append({"label":label,"quarter_groups":size,"overall":window(ordered),"first_quarter":window(ordered[:size]),"last_quarter":window(ordered[-size:]),
            "scope":"Within-endpoint descriptive first/last group quarters, not a matched-budget validation curve"})
        by_initial=defaultdict(list)
        for rs in ordered:
            raw=probe["raw_success_counts"][rs[0]["id"]]
            by_initial["probe_mixed_retained" if raw in (1,2,3) else "probe_all_correct_excluded" if raw == 4 else "probe_all_wrong_excluded"].append(rs)
        excluded_signal.append({"label":label, "buckets":{k:window(v) for k,v in by_initial.items()},
            "scope":"Actual policy trajectory only. Mixed rewards on excluded items show finite-K screening is not an oracle; they are not counterfactual F outcomes."})
        ledger=jsonl_read(directory/"costs.jsonl")
        grads=[r for r in ledger if r["phase"] == "gradient_diagnostic"]
        updates=[r for r in ledger if r["phase"] == "optimizer_weight_update"]
        signals=jsonl_read(directory/"rollout_group_signal.jsonl")
        disagreements=[]
        for g,s,u in zip(grads,signals,updates,strict=True):
            effective=any(a != 0 and n > 0 for a,n in zip(s["actual_advantages"],s["actual_loss_mask_tokens_per_completion"],strict=True))
            if effective != (g["nonzero_gradient_elements"] > 0):
                disagreements.append(g["optimizer_step"])
            assert g["finite_gradients"] and not g["frozen_parameter_gradients"]
            assert math.isfinite(u["actual_weight_delta_l2"])
        optimizer_checks.append({"label":label,"actual_steps":len(grads),"effective_signal_vs_nonzero_gradient_disagreements":disagreements,
            "zero_gradient_steps_with_weight_change":sum(u["zero_gradient_step"] and u["actual_changed_weight_elements"] > 0 for u in updates)})
        checkpoint=ROOT/"checkpoints"/c["training_run"]
        digest=checkpoint_hash(checkpoint)
        assert digest == c["checkpoint_sha256"]
        optimizer_path=checkpoint/"optimizer_diagnostic_state.pt"
        assert optimizer_path.exists()
        checkpoints.append({"label":label,"checkpoint":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":digest,
            "optimizer_diagnostic_state_sha256":sha256_file(optimizer_path),"optimizer_diagnostic_state_bytes":optimizer_path.stat().st_size,
            "exact_resume_supported":False})
        eval_dir=ROOT/"runs"/(c["training_run"]+"-development")
        eval_protocol=read(eval_dir/"evaluation_protocol.json")
        assert eval_protocol["adapter_sha256"] == digest and eval_protocol["split"] == "validation"
        assert read(eval_dir/"resolved_evaluation_ids.json")["ids"] == read(EXP/"development_ids.json")["comparison"]
        expected_debit=probe["probe_process_wall_seconds"] if c["arm"] == "F" else 0
        assert c["initial_probe_standalone_debit_seconds"] == expected_debit
        paid=sum(c[k] for k in ("training_process_seconds","initial_probe_standalone_debit_seconds",
            "parent_preparation_and_training_audit_seconds","evaluation_process_seconds","parent_evaluation_preparation_and_audit_seconds"))
        assert abs(paid-c["charged_end_to_end_seconds"]) < .001
        assert paid <= c["common_cap_seconds"]
    session=read(EXP/"session.json")
    assert status["finished_unix"] <= session["deadline_unix"]
    upper_before_first=process[0]["started_unix"]-session["started_unix"]
    model_process_seconds=sum(r["process_wall_seconds"] for r in process)
    interruption=read(EXP/"interruption_receipt.json") if (EXP/"interruption_receipt.json").exists() else None
    failed_upper=interruption["failed_attempt_conservative_upper_bound_seconds"] if interruption else 0
    failed_lower=interruption["failed_attempt_observed_minimum_seconds"] if interruption else 0
    conservative_model_upper=model_process_seconds+upper_before_first+failed_upper
    assert conservative_model_upper <= session["max_wall_seconds"]
    token_parts={
        "S2_calibration_development_evaluation":read(ROOT/"runs"/(FAMILY+"-S2-development")/"validation/metrics.json")["generated_tokens"],
        "calibration_K4_probe16":read(ROOT/"runs"/(FAMILY+"-calibration-probe")/"probe_result.json")["generated_tokens"],
        "calibration_GRPO8":read(ROOT/"runs"/(FAMILY+"-calibration-train")/"training_result.json")["generated_tokens"],
        "shared_K4_probe64_once":read(ROOT/"runs"/(FAMILY+"-probe")/"probe_result.json")["generated_tokens"],
        "six_completed_main_training_processes":sum(read(ROOT/"runs"/c["training_run"]/"training_result.json")["generated_tokens"] for c in costs.values()),
        "six_independent_main_evaluations":sum(read(ROOT/"runs"/(c["training_run"]+"-development")/"validation/metrics.json")["generated_tokens"] for c in costs.values()),
        "interrupted_attempt_observed_outputs":interruption.get("observed_failed_attempt_generated_tokens",0) if interruption else 0}
    quality=[]
    flagged="gsm8k-train-06831"
    for label,c in costs.items():
        rs=jsonl_read(ROOT/"runs"/c["training_run"]/"rollouts.jsonl")
        by_group=defaultdict(list)
        for r in rs:
            by_group[r["generation_batch"]].append(r)
        affected=[v for v in by_group.values() if v[0]["id"] == flagged]
        other=[v for v in by_group.values() if v[0]["id"] != flagged]
        quality.append({"label":label,"flagged_id":flagged,"flagged_item":window(affected),
            "remaining_items_postfreeze_descriptive_only":window(other),
            "primary_protocol_and_all_outcomes_still_include_flagged_item":True})
    receipt={"family":FAMILY,"verified_unix":time.time(),"all_checks_passed":True,
        "frozen_execution_source_unchanged":True,"candidate_pool_n":len(candidates),"filtered_pool_n":len(plan["retained_ids"]),
        "candidate_near_pairs_crossing_sft_or_validation":crossing,"train_val_and_sft_overlap":False,
        "holdout_validation":"Only immutable bytes hashed, no questions/answers loaded or model evaluation performed",
        "holdout_sha256":holdout_hashes,"test_and_svamp_model_evaluation":False,
        "full_shared_probe_physical_seconds":probe["probe_process_wall_seconds"],
        "sum_F_probe_debits_seconds":3*probe["probe_process_wall_seconds"],
        "wrapped_model_process_seconds":model_process_seconds,
        "physical_generated_tokens_observed_parts":token_parts,
        "physical_generated_tokens_observed_lower_bound":sum(token_parts.values()),
        "unlogged_inflight_generation_tokens_upper_bound":interruption.get("unlogged_generation_token_upper_bound_from_one_G4_group",0) if interruption else 0,
        "physical_generation_token_scope":"No duplicated shared-probe debit tokens; interrupted final in-flight group may be unlogged, so do not label the observed sum an exact whole-session total.",
        "unwrapped_preflight_and_preparation_upper_bound_seconds":upper_before_first,
        "interrupted_attempt_observed_lower_seconds":failed_lower,
        "interrupted_attempt_conservative_upper_seconds":failed_upper,
        "conservative_model_related_process_upper_bound_seconds":conservative_model_upper,
        "session_wall_through_last_model_seconds":status["finished_unix"]-session["started_unix"],
        "session_max_seconds":session["max_wall_seconds"],"all_main_checkpoints_fixed_before_evaluation":True,
        "runtime_artifact_directories_including_interrupted":len(list((ROOT/"runs").glob(FAMILY+"-*"))),
        "interrupted_runtime_directories":1 if interruption else 0,
        "optimizer_checks":optimizer_checks,"checkpoints":checkpoints,
        "physical_vs_charged_cost_note":"Do not sum the three F standalone probe debits into physical one-time probe cost. Nested Ledger stages are not added to process wall time."}
    json_write(OUT/"signal_drift.json",drift)
    json_write(OUT/"screened_out_training_signal.json",excluded_signal)
    json_write(OUT/"checkpoint_manifest.json",checkpoints)
    json_write(OUT/"flagged_training_item_signal.json",quality)
    json_write(OUT/"evidence_audit.json",receipt)
    print(json.dumps({"all_checks_passed":True,"model_related_upper_bound_seconds":conservative_model_upper,
        "session_wall_through_model_seconds":receipt["session_wall_through_last_model_seconds"],"optimizer_checks":optimizer_checks},indent=2))


if __name__=="__main__":
    main()
