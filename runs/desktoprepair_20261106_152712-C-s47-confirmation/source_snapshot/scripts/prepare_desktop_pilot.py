"""Choose a bounded desktop development pilot from completed smoke costs, before C/D/E."""
import json
from pathlib import Path
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file

PREFIX = "desktoprocm10pilot_20261105"
smoke = "desktoprocm10bf16_20261105"
status = json.loads((ROOT / f"environment/{smoke}-status.json").read_text(encoding="utf-8-sig"))
assert status["status"] == "completed_smoke_only"
config_path = ROOT / "configs/desktop_rocm10_bf16_pilot.yaml"
plan_path = ROOT / "experiments/desktop_bf16_development_plan_20261105.json"
if config_path.exists() or plan_path.exists():
    raise FileExistsError("Preserve the existing pilot plan/config")
config = yaml.safe_load((ROOT / "configs/desktop_rocm10_bf16_smoke.yaml").read_text(encoding="utf-8"))
config.update(stage="desktop_rocm10_bf16_budget_pilot", hardware_profile="rocm_unverified")
config["generation"]["max_new_tokens"] = 512
config["sft"].update(pool_size=128, steps=32, gradient_accumulation_steps=4, max_length=768)
review_path = ROOT / "experiments/desktop_expanded_train_review_20261105.json"
review = json.loads(review_path.read_text(encoding="utf-8"))
exclusions = [row["id"] for row in review["items"] if row["decision"] == "exclude"]
assert review["retained_count"] == 28 and not review["heldout_accessed"]
config["grpo"].update(pool_size=32, pool_offset=128, excluded_train_ids=exclusions,
    steps=100000, refresh_interval=8, validation_interval=100000,
    max_wall_seconds=420, final_validation_reserve_seconds=120, max_accounted_tokens=1000000)
config["evaluation"]["validation_size"] = 32
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
measurements = {}
for arm in "CDE":
    directory = ROOT / f"runs/{smoke}-{arm}-s13"
    result = json.loads((directory / "training_result.json").read_text(encoding="utf-8"))
    costs = jsonl_read(directory / "costs.jsonl")
    total = next(row for row in costs if row["phase"] == "grpo_total")
    measurements[arm] = {"wall_seconds": total["wall_seconds"], "peak_allocated_bytes": total["peak_allocated_bytes"],
        "peak_reserved_bytes": total["peak_reserved_bytes"], "probe_wall_seconds": sum(row["wall_seconds"] for row in costs if row["phase"] == "difficulty_probe"),
        "adapter_updated": result["adapter_updated"], "zero_variance_fraction": result["zero_variance_group_fraction"],
        "weight_change_events": result["sampler_weight_change_events"], "training_truncation_rate": result["training_truncation_rate"]}
json_write(plan_path, {
    "prefix": PREFIX, "type": "bounded development pilot; not final holdout authorization", "allows_holdout": False,
    "new_AB_validation_size": 32, "grpo_candidate_pool_size": 32, "expected_retained_pool_size": 28,
    "audited_train_exclusions": exclusions, "training_label_audit": str(review_path.relative_to(ROOT)).replace("\\", "/"),
    "training_label_audit_sha256": sha256_file(review_path), "shared_policy_and_probe_budget_seconds": 300,
    "reserve_rule": "30*ceil(B_validation_seconds/30)+30, added to the common 300-second policy/probe allowance before any arm starts",
    "seed13_order": ["uniform", "dynamic", "static"],
    "replication_order": [[29, "dynamic"], [29, "uniform"], [47, "uniform"], [47, "dynamic"]],
    "seeds": [13, 29, 47], "static_seeds": [13],
    "checkpoint_rule": "Keep each budget endpoint without selecting the best development accuracy; every C/D/E starts from the same actual desktop 128-item SFT checkpoint.",
    "stop_rules": ["Stop on native/precision/runtime failure; preserve all partial evidence",
        "Stop replications if all three seed-13 arms have zero adapter update",
        "Report incomplete probes, natural-boundary over/undershoot and actual costs; do not tune from heldout outcomes"],
    "scale_rationale": "Smoke peak allocation below 5 GiB leaves room on 15.92 GiB RX 9070 XT. Increase output cap 256 to 512 to inspect truncation, rebuild A/B at that cap, enlarge disjoint training pool after training-only review. Initial/refreshed probes are charged. Do not claim this 28-item pilot is a full GSM8K study.",
    "smoke_measurements": measurements, "formal_comparison_completed": False,
    "no_main_capacity_or_quality_claim": True
})
print(config_path)
print(plan_path)
