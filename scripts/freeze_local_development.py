"""Calibrate the declared local development budget, then freeze it before C/D/E.

Run only after the A, SFT, and B evaluation processes have exited. This is not
authorization to evaluate the official test set or external transfer set.
"""
import argparse
import math
import re
import time
from pathlib import Path

import yaml

from reasoning_rl.runtime import ROOT, checkpoint_hash, code_hash, json_write, jsonl_read, sha256_file
from reasoning_rl.data import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--development-plan", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--sft-evaluation", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    output = ROOT / args.output
    if output.exists():
        raise FileExistsError("Development freeze already exists; preserve it")
    started = [p.name for p in (ROOT / "runs").glob(args.prefix + "-*-s*") if p.is_dir()]
    if started:
        raise ValueError(f"C/D/E runs already exist; no budget change allowed: {started}")
    plan = read(ROOT / args.development_plan)
    if plan.get("allows_holdout") is not False or plan["prefix"] != args.prefix:
        raise ValueError("Use a matching development-only plan")
    config_path = ROOT / args.config
    original = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(original)
    a_dir, b_dir = ROOT / "runs" / args.baseline, ROOT / "runs" / args.sft_evaluation
    a_metric, b_metric = read(a_dir / "validation/metrics.json"), read(b_dir / "validation/metrics.json")
    a_ids = [r["id"] for r in jsonl_read(a_dir / "validation/predictions.jsonl")]
    b_ids = [r["id"] for r in jsonl_read(b_dir / "validation/predictions.jsonl")]
    if a_ids != b_ids or len(b_ids) != plan["new_AB_validation_size"]:
        raise ValueError("A/B must use the declared identical development questions")
    for directory, metric in ((a_dir, a_metric), (b_dir, b_metric)):
        if metric["n"] != len(b_ids) or metric["phase"] != "validation":
            raise ValueError("Incomplete or non-validation endpoint")
        if read(directory / "config.json") != config:
            raise ValueError("A/B config must match before budget calibration")
    checkpoint = ROOT / args.sft_checkpoint
    checkpoint_sha = checkpoint_hash(checkpoint)
    if read(b_dir / "evaluation_protocol.json")["adapter_sha256"] != checkpoint_sha:
        raise ValueError("B evaluation does not use the specified real SFT checkpoint")
    b_costs = jsonl_read(b_dir / "costs.jsonl")
    b_seconds = sum(row["wall_seconds"] for row in b_costs if row["phase"] == "validation")
    if b_seconds <= 0:
        raise ValueError("Missing measured B validation cost")
    reserve = 30 * math.ceil(b_seconds / 30) + 30
    usable = plan["shared_policy_and_probe_budget_seconds"]
    replacements = {"max_wall_seconds": usable + reserve, "final_validation_reserve_seconds": reserve}
    for key, value in replacements.items():
        original, matches = re.subn(rf"(?m)^  {key}:.*$", f"  {key}: {value}", original)
        if matches != 1:
            raise ValueError(f"Expected one GRPO budget field: {key}")
    calibrated = yaml.safe_load(original)
    for key, value in replacements.items():
        if calibrated["grpo"][key] != value:
            raise ValueError("Budget replacement failed")
    offset = config["grpo"].get("pool_offset", 0)
    candidate = load_split("train", "select", limit=offset+config["grpo"]["pool_size"], seed=config["data_seed"])[offset:]
    candidate_ids = [row["id"] for row in candidate]
    excluded = config["grpo"].get("excluded_train_ids", [])
    if excluded != plan["audited_train_exclusions"] or len(candidate_ids) != plan["grpo_candidate_pool_size"]:
        raise ValueError("Predeclared training pool audit differs")
    actual_pool = [item for item in candidate_ids if item not in excluded]
    if len(actual_pool) != plan["expected_retained_pool_size"]:
        raise ValueError("Unexpected retained training pool size")
    sft_meta = read(checkpoint / "rrl_checkpoint.json")
    if sft_meta["stage"] != "sft" or set(actual_pool) & set(sft_meta["training_ids"]):
        raise ValueError("Require a real SFT checkpoint and disjoint training pools")
    config_path.write_text(original, encoding="utf-8", newline="\n")
    result = {"type": "frozen development protocol, not final holdout authorization",
              "created_unix": time.time(), "prefix": args.prefix, "config": calibrated,
              "code_sha256": code_hash(), "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
              "sft_checkpoint_sha256": checkpoint_sha, "sft_checkpoint": args.sft_checkpoint,
              "grpo_candidate_pool_ids": candidate_ids, "grpo_pool_ids": actual_pool,
              "audited_train_exclusions": excluded, "training_label_audit": plan.get("training_label_audit", "reports/training_label_audit.json"),
              "sft_grpo_disjoint": True, "seeds": [13, 29, 47],
              "first_run_order": [[13, strategy] for strategy in plan["seed13_order"]],
              "replication_order": plan["replication_order"],
              "stop_if_first_seed_all_three_have_no_parameter_update": True,
              "stop_rules": plan["stop_rules"], "baseline_A_validation": args.baseline,
              "sft_B_validation": args.sft_evaluation, "validation_ids": b_ids,
              "calibration": {"B_validation_wall_seconds": b_seconds,
                              "reserve_rule": "30*ceil(B_validation_seconds/30)+30",
                              "final_validation_reserve_seconds": reserve,
                              "policy_plus_probe_budget_seconds": usable,
                              "grpo_wall_cap_seconds": usable + reserve},
              "selection_rule": plan["checkpoint_rule"],
              "allows_holdout": False, "final_comparison_completed": False}
    json_write(output, result)
    print(f"Frozen development protocol: {output}")
    print(f"B validation {b_seconds:.3f}s; reserve {reserve}s; GRPO cap {usable+reserve}s")


if __name__ == "__main__":
    main()
