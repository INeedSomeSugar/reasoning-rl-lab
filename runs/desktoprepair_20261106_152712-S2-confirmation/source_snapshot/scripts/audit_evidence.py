"""Inspect saved evidence without running a model or opening held-out question content."""
import hashlib
import json
import math
from pathlib import Path
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, checkpoint_hash, sha256_file
from reasoning_rl.data import load_split
from reasoning_rl.scoring import score_answer


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def snapshot_hash(root):
    h = hashlib.sha256()
    for directory in ("src", "configs"):
        for p in sorted((root/directory).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".yaml", ".json"}:
                h.update(p.relative_to(root).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()


train_ids = {r["id"] for r in load_split("train", "develop")}
records = []
errors = []
unavailable_checkpoints = []
for directory in sorted((ROOT/"runs").glob("*")):
    if not directory.is_dir() or not (directory/"provenance.json").exists():
        continue
    record = {"run": directory.name}
    provenance = read_json(directory/"provenance.json")
    record["source_snapshot_matches"] = snapshot_hash(directory/"source_snapshot") == provenance["code_sha256"]
    if not record["source_snapshot_matches"]:
        errors.append(directory.name+": source snapshot mismatch")
    result_path = directory/"training_result.json"
    if not result_path.exists():
        record["status"] = "not_a_completed_training_run"
        records.append(record)
        continue
    result = read_json(result_path)
    record.update(status="completed_training_run", adapter_updated=result.get("adapter_updated"))
    # Saved metadata may contain the laptop's absolute Windows path after a Git clone to Linux.
    checkpoint_name = result["checkpoint"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    checkpoint = ROOT/"checkpoints"/checkpoint_name
    if checkpoint.is_dir():
        record["checkpoint_hash_matches"] = checkpoint_hash(checkpoint) == result["checkpoint_sha256"]
        if not record["checkpoint_hash_matches"]:
            errors.append(directory.name+": checkpoint mismatch")
    else:
        record["checkpoint_hash_matches"] = None
        record["checkpoint_status"] = "weights unavailable in this checkout; metadata is not a weight verification"
        unavailable_checkpoints.append(directory.name)
    if (directory/"comparison_contract.json").exists():
        contract = read_json(directory/"comparison_contract.json")
        record["pool_is_train_only"] = set(contract["pool_ids"]) <= train_ids
        selections = jsonl_read(directory/"selections.jsonl")
        record["selections_in_pool"] = all(i in contract["pool_ids"] for row in selections for i in row["ids"])
        rollouts = jsonl_read(directory/"rollouts.jsonl")
        generated = sum(r["generated_tokens"] for r in rollouts)
        record["rollout_tokens_reconcile"] = generated == result["generated_tokens"]
        record["reward_sum_reconciles"] = sum(r["reward"] for r in rollouts) == result["reward_sum"]
        record["reward_recomputation_matches"] = all(float(score_answer(r["prediction"], r["gold_answer"]).correct) == r["reward"] for r in rollouts)
        groups = {}
        for row in rollouts:
            key = row["generation_batch"], row.get("group_within_batch", 0)
            groups.setdefault(key, []).append(row)
        record["groups_are_single_question"] = all(len({r["id"] for r in g}) == 1 for g in groups.values())
        zero = sum(len({r["reward"] for r in group}) == 1 for group in groups.values())
        record["zero_variance_groups_reconcile"] = zero == result["zero_variance_groups"]
        if (directory/"sampler_final.json").exists():
            sampler = read_json(directory/"sampler_final.json")
            weights = sampler.get("weights")
            if weights:
                record["sampler_weights_normalized"] = math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
                record["uniform_floor_preserved"] = min(weights.values()) >= sampler["uniform_mix"]/len(weights)-1e-12
        costs = jsonl_read(directory/"costs.jsonl")
        gradients = [c for c in costs if c["phase"] == "gradient_diagnostic"]
        record["gradients_finite"] = all(c["finite_gradients"] and not c["frozen_parameter_gradients"] for c in gradients)
        record["nonzero_gradient_steps"] = sum(c["nonzero_gradient_elements"] > 0 for c in gradients)
        for key, value in record.items():
            if isinstance(value, bool) and key != "adapter_updated" and not value:
                errors.append(directory.name+": "+key)
    records.append(record)

output = {"status": "fail" if errors else "pass_with_unavailable_checkpoints" if unavailable_checkpoints else "pass",
          "errors": errors, "unavailable_checkpoints": unavailable_checkpoints, "runs": records,
          "holdout_content_read": False,
          "final_claims_present": any((ROOT/"experiments/final_claims").glob("*.json")),
          "scope": "File integrity, split membership, raw reward/token/gradient reconciliation; not a scientific benefit claim"}
json_write(ROOT/"reports/evidence_audit.json", output)
print(json.dumps({"status": output["status"], "runs_checked": len(records), "errors": errors}, indent=2))
if errors:
    raise SystemExit(1)
