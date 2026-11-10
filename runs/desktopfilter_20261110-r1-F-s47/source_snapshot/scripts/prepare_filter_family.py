"""Preregister cost-only calibration and the maximum common train candidate pool."""
import datetime as dt
import json
import subprocess
import time
import yaml
from transformers import AutoTokenizer
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file, checkpoint_hash
from reasoning_rl.data import load_split
from reasoning_rl.modeling import render_prompt
from filter_common import EXP, FAMILY, OLD, START, START_HASH, utc_now


def main():
    EXP.mkdir(parents=True, exist_ok=False)
    now = time.time()
    json_write(EXP / "session.json", {"family": FAMILY, "started_utc": utc_now(),
        "started_unix": now, "deadline_unix": now + 14400, "max_wall_seconds": 14400,
        "time_scope": "Conservative whole-session limit from preregistration, including code/calibration/model failures/evaluation; model process seconds also summed separately.",
        "initial_local_head": "5246f0c8a2b74331dc5927cb31516c38bb0c967e",
        "inherited_remote_main": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "initial_worktree_clean": True, "holdout_opened": False})
    config = yaml.safe_load((ROOT / "configs/desktoprepair_20261106_152712_grpo_s13.yaml").read_text())
    assert checkpoint_hash(ROOT / START) == START_HASH
    train = load_split("train", "select", seed=config["data_seed"])
    val = load_split("validation", "evaluate_validation", seed=config["data_seed"])
    sft_ids = {r["id"] for r in train[:128]}  # exclude nominal SFT pool, not only 32 actually supervised
    val_ids = {r["id"] for r in val}
    excluded = set(config["grpo"]["excluded_train_ids"])
    blocked = sft_ids | val_ids | excluded
    pairs = jsonl_read(ROOT / "data/audit/near_pairs.jsonl")
    while True:
        previous = set(blocked)
        for p in pairs:
            if p["a"] in blocked or p["b"] in blocked:
                blocked.update((p["a"], p["b"]))
        if previous == blocked:
            break
    tok = AutoTokenizer.from_pretrained(ROOT / "models/qwen3_17b", local_files_only=True)
    candidates, too_long = [], []
    for row in train[128:]:
        if row["id"] in blocked:
            continue
        if len(tok.encode(render_prompt(tok, row["question"]), add_special_tokens=False)) > config["generation"]["max_prompt_length"]:
            too_long.append(row["id"])
            continue
        candidates.append(row["id"])
        if len(candidates) == 256:
            break
    assert len(candidates) == 256
    json_write(EXP / "candidate_preregistration.json", {"ids": candidates,
        "rule": "Fixed preexisting train shuffle seed20261105 after nominal SFT128; exclude flagged items and near-duplicate connected components touching SFT/validation/flags; drop overlong prompts. Formal pool is a cost-selected prefix, never reward/dev-selected.",
        "nominal_sft_pool_ids": sorted(sft_ids), "audited_excluded_train_ids": sorted(excluded),
        "additional_connected_component_exclusions": sorted(blocked - sft_ids - val_ids - excluded),
        "overlong_skipped": too_long, "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
        "near_pairs_sha256": sha256_file(ROOT / "data/audit/near_pairs.jsonl"),
        "quality_limit": "Historical flagged items excluded; remaining reference solutions not exhaustively human reviewed."})
    dev = json.loads((OLD / "development_ids.json").read_text())["confirmation"]
    json_write(EXP / "development_ids.json", {"comparison": dev,
        "scope": "Previously used confirmation128, now explicitly REUSED DEVELOPMENT data. No independent-confirmation or test claim."})
    config["evaluation"].update(validation_ids_path=f"experiments/{FAMILY}/development_ids.json",
        validation_subset="comparison", validation_size=128)
    config["grpo"].update(audit_optimizer_deltas=True, audit_loss_signal=True)
    config["stage"] = "prefilter_cost_calibration_development_only"
    path = ROOT / "configs" / f"{FAMILY}_calibration.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    json_write(EXP / "calibration_preregistration.json", {"registered_utc": utc_now(),
        "seeds": [13, 29, 47], "probe_seed": 20261110, "probe_k": 4,
        "filter_rule": "Retain iff RAW integer successes in {1,2,3} out of4, no smoothed probabilities.",
        "calibration_probe_ids": candidates[:16], "calibration_train_ids": candidates[:32],
        "pool_capacity_choices_descending": [256, 192, 128, 96, 64],
        "maximum_probe_fraction_of_training_allocation": 0.45,
        "training_allocation_choices_descending_seconds": [1500, 1200, 900, 600],
        "evaluation_reserve_rule": "ceil(actual fresh-process S2 confirmation128 evaluation seconds/30)*30+90",
        "capacity_rule": "Largest listed train allocation leaving 600s whole-session slack for six endpoints, one probe and freezing; then largest prefix whose conservative probe estimate <=45% of train allocation.",
        "probe_estimate_rule": "max(1.25*calibration16 probe seconds per item, historical60 probe seconds per item)*N + measured calibration process setup/saving overhead",
        "calibration_train_optimizer_steps": 8,
        "prefilter_cache_policy": "One shared fixed-S2, fixed-probe-seed full probe physically run once. Charge FULL probe process cost to EACH F endpoint for primary standalone comparison. Also show physical one-time and amortized costs separately. Every F pays its own cache validation/read/filter preparation inside its train process.",
        "empty_filtered_pool": "Abort U/F main comparison and preserve failed mechanism evidence, no alternative seed/pool by reward.",
        "no_tuning_by_development_outcomes": True, "holdout": "GSM8K test/SVAMP sealed"})
    print(json.dumps({"family": FAMILY, "maximum_candidates": len(candidates), "development": "reused confirmation128", "s2": START_HASH}))


if __name__ == "__main__":
    main()
