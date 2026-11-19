"""Preregister bounded SFT diagnostics and disjoint development sets BEFORE inference."""
import copy
import datetime as dt
import json
import random
from pathlib import Path
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file, code_hash
from reasoning_rl.data import load_split

FAMILY = "desktoprepair_20261106_152712"
OUT = ROOT / "experiments" / FAMILY


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the registered family")
    OUT.mkdir(parents=True)
    val = load_split("validation", "evaluate_validation", seed=20261105)
    val_ids = {r["id"] for r in val}
    exposure = {i: [] for i in val_ids}
    for path in sorted((ROOT / "runs").glob("**/predictions.jsonl")):
        for row in jsonl_read(path):
            if row.get("id") in val_ids:
                exposure[row["id"]].append(str(path.relative_to(ROOT)))
    old = [r["id"] for r in jsonl_read(ROOT / "runs/desktoprocm10pilot_20261105-A-val/validation/predictions.jsonl")]
    available = sorted(val_ids - set(old), key=lambda i: (bool(exposure[i]), i))
    # Shuffle only within the exposure tiers; no answer or model outcome is consulted.
    rng = random.Random(20261106)
    unseen = [i for i in available if not exposure[i]]
    seen = [i for i in available if exposure[i]]
    rng.shuffle(unseen)
    rng.shuffle(seen)
    available = unseen + seen
    sets = {"comparison": available[:128], "confirmation": available[128:256], "regression": old}
    assert all(len(sets[k]) == n for k, n in (("comparison", 128), ("confirmation", 128), ("regression", 32)))
    assert not set(sets["comparison"]) & set(sets["confirmation"])
    json_write(OUT / "development_ids.json", sets)
    json_write(OUT / "prior_exposure.json", {
        "recorded_before_current_inference": True,
        "search_scope": "all runs/**/predictions.jsonl; historical artifacts absent from this checkout cannot be audited",
        "prior_prediction_files_by_validation_id": {i: sorted(set(p)) for i, p in sorted(exposure.items()) if p},
        "subsets": {k: {"n": len(ids), "previously_predicted_ids": [i for i in ids if exposure[i]]} for k, ids in sets.items()},
        "qualification": "Both new subsets are drawn from an already developer-accessible validation split; neither is a formal independent test."})
    baseline = yaml.safe_load((ROOT / "configs/desktop_rocm10_bf16_pilot.yaml").read_text(encoding="utf-8"))
    baseline["stage"] = "desktop_sft_repair_development"
    baseline["evaluation"].update(validation_size=128,
        validation_ids_path=f"experiments/{FAMILY}/development_ids.json", validation_subset="comparison")
    configs = {}
    # Four candidates, including a faithful optimization-strength reproduction.
    candidates = [("S1", 1e-4, 32), ("S2", 1e-5, 8), ("S3", 1e-5, 32), ("S4", 3e-5, 16)]
    for label, lr, steps in candidates:
        c = copy.deepcopy(baseline)
        c["sft"].update(learning_rate=lr, steps=steps)
        relative = f"configs/{FAMILY}_{label}.yaml"
        (ROOT / relative).write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8", newline="\n")
        configs[label] = relative
    for label, subset, size in (("eval", "comparison", 128), ("confirm", "confirmation", 128), ("regression", "regression", 32)):
        c = copy.deepcopy(baseline)
        c["evaluation"].update(validation_subset=subset, validation_size=size)
        relative = f"configs/{FAMILY}_{label}.yaml"
        (ROOT / relative).write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8", newline="\n")
        configs[label] = relative
    start = dt.datetime(2026, 11, 6, 7, 27, 12, tzinfo=dt.timezone.utc)
    json_write(OUT / "session.json", {"family": FAMILY, "started_utc": start.isoformat(),
        "deadline_utc": (start + dt.timedelta(hours=6)).isoformat(), "max_wall_seconds": 21600,
        "cleanup_reserve_seconds": 900, "holdout_opened": False,
        "initial_local_head": "4e938eb696b1cbb4290c14515d30069ff6e50113",
        "verified_remote_main": "4e938eb696b1cbb4290c14515d30069ff6e50113", "initial_worktree_clean": True})
    json_write(OUT / "sft_preregistration.json", {
        "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "family": FAMILY,
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
        "development_ids_sha256": sha256_file(OUT / "development_ids.json"),
        "configs": configs, "candidates": [{"name": x, "learning_rate": lr, "steps": st,
            "pool_size": 128, "gradient_accumulation_steps": 4} for x, lr, st in candidates],
        "limit": "At most these four SFT candidates; no additional SFT tuning in this session",
        "checks_before_training": ["actual tokenizer boundaries and full gold", "causal label shift", "prompt and padding masks",
            "EOS and chat template", "base/fresh-untrained-LoRA/saved-reloaded logits and predictions"],
        "selection_rule": {
            "eligible": "On fixed comparison128: correct >= A_correct - 2; truncations <= A_truncations + 2; parse_errors <= A_parse_errors + 2; all gradients finite and adapter updated",
            "rank": "Max correct, then fewer truncations, then lower learning_rate*steps, then candidate order",
            "route": "Select highest eligible real SFT. If none is eligible, use explicitly verified PUBLIC_POSTTRAINED_A with fresh zero-output LoRA as GRPO initialization, never label it SFT.",
            "no_forced_improvement": True},
        "old32": "regression diagnosis only, never candidate selection",
        "confirmation128": "sealed until the sampling protocol and all nine C/D/E endpoint checkpoints are fixed; one stage opening for A, selected SFT if any, and all nine endpoints; no subsequent tuning",
        "sampling_next": "Measure full-pool probe and real rollout before freezing one rollout-cache dynamic mainline, C/D/E seeds13,29,47. No claim against historical full-pool D without a new same-budget baseline.",
        "code_sha256_at_registration": code_hash(), "test_and_svamp": "sealed"})
    print(json.dumps({"family": FAMILY, "subsets": {k: len(v) for k, v in sets.items()},
                      "previously_exposed": {k: sum(bool(exposure[i]) for i in v) for k, v in sets.items()}}, indent=2))


if __name__ == "__main__":
    main()
