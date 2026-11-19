"""Time-only B/E/C and complete35-unit protocol freeze after three actual probes."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import yaml

from reasoning_rl.runtime import ROOT, checkpoint_hash, code_hash, sha256_file, json_write, jsonl_read
from reasoning_rl.prefiltering import matched_random_subset, raw_success_filter
from large_filter_common import START, START_HASH, PROBE_SEEDS, RANDOM_SEEDS, read, write_new, utc_now, matrix_order


def freeze(family):
    exp = ROOT / "experiments" / family
    if (exp / "protocol_frozen.json").exists():
        raise FileExistsError("Preserve the existing frozen family")
    data = read(exp / "data_frozen.json")
    prereg = read(exp / "preregistration.json")
    probes = read(exp / "all_probes_complete.json")["probes"]
    candidates = read(exp / "candidate_pool.json")["ids"]
    if any((p / "fresh_development/predictions.jsonl").exists() for p in (ROOT / "runs").glob(family+"*")):
        raise ValueError("New development already opened before this main freeze")
    assert checkpoint_hash(ROOT / START) == START_HASH
    manifests, filtered_sets, random_sets = {}, {}, {}
    for seed, random_seed in zip(PROBE_SEEDS, RANDOM_SEEDS, strict=True):
        item = probes[str(seed)]
        run_dir = ROOT / "runs" / item["run"]
        predictions_path = run_dir / "predictions.jsonl"
        retained, counts = raw_success_filter(candidates, jsonl_read(predictions_path), 4)
        if not retained:
            write_new(exp / "degenerate_empty_filter_stop.json", {"probe_seed": seed, "all_probes": probes,
                "reason": "Pre-registered empty-set rule; no main training or alternate K/seed/pool permitted"})
            raise RuntimeError("A registered F block is empty; main matrix is unavailable")
        assert sha256_file(predictions_path) == item["predictions_sha256"]
        filter_path = exp / f"prefilter_j{seed}.json"
        write_new(filter_path, {"candidate_pool_sha256": sha256_file(exp / "candidate_pool.json"),
            "s2_checkpoint_sha256": START_HASH, "probe_seed": seed, "k": 4,
            "predictions_path": predictions_path.relative_to(ROOT).as_posix(), "predictions_sha256": sha256_file(predictions_path),
            "retained_ids": retained, "raw_success_counts": counts, "raw_success_histogram": dict(Counter(counts.values())),
            "probe_process_wall_seconds": item["P_seconds"], "F_endpoint_count": 5,
            "eligibility": "RAW successes in{1,2,3}; no smoothing, target size, or performance tuning"})
        started = time.perf_counter()
        random_ids = matched_random_subset(candidates, len(retained), random_seed)
        draw_seconds = time.perf_counter()-started
        random_path = exp / f"random_j{seed}.json"
        write_new(random_path, {"ids": random_ids, "random_subset_seed": random_seed,
            "candidate_pool_sha256": sha256_file(exp / "candidate_pool.json"),
            "matched_prefilter_manifest_path": filter_path.relative_to(ROOT).as_posix(),
            "matched_prefilter_manifest_sha256": sha256_file(filter_path), "M": len(retained),
            "physical_subset_draw_seconds": draw_seconds, "overlap_with_F": len(set(random_ids)&set(retained)),
            "draw_scope": "Uniform without replacement from FULL256, not F complement; no R performance observed",
            "probe_matching_debit_seconds": item["P_seconds"], "physical_probe_required_by_R": False})
        manifests[str(seed)] = {"F_path": filter_path.relative_to(ROOT).as_posix(), "F_sha256": sha256_file(filter_path),
            "R_path": random_path.relative_to(ROOT).as_posix(), "R_sha256": sha256_file(random_path), "M": len(retained)}
        filtered_sets[str(seed)], random_sets[str(seed)] = set(retained), set(random_ids)
    selection_diagnostics = {"F_jaccard": {}, "per_item_F_selection_frequency": {
        i: sum(i in s for s in filtered_sets.values()) for i in candidates},
        "R_overlap_with_F": {j: len(random_sets[j]&filtered_sets[j]) for j in filtered_sets}}
    for a in filtered_sets:
        for b in filtered_sets:
            selection_diagnostics["F_jaccard"][a+"/"+b] = len(filtered_sets[a]&filtered_sets[b])/len(filtered_sets[a]|filtered_sets[b])
    write_new(exp / "selection_diagnostics.json", selection_diagnostics)
    B = max(3600, 300*math.ceil(3*max(x["P_seconds"] for x in probes.values())/300))
    calibration = read(exp / "calibration_complete.json")
    N = data["development_n"]
    eval_estimate = calibration["old64_evaluation_process_seconds"]*N/64
    # Prospective reliability amendment, recorded before main freeze/fresh outcomes.
    amendment = read(exp / "evaluation_reserve_amendment_before_main.json")
    assert amendment["new_development_scores_observed"] is False
    E = 300*math.ceil((2.0*eval_estimate+300)/300)
    O = 300
    budget = {"B_seconds": B, "E_seconds": E, "O_seconds": O, "C_seconds": B+E+O,
        "P_seconds": {j: p["P_seconds"] for j, p in probes.items()},
        "eval_old64_process_seconds": calibration["old64_evaluation_process_seconds"],
        "eval_newN_linear_estimate_seconds": eval_estimate,
        "evaluation_reserve_unused_never_refills_training": True,
        "evaluation_reserve_amendment_sha256": sha256_file(exp / "evaluation_reserve_amendment_before_main.json"),
        "basis": "Only complete probe durations and historical-development timing; no fresh-development scores exist"}
    cfg = yaml.safe_load((ROOT / "configs" / f"{family}_calibration.yaml").read_text())
    cfg["stage"] = "larger_clean_pool_fixed_prefilter_random_replication_frozen_development"
    cfg["grpo"].update(pool_size=len(candidates), steps=100000, excluded_train_ids=prereg["known_questionable_ids"],
        max_wall_seconds=B, max_accounted_tokens=1000000000, candidate_pool_path=f"experiments/{family}/candidate_pool.json",
        final_validation_reserve_seconds=0, endpoint_eval_reserve_seconds=E, common_end_to_end_cap_seconds=B+E+O,
        sampling_mechanism="uniform_frozen_pool_v1", standalone_endpoints_only=True,
        audit_loss_signal=True, audit_optimizer_deltas=True, save_optimizer_state=True)
    cfg["grpo"]["account_outer_process_start"] = True
    cfg["evaluation"] = {"fresh_development_manifest": data["fresh_manifest_path"],
        "fresh_development_manifest_sha256": data["fresh_manifest_sha256"], "fresh_development_size": N,
        "validation_size": 64, "test_size": None, "external_size": None}
    # The evaluation config is common to every independent process, including A/S2.
    eval_path = ROOT / "configs" / f"{family}_evaluation.yaml"
    eval_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    configs = {}
    for unit in matrix_order():
        options = json.loads(json.dumps(cfg))
        options["seed"] = unit["seed"]
        options["grpo"]["experiment_arm"] = unit["arm"]
        if unit["arm"] != "U":
            key = "prefilter_manifest_path" if unit["arm"] == "F" else "random_subset_manifest_path"
            options["grpo"][key] = manifests[str(unit["probe_seed"])][unit["arm"]+"_path"]
        path = ROOT / "configs" / (family+"_"+unit["key"]+".yaml")
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8", newline="\n")
        configs[unit["key"]] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
    order = read(exp / "registered_matrix_order.json")
    assert order["order"] == matrix_order()
    preparation_elapsed = time.time()-read(exp / "session.json")["registered_unix"]
    primary_train_upper = sum(B-(0 if u["arm"] == "U" else probes[str(u["probe_seed"])]["P_seconds"])+O/2 for u in matrix_order())
    primary_eval_upper = 37*(E+O/2)
    retry_reserve = primary_train_upper+primary_eval_upper
    additional_margin = 0.2*(primary_train_upper+primary_eval_upper+retry_reserve)
    analysis_reserve = 6*3600
    total_hard = math.ceil((preparation_elapsed+primary_train_upper+primary_eval_upper+retry_reserve+additional_margin+analysis_reserve)/300)*300
    session_limit = {"frozen_utc": utc_now(), "registered_unix": read(exp / "session.json")["registered_unix"],
        "preparation_calibration_probes_elapsed_wall_seconds": preparation_elapsed,
        "remaining_primary_training_process_upper_seconds": primary_train_upper,
        "remaining_primary_evaluation_process_upper_seconds": primary_eval_upper,
        "failure_reserve_one_full_attempt_for_every_training_and_evaluation_unit_seconds": retry_reserve,
        "scheduling_and_audit_margin_seconds": additional_margin, "analysis_reserve_seconds": analysis_reserve,
        "total_session_hard_seconds": total_hard,
        "deadline_unix": read(exp / "session.json")["registered_unix"]+total_hard,
        "likely_remaining_model_seconds_timing_estimate": sum(B-(0 if u["arm"] == "U" else probes[str(u["probe_seed"])]["P_seconds"]) for u in matrix_order())+37*eval_estimate,
        "not_permission_for_extra_experiments": True, "each_unit_max_attempts": 2,
        "budget_cannot_be_extended_after_fresh_evaluation": True}
    write_new(exp / "session_limit_frozen.json", session_limit)
    frozen_files = {}
    for folder in ("src", "configs", "scripts", "tests", "requirements"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".yaml", ".json", ".txt", ".ps1"}:
                rel = p.relative_to(ROOT).as_posix()
                frozen_files[rel] = sha256_file(p)
                destination = exp / "frozen_source" / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, destination)
    for relative in ("pyproject.toml", "scripts/desktop_env.ps1", "environment/requirements-desktop-rocm10.lock.txt"):
        p = ROOT / relative
        frozen_files[relative] = sha256_file(p)
        destination = exp / "frozen_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    write_new(exp / "frozen_source_manifest.json", frozen_files)
    artifact_paths = ["preregistration.json", "evaluation_reserve_amendment_before_main.json", "remote_baseline_checked.json", "history_audit.json", "history_sources.jsonl", "history_exposure_ledger.jsonl",
        "historical_exclusions.jsonl", "clusters.jsonl", "candidate_order.json", "candidate_review_pack.jsonl", "candidate_review_completed.jsonl",
        "candidate_pool.json", "data_frozen.json", "registered_matrix_order.json", "all_probes_complete.json",
        "selection_diagnostics.json", "session_limit_frozen.json", "frozen_source_manifest.json",
        "preflight.json", "calibration_complete.json", "probe_inputs_frozen.json", "history_restoration.jsonl",
        "near_pairs_train_only.jsonl", "offline_training_audit_smoke.json", "full_process_clock_smoke.json"]
    artifact_paths += [f"{a}_j{j}.json" for j in PROBE_SEEDS for a in ("prefilter", "random")]
    protocol = {"family": family, "frozen_utc": utc_now(), "status": "frozen_before_any_main_training_or_new_development_evaluation",
        "base_commit": read(exp / "session.json")["initial_head"], "common_start": START, "common_start_sha256": START_HASH,
        "model": read(ROOT / "configs/models.lock.json")["qwen3_17b"] if "qwen3_17b" in read(ROOT / "configs/models.lock.json") else read(ROOT / "configs/models.lock.json"),
        "budget": budget, "session_limit": session_limit, "candidate_n": len(candidates), "development_n": N,
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"), "fresh_manifest_path": data["fresh_manifest_path"],
        "reservation_registry_sha256": sha256_file(ROOT / "data/fresh_development/registry.json"),
        "fresh_manifest_sha256": data["fresh_manifest_sha256"], "subsets": manifests, "configs": configs,
        "evaluation_config": {"path": eval_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(eval_path)},
        "order": order["order"], "evaluation_order": order["evaluation_order"], "code_sha256": code_hash(),
        "artifacts": {p: sha256_file(exp / p) for p in artifact_paths},
        "fixed_variables": {"precision": "nonquantized BF16/SDPA", "lora": cfg["lora"], "generation": cfg["generation"],
            "GRPO": {k: cfg["grpo"][k] for k in ("learning_rate", "num_generations", "gradient_accumulation_steps", "loss_type", "scale_rewards", "beta", "epsilon")},
            "prompt": "math_numeric_v2", "scorer": "numeric_final_v2", "enable_thinking": False,
            "unchanged_existing_behaviors": "token importance ratio; one iteration; truncated completion mask; AdamW/constant scheduler; zero-gradient Adam steps retained"},
        "statistics": prereg["statistics"], "decisions": prereg["decisions"], "blind_review": prereg["blind_review"],
        "R_matching": "Deduct full P_j for equal trainable time; not an inherent physical random-selection cost; no claim about an unrun longer-training R",
        "training_clock": "B starts at controller preparation before full source/model/tokenizer cache verification. Recorded outer UNIX start is passed to the child and converted once into its monotonic budget clock, including interpreter imports/loading/training/saving. Post-checkpoint parent audit and natural-boundary overshoot are charged and reported under C.",
        "U_reuse": "Only five independent U endpoints; reused in the15 paired F-U cells and in resampling dependencies",
        "missingness": "Preserve failed attempts; maxone full retry fromS2/config; first success. Freeze all35 terminal statuses before fresh evaluation; report all missing cells, never silently delete.",
        "holdout": "GSM8K test/SVAMP sealed. Fresh development is isolated official train, project-record unseen only; public pretraining contamination unknown."}
    write_new(exp / "protocol_frozen.json", protocol)
    print(json.dumps({"family": family, "B": B, "E": E, "O": O, "C": B+E+O,
        "M": {j: m["M"] for j, m in manifests.items()}, "likely_remaining_model_hours": session_limit["likely_remaining_model_seconds_timing_estimate"]/3600,
        "session_hard_hours": total_hard/3600}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    freeze(p.parse_args().family)
