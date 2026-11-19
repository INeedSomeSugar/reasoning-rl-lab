"""Independent closeout invariants, source/data/model hashes and a portable artifact index."""
import argparse
from collections import Counter
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file, checkpoint_hash
from reasoning_rl.data import load_split
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, START, START_HASH, TRAIN_SEEDS, PROBE_SEEDS
from run_large_filter import verify_frozen, expected_pool
from large_filter_cost_accounting import accounting


def verify_artifact_manifest(family):
    path = ROOT / "experiments" / family / "artifact_manifest.json"
    manifest = read(path)
    assert manifest["family"] == family
    assert manifest["formal_holdout_content_excluded"] and manifest["self_excluded_to_avoid_hash_cycle"]
    files = manifest["files"]
    assert files and path.relative_to(ROOT).as_posix() not in files
    total_bytes = 0
    for name, expected in files.items():
        artifact = ROOT / name
        assert artifact.is_file(), name
        assert artifact.stat().st_size == expected["bytes"], name
        assert sha256_file(artifact) == expected["sha256"], name
        total_bytes += expected["bytes"]
    print({"family": family, "verified_files": len(files), "verified_bytes": total_bytes,
        "manifest_sha256": sha256_file(path)})


def check_pair_results(out, summary, plan, table, bootstrap):
    n = summary["n"]
    indices = {key: i for i, key in enumerate(bootstrap["keys"])}
    joint = bootstrap["joint_accuracies"]
    incomplete = summary.get("available_pair_comparisons")
    with (out / "paired_differences.csv").open(encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))
    assert [(int(r["probe_seed"]), int(r["training_seed"])) for r in pairs] == [(j, s) for j in PROBE_SEEDS for s in TRAIN_SEEDS]
    for arm in ("U", "R", "S2", "A"):
        name = f"F-{arm}"
        pp_field = f"F_minus_{arm}_pp"
        observed, columns = [], []
        weights = Counter()
        for pair in pairs:
            j, seed = pair["probe_seed"], pair["training_seed"]
            f_key = f"F-j{j}-s{seed}"
            c_key = f"R-j{j}-s{seed}" if arm == "R" else f"U-s{seed}" if arm == "U" else arm
            weights.update({f_key: 1, c_key: -1})
            f, c = table[f_key]["correct"], table[c_key]["correct"]
            if f is None or c is None:
                assert pair[pp_field] == ""
            else:
                value = 100*(f-c)/n
                assert math.isclose(float(pair[pp_field]), value, abs_tol=1e-10)
                observed.append(value)
                columns.append(joint[:, indices[f_key]]-joint[:, indices[c_key]])
        if incomplete:
            results = incomplete[name]
            assert len(observed) == results["available"]["pairs"]
            assert (results["original_point_estimate_pp"] is None) == (len(observed) < 15)
            known = sum(weight*table[key]["correct"] for key, weight in weights.items() if table[key]["correct"] is not None)
            missing = {k: v for k, v in weights.items() if table[k]["correct"] is None}
            sensitivity = results["full_matrix_sensitivity"]
            assert sensitivity["missing_endpoint_cell_weights"] == missing
            expected = [(known + n*sum(op(0, coef) for coef in missing.values()))*100/(15*n) for op in (min, max)]
            assert np.allclose(expected, sensitivity["bounds_pp"], atol=1e-12, rtol=0)
            if observed:
                assert math.isclose(sum(observed)/len(observed), results["available"]["mean_pp"], abs_tol=1e-10)
                quantiles = np.quantile(np.asarray(columns).mean(axis=0)*100, [0.025, 0.975])
                assert np.allclose(quantiles, results["available"]["conditional_percentile95_pp"], atol=1e-10, rtol=0)
        else:
            assert len(observed) == 15
            assert math.isclose(sum(observed)/15, summary[f"F_minus_{arm}_mean_pp"], abs_tol=1e-10)


def main(family, final=False):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    assert checkpoint_hash(ROOT / START) == START_HASH
    manifest = read(ROOT / "data/manifest.json")
    train_source = ROOT / "data/raw/gsm_train.jsonl"
    train_source_sha = sha256_file(train_source)
    assert train_source_sha == manifest["source_sha256"]["gsm_train"]
    raw_train = jsonl_read(train_source)
    assert len(raw_train) == manifest["official_train_count"]
    verified_ids = set()
    for split in ("train", "validation"):
        split_path = ROOT / manifest["splits"][split]["path"]
        assert sha256_file(split_path) == manifest["splits"][split]["sha256"]
        for row in jsonl_read(split_path):
            index = int(row["id"].rsplit("-", 1)[1])
            assert row["id"] == f"gsm8k-train-{index:05d}" and row["id"] not in verified_ids
            source_row = raw_train[index]
            assert row["question"] == source_row["question"] and row["solution"] == source_row["answer"], row["id"]
            verified_ids.add(row["id"])
    assert len(verified_ids) + manifest["quarantined_train_count"] == len(raw_train)
    fresh = read(ROOT / plan["fresh_manifest_path"])
    development = jsonl_read(ROOT / fresh["rows_path"])
    assert sha256_file(ROOT / fresh["rows_path"]) == fresh["rows_sha256"]
    assert len(development) == fresh["n"] == plan["development_n"]
    train = jsonl_read(ROOT / manifest["splits"]["train"]["path"])
    original = {r["id"]: r for r in train}
    for row in development:
        for key in ("question", "solution", "answer", "question_sha256", "official_split", "source"):
            assert row[key] == original[row["id"]][key]
        assert row["split"] == "fresh_development" and row["original_project_split"] == "train"
    pool = read(exp / "candidate_pool.json")["ids"]
    exclusions = {r["id"] for r in jsonl_read(exp / "historical_exclusions.jsonl")}
    assert not set(fresh["ids"]) & (set(pool)|exclusions)
    assert not set(pool) & exclusions
    assert not set(fresh["ids"]) & {r["id"] for r in load_split("train", "select")}
    clusters = jsonl_read(exp / "clusters.jsonl")
    id_cluster = {i: c["cluster_id"] for c in clusters for i in c["ids"]}
    members = {c["cluster_id"]: set(c["ids"]) for c in clusters}
    assert len({id_cluster[i] for i in pool}) == len(pool) == 256
    assert not {id_cluster[i] for i in pool} & {id_cluster[i] for i in fresh["ids"]}
    for cluster in fresh["clusters"]:
        assert members[cluster] <= set(fresh["ids"])
    lock = read(exp / "all_training_endpoints_fixed.json")
    opened = read(exp / "fresh_development_opened.json")
    assert sha256_file(exp / "all_training_endpoints_fixed.json") == opened["endpoint_lock_sha256"]
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    for key, obj in lock["objects"].items():
        if obj["checkpoint"]:
            assert checkpoint_hash(ROOT / obj["checkpoint"]) == obj["checkpoint_sha256"], key
    processes = jsonl_read(exp / "process_costs.jsonl")
    train_labels = {a["label"] for item in lock["training_units"].values() for a in item["attempts"]}
    evaluation = [p for p in processes if p["label"].startswith("eval-")]
    trained = [p for p in processes if p["label"] in train_labels]
    freeze_time = dt.datetime.fromisoformat(plan["frozen_utc"]).timestamp()
    lock_time = dt.datetime.fromisoformat(lock["frozen_utc"]).timestamp()
    eval_states = read(exp / "evaluation_units.json")
    physical_recheck, _, _, interruptions = accounting(exp, lock["training_units"], eval_states)
    assert all(freeze_time < r["started_unix"] < r["next_OS_boot_event_unix"] <= lock_time for r in interruptions)
    assert min(p["started_unix"] for p in trained) > freeze_time
    assert max(p["finished_unix"] for p in trained) <= lock_time
    assert min(p["started_unix"] for p in evaluation) > lock_time
    assert not [p for p in (ROOT / "runs").glob(family+"*") if (p / "test/predictions.jsonl").exists() or (p / "external/predictions.jsonl").exists()]
    common = None
    from reasoning_rl.prefiltering import resolve_frozen_pool
    available = load_split("train", "select", seed=20261105)
    for unit in plan["order"]:
        cfg = yaml.safe_load((ROOT / plan["configs"][unit["key"]]["path"]).read_text())
        resolved, _, debit, evidence = resolve_frozen_pool(cfg["grpo"], available, START_HASH)
        assert [r["id"] for r in resolved] == expected_pool(family, unit)
        expected_debit = 0 if unit["arm"] == "U" else plan["budget"]["P_seconds"][str(unit["probe_seed"])]
        assert debit == expected_debit
        comparable = json.loads(json.dumps(cfg))
        comparable.pop("seed")
        for field in ("experiment_arm", "prefilter_manifest_path", "random_subset_manifest_path"):
            comparable["grpo"].pop(field, None)
        if common is None:
            common = comparable
        assert comparable == common
    assert Counter(u["arm"] for u in plan["order"]) == {"U": 5, "F": 15, "R": 15}
    summary = read(out / "summary.json")
    for field in ("interrupted_process_count", "all_attempt_model_process_seconds_lower_bound",
                  "all_attempt_model_process_seconds_upper_bound", "failed_model_process_seconds_lower_bound",
                  "failed_model_process_seconds_upper_bound"):
        assert math.isclose(summary["physical_costs"][field], physical_recheck[field], abs_tol=1e-8)
    table = {r["endpoint"]: r for r in summary["endpoints"]}
    missing = summary["missing_evaluation_objects"]
    assert len(table) == len(eval_states) == 37
    complete_keys = [k for k in plan["evaluation_order"] if eval_states[k]["status"] == "first_success_fixed"]
    assert summary["successful_evaluation"] == len(complete_keys) and set(missing) == set(table)-set(complete_keys)
    for key, row in eval_states.items():
        assert len(row["attempts"]) <= 2 and row["status"] == table[key]["evaluation_status"]
        if key in missing:
            assert table[key]["correct"] is None and table[key]["charged_end_to_end_seconds"] is None
            if row["status"] == "exhausted_failure":
                assert len(row["attempts"]) == 2 and all(a["status"].startswith("failed") for a in row["attempts"])
        else:
            p = ROOT / "runs" / row["run_name"] / "fresh_development/predictions.jsonl"
            assert sha256_file(p) == row["predictions_sha256"]
            raw = jsonl_read(p)
            assert [r["id"] for r in raw] == fresh["ids"]
            assert sum(r["correct"] for r in raw) == table[key]["correct"] and table[key]["evaluated_n"] == len(raw)
    if missing:
        assert summary["original_full_matrix_inference_available"] is False
        assert summary["development_evidence_threshold_met"] is None and summary["engineering_signal_threshold_met"] is None
    with np.load(out / "joint_cluster_bootstrap.npz", allow_pickle=False) as bootstrap:
        replicates = plan["statistics"]["conditional_cluster_bootstrap_replicates"]
        assert bootstrap["joint_accuracies"].shape == (replicates, len(complete_keys))
        assert bootstrap["keys"].tolist() == complete_keys
        check_pair_results(out, summary, plan, table, bootstrap)
    review = read(out / "blind_review_summary.json")
    assert review["pending_cases"] == 0 and review["human_reviewed"] is False
    assert review["annotation_sha256_fixed_before_unblinding"] == sha256_file(exp / "all_blind_annotations_fixed_before_unblinding.jsonl")
    assert review["exact_text_deduplication"]["unique_response_units_actually_annotated"] > 0
    broken = []
    for markdown in (out / "findings.md", ROOT / "docs/REPRODUCING_FILTER_REPLICATION_20261110.md"):
        for match in re.finditer(r"\]\(([^\n)]+)\)", markdown.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("https://", "http://", "#", "codex://")):
                continue
            path = (markdown.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                broken.append({"source": str(markdown.relative_to(ROOT)), "target": target})
    assert not broken, broken
    figures = ["endpoint_accuracies", "paired_differences", "conditional_intervals", "signal_efficiency", "coverage_concentration", "signal_time_windows", "cost_breakdown", "probe_selection"]
    assert all((out / (f+".png")).exists() and (out / (f+".svg")).exists() for f in figures)
    visual_audit = exp / "figure_visual_audit_20261119.json"
    if final:
        visual_receipt = read(visual_audit)
        assert visual_receipt["reviewer_kind"] == "Agent"
        assert set(visual_receipt["figures"]) == set(figures)
        for name, record in visual_receipt["figures"].items():
            assert record["png_sha256"] == sha256_file(out / (name+".png"))
            assert record["svg_sha256"] == sha256_file(out / (name+".svg"))
    receipt = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "family": family,
        "frozen_execution_source_verified": True, "original_data_and_S2_unchanged": True,
        "official_raw_train_sha256": train_source_sha, "raw_train_projection_verified_rows": len(verified_ids),
        "candidate_and_fresh_clusters_historically_disjoint": True, "ordinary_train_excludes_reserved_development": True,
        "all35_endpoint_attempts_before_fresh_evaluation": True, "only_intervention_and_seed_config_differences": True,
        "U_distinct_trained_endpoints": 5, "F_distinct_trained_endpoints": 15, "R_distinct_trained_endpoints": 15,
        "training_seed_blocks": 5, "probe_selection_blocks": 3, "candidate_pools": 1,
        "U_reused_across_probe_blocks": True, "F_R_endpoints_not15_independent_selection_draws": True,
        "registered_evaluation_objects": 37, "successful_evaluation_objects": len(complete_keys),
        "available_objects_same_actual_N": fresh["n"], "missing_evaluation_objects_preserved": missing,
        "full_matrix_inference_available": not missing, "joint_bootstrap_replicates": replicates,
        "pair_means_missing_bounds_and_conditional_intervals_independently_recomputed": True,
        "registered_blind_scope_complete": review.get("complete_registered_scope", True),
        "report_and_reproduction_links_checked": True, "figure_files_checked": len(figures),
        "visual_inspection_receipt": visual_audit.relative_to(ROOT).as_posix() if final else None,
        "interrupted_attempt_costs_included_as_bounds": len(interruptions), "unknown_exit_codes_not_imputed": True,
        "formal_holdout_model_predictions_created": False, "remote_published": False}
    json_write(exp / "closeout_invariants.json", receipt)
    if final:
        session = read(exp / "session.json")
        selected_checkpoints = {ROOT / u["checkpoint"]: key for key, u in lock["training_units"].items()
            if u["status"] == "first_success_fixed"}
        diagnostic_suffixes = {"preflight-a0", "calibration-train-a0", "accounting-clock-smoke-a0"}
        checkpoint_roles = [{"path": START, "role": "common_real_S2_anchor_and_all_training_initialization",
            "checkpoint_sha256": START_HASH, "primary_training_endpoint": False}]
        for directory in sorted((ROOT / "checkpoints").glob(family+"*")):
            if not directory.is_dir():
                continue
            suffix = directory.name[len(family)+1:]
            role = "primary_first_success_training_endpoint" if directory in selected_checkpoints else (
                "old_exposed_data_preparation_diagnostic" if suffix in diagnostic_suffixes
                else "preserved_nonselected_training_attempt")
            checkpoint_roles.append({"path": directory.relative_to(ROOT).as_posix(), "role": role,
                "endpoint": selected_checkpoints.get(directory), "primary_training_endpoint": directory in selected_checkpoints,
                "files": sorted(p.name for p in directory.iterdir() if p.is_file())})
        assert sum(r["primary_training_endpoint"] for r in checkpoint_roles) == 35
        json_write(exp / "checkpoint_roles.json", {"family": family, "checkpoints": checkpoint_roles,
            "public_anchor_A": {"revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "files": read(exp / "preflight.json")["public_model_files"]},
            "diagnostic_and_failed_attempts_are_not_additional_primary_endpoints": True})
        json_write(exp / "session_final.json", {"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "registered_unix": session["registered_unix"], "finished_unix": time.time(),
            "elapsed_registered_session_wall_seconds": time.time()-session["registered_unix"],
            "hard_limit_seconds": plan["session_limit"]["total_session_hard_seconds"],
            "measured_model_process_seconds": sum(p["process_wall_seconds"] for p in processes),
            "measured_model_process_seconds_scope": "Exact completed receipts only; additional interrupted costs in all-attempt bounds",
            "all_attempt_model_process_seconds_lower_bound": physical_recheck["all_attempt_model_process_seconds_lower_bound"],
            "all_attempt_model_process_seconds_upper_bound": physical_recheck["all_attempt_model_process_seconds_upper_bound"],
            "failed_model_process_seconds_lower_bound": physical_recheck["failed_model_process_seconds_lower_bound"],
            "failed_model_process_seconds_upper_bound": physical_recheck["failed_model_process_seconds_upper_bound"],
            "finished_before_frozen_deadline": time.time() <= plan["session_limit"]["deadline_unix"],
            "unmeasured_initial_orientation_before_registration": session["initial_orientation_cost"],
            "model_tasks": read(exp / "all_model_tasks_complete.json"), "available_blinded_Agent_review_complete": True,
            "registered_blind_scope_complete": review.get("complete_registered_scope", True),
            "all_requested_local_artifacts_completed": not missing,
            "local_evidence_delivered_with_missing_evaluations": bool(missing),
            "formal_holdouts_opened": False, "remote_published": False})
        paths = set()
        for directory in (exp, out, ROOT / "data/fresh_development" / family):
            paths.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for directory in (ROOT / "runs", ROOT / "checkpoints"):
            for sub in directory.glob(family+"*"):
                paths.update(p for p in sub.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for directory in ("src", "scripts", "tests", "analysis", "requirements"):
            paths.update(p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".txt", ".ps1"})
        paths.update(p for p in (ROOT / "configs").glob(family+"*"))
        paths.update(p for p in (ROOT / "logs").glob(family+"*"))
        paths.update(p for p in (ROOT / "environment").glob(family+"*"))
        paths.update(ROOT / name for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md", "pyproject.toml", "data/manifest.json",
            "data/audit/near_pairs.jsonl", "data/audit/summary.json", "data/audit/quarantine.json", "data/raw/sources.lock.json", "data/raw/gsm_train.jsonl",
            "data/processed/train.jsonl", "data/processed/validation.jsonl", "configs/models.lock.json", "configs/data.sources.lock.json",
            "data/fresh_development/registry.json", "docs/INDEX.md", "docs/REPRODUCING_FILTER_REPLICATION_20261110.md"))
        paths.update(ROOT / item["path"] for item in read(exp / "preflight.json")["public_model_files"])
        paths.update(p for p in (ROOT / START).iterdir() if p.is_file())
        target = exp / "artifact_manifest.json"
        paths.discard(target)
        files = {p.relative_to(ROOT).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(paths)}
        json_write(target, {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "family": family,
            "files": files, "formal_holdout_content_excluded": True, "self_excluded_to_avoid_hash_cycle": True,
            "checkpoint_hash_scope": "Protocol checkpoint digest covers adapter_config/model/rrl_checkpoint; every optimizer/RNG diagnostic file has an additional per-file SHA256 here."})
    print(receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--verify-only", action="store_true", help="Read-only verification of the recorded final SHA256 file list")
    args = p.parse_args()
    if args.verify_only:
        verify_artifact_manifest(args.family)
    else:
        main(args.family, args.final)
