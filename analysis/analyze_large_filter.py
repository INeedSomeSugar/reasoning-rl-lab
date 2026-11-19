"""Offline independent evidence audit and joint-cluster statistics; no model calls."""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_write, sha256_file, checkpoint_hash
from reasoning_rl.scoring import score_answer, group_advantages
from reasoning_rl.replication_statistics import paired_difference_matrix, joint_cluster_accuracies, intervals_from_joint
from reasoning_rl.prefiltering import raw_success_filter, matched_random_subset

sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import START, START_HASH, TRAIN_SEEDS, PROBE_SEEDS, read
from large_filter_cost_accounting import accounting, endpoint_failure_columns, operational_rate_bounds
from large_filter_missing_statistics import differences, signal_rates, transitions_from_s2


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audited_training(unit, original_s2):
    directory = ROOT / "runs" / unit["run_name"]
    result = read(directory / "training_result.json")
    contract = read(directory / "comparison_contract.json")
    assert contract["initial_checkpoint_sha256"] == START_HASH
    precision = read(directory / "model_precision.json")
    for phase in ("loader", "trainer"):
        snapshot = precision[phase]
        assert snapshot["is_loaded_in_4bit"] is False and snapshot["attention_implementation"] == "sdpa"
        assert snapshot["input_embedding"] == snapshot["output_head"] == "torch.bfloat16"
        assert snapshot["parameter_devices"] == ["cuda:0"]
    pool = contract["pool_ids"]
    groups = defaultdict(list)
    selections = jsonl_read(directory / "selections.jsonl")
    counts = Counter({k: 0 for k in ("all_correct_groups", "all_wrong_groups", "mixed_groups",
        "effective_completions", "nonzero_advantage_completions", "parse_failures", "format_failures", "truncations")})
    by_item, effective_by_item = Counter(), Counter()
    for selection in selections:
        assert selection["sampler_updates"] == 0 and len(selection["ids"]) == 1
        assert set(selection["weights"]) == set(pool)
        assert all(abs(w-1/len(pool)) < 1e-12 for w in selection["weights"].values())
        by_item.update(selection["ids"])
    for row in jsonl_read(directory / "rollouts.jsonl"):
        score = score_answer(row["prediction"], row["gold_answer"]).as_dict()
        assert all(row[k] == v for k, v in score.items())
        assert row["reward"] == float(row["correct"])
        assert len(row["completion_token_ids"]) == row["generated_tokens"]
        assert row["id"] in pool
        groups[row["generation_batch"]].append(row)
        counts["parse_failures"] += int(not row["parse_ok"])
        counts["format_failures"] += int(not row["format_ok"])
        counts["truncations"] += int(row["truncated"])
    signals = {r["generation_batch"]: r for r in jsonl_read(directory / "rollout_group_signal.jsonl")}
    group_effective, group_mixed, group_totals, group_generated, group_truncated = {}, {}, {}, {}, {}
    for batch, rows in groups.items():
        assert len(rows) == 4 and len({r["id"] for r in rows}) == 1
        rewards = [r["reward"] for r in rows]
        total = sum(rewards)
        counts["all_correct_groups" if total == 4 else "all_wrong_groups" if total == 0 else "mixed_groups"] += 1
        actual = signals[batch]
        expected = group_advantages(rewards)
        assert actual["mask_and_advantage_checks_passed"] and actual["raw_rewards"] == rewards
        assert all(abs(a-b) < 2e-6 for a, b in zip(actual["actual_advantages"], expected, strict=True))
        masks = [0 if r["truncated"] else r["generated_tokens"] for r in rows]
        assert actual["actual_loss_mask_tokens_per_completion"] == masks
        counts["nonzero_advantage_completions"] += sum(a != 0 for a in expected)
        eff = sum(a != 0 and mask > 0 for a, mask in zip(expected, masks, strict=True))
        counts["effective_completions"] += eff
        effective_by_item[rows[0]["id"]] += eff
        group_effective[batch] = eff
        group_mixed[batch] = int(0 < total < 4)
        group_totals[batch] = int(total)
        group_generated[batch] = sum(r["generated_tokens"] for r in rows)
        group_truncated[batch] = sum(r["truncated"] for r in rows)
    ledger = jsonl_read(directory / "costs.jsonl")
    grads = [r for r in ledger if r["phase"] == "gradient_diagnostic"]
    updates = [r for r in ledger if r["phase"] == "optimizer_weight_update"]
    steps = result["completed_optimizer_steps"]
    assert len(groups) == len(signals) == len(selections) == len(grads) == len(updates) == steps
    assert all(g["finite_gradients"] and not g["frozen_parameter_gradients"] for g in grads)
    assert counts["effective_completions"] == result["actual_effective_advantage_completions"]
    assert counts["nonzero_advantage_completions"] == result["actual_nonzero_advantage_completions"]
    loss = jsonl_read(directory / "loss_signal.jsonl")
    assert len(loss) == 4*steps
    loss_by_step = defaultdict(list)
    for row in loss:
        loss_by_step[row["optimizer_step_zero_based"]].append(row)
    for step, entries in loss_by_step.items():
        assert len(entries) == 4
        sig = signals[step]
        actual = sorted((round(a, 6), int(m)) for e in entries for a, m in zip(e["advantages"], e["completion_loss_tokens"], strict=True))
        expected = sorted((round(a, 6), int(m)) for a, m in zip(sig["actual_advantages"], sig["actual_loss_mask_tokens_per_completion"], strict=True))
        assert actual == expected
    counts["nonzero_gradient_steps"] = sum(g["nonzero_gradient_elements"] > 0 for g in grads)
    counts["zero_gradient_adam_changed_steps"] = sum(u["zero_gradient_step"] and u["actual_changed_weight_elements"] > 0 for u in updates)
    counts["actual_parameter_changed_steps"] = sum(u["actual_changed_weight_elements"] > 0 for u in updates)
    for g, u in zip(grads, updates, strict=True):
        assert g["optimizer_step"] == u["optimizer_step"]
        assert u["zero_gradient_step"] == (g["nonzero_gradient_elements"] == 0)
    checkpoint = ROOT / unit["checkpoint"]
    assert checkpoint_hash(checkpoint) == unit["checkpoint_sha256"] == result["checkpoint_sha256"]
    from safetensors.torch import load_file
    import torch
    actual = load_file(checkpoint / "adapter_model.safetensors")
    delta = sum(float(((actual[k].float()-original_s2[k].float())**2).sum()) for k in actual)**0.5
    assert math.isclose(delta, result["adapter_delta_l2"], rel_tol=1e-5, abs_tol=1e-7)
    state = torch.load(checkpoint / "optimizer_diagnostic_state.pt", map_location="cpu", weights_only=True)
    assert state["exact_resume_supported"] is False
    states = list(state["optimizer"]["state"].values())
    assert states
    for item in states:
        for value in item.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
        assert float(item["step"]) == steps
    progress = [r for r in ledger if r["phase"] == "budget_progress"]
    total_time = max(r["cumulative_training_wall_seconds"] for r in progress)
    windows = []
    for index, label in enumerate(("first", "middle", "last")):
        selected = [r for r in progress if min(2, int(r["cumulative_training_wall_seconds"]/max(total_time, 1e-9)*3)) == index]
        batches = [r["optimizer_step"]-1 for r in selected]
        denominator = len(batches)
        windows.append({"window": label, "groups": denominator,
            "all_correct_groups": sum(group_totals[b] == 4 for b in batches),
            "all_wrong_groups": sum(group_totals[b] == 0 for b in batches),
            "mixed_groups": sum(group_mixed[b] for b in batches), "effective_completions": sum(group_effective[b] for b in batches),
            "mixed_group_fraction": sum(group_mixed[b] for b in batches)/denominator if denominator else None,
            "nonzero_gradient_steps": sum(grads[b]["nonzero_gradient_elements"] > 0 for b in batches),
            "generated_tokens": sum(group_generated[b] for b in batches),
            "generated_completions": 4*denominator,
            "mean_generated_tokens_per_completion": sum(group_generated[b] for b in batches)/(4*denominator) if denominator else None,
            "truncated_completions": sum(group_truncated[b] for b in batches),
            "window_definition": "Three equal observed-training-time windows, assigned by optimizer completion time"})
    assert sum(w["groups"] for w in windows) == steps
    assert all(w["all_correct_groups"]+w["all_wrong_groups"]+w["mixed_groups"] == w["groups"] for w in windows)
    assert sum(w["generated_tokens"] for w in windows) == result["generated_tokens"]
    assert sum(w["truncated_completions"] for w in windows) == counts["truncations"]
    effective_total = sum(effective_by_item.values())
    coverage = [{"id": i, "selections": by_item[i], "effective_completions": effective_by_item[i]} for i in pool]
    summary = {"optimizer_steps": steps, "pool_n": len(pool), **dict(counts),
        "zero_variance_fraction": (counts["all_correct_groups"]+counts["all_wrong_groups"])/steps,
        "distinct_training_ids": len(by_item), "coverage_fraction": len(by_item)/len(pool),
        "repetitions_min_including_zero": min(by_item[i] for i in pool), "repetitions_max": max(by_item.values()),
        "effective_top10_question_share": sum(sorted(effective_by_item.values(), reverse=True)[:10])/effective_total if effective_total else None,
        "adapter_delta_l2_independent": delta, "adam_state_entries": len(states), "adam_state_steps": steps,
        "optimizer_state_sha256": sha256_file(checkpoint / "optimizer_diagnostic_state.pt"),
        "peak_allocated_bytes": max(r.get("peak_allocated_bytes") or 0 for r in ledger),
        "peak_reserved_bytes": max(r.get("peak_reserved_bytes") or 0 for r in ledger),
        "generated_tokens": result["generated_tokens"], "prompt_tokens": result["generation_prompt_tokens"],
        "policy_loss_input_tokens": result["processed_tokens"], "wall_budget_overshoot_seconds": result["wall_budget_overshoot_seconds"]}
    return summary, windows, coverage


def blind_pack(exp, plan, predictions, development, endpoints, missing_plan=None):
    pack_path = exp / "blind_review_pack.jsonl"
    if pack_path.exists():
        return
    rng = random.Random(plan["blind_review"]["seed"])
    cases, identities = [], []
    unavailable = []
    rows = {r["id"]: r for r in development}
    seed29_pairs = [(f"F-j{j}-s29", c, "registered_seed29", rng) for j in PROBE_SEEDS
        for c in ("U-s29", f"R-j{j}-s29", "S2")]
    pairs = list(seed29_pairs)
    if missing_plan:
        supplement = missing_plan["review"]["supplement"]
        left, right = supplement["left"], supplement["right"]
        same_driver = endpoints[left]["evaluation_driver_version"] and endpoints[left]["evaluation_driver_version"] == endpoints[right]["evaluation_driver_version"]
        if "U-s29" not in predictions and left in predictions and right in predictions and same_driver:
            pairs.append((left, right, "exploratory_same_driver_seed13", random.Random(f"{supplement['seed']}:exploratory-same-driver")))
    for f, comparator, review_scope, pair_rng in pairs:
        missing = [key for key in (f, comparator) if key not in predictions]
        if missing:
            assert missing_plan is not None
            unavailable.append({"comparison": f+"/"+comparator, "missing_endpoints": missing})
            continue
        left, right = predictions[f], predictions[comparator]
        strata = {"discordant": [], "common_correct": [], "common_wrong": []}
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            strata["discordant" if a["correct"] != b["correct"] else "common_correct" if a["correct"] else "common_wrong"].append(i)
        chosen = [(i, "discordant") for i in strata["discordant"]]
        for name in ("common_correct", "common_wrong"):
            chosen.extend((i, name) for i in pair_rng.sample(strata[name], min(16, len(strata[name]))))
        for index, stratum in chosen:
            a, b = left[index], right[index]
            case = f"case-{pair_rng.getrandbits(64):016x}"
            options = [(f, a), (comparator, b)]
            pair_rng.shuffle(options)
            responses, map_back = [], []
            for key, value in options:
                alias = f"response-{pair_rng.getrandbits(64):016x}"
                responses.append({"alias": alias, "prediction": value["prediction"], "truncated": value["truncated"]})
                map_back.append({"alias": alias, "endpoint_key": key, "correct_under_frozen_scorer": value["correct"]})
            row = rows[a["id"]]
            cases.append({"case_id": case, "id": a["id"], "question": row["question"],
                "gold_answer": row["answer"], "original_reference_solution": row["solution"], "responses": responses})
            identities.append({"case_id": case, "comparison": f+"/"+comparator, "stratum": stratum,
                "review_scope": review_scope, "responses": map_back})
    rng.shuffle(cases)
    assert len({r["case_id"] for r in cases}) == len(cases)
    jsonl_write(pack_path, cases)
    jsonl_write(exp / "blind_review_identity_sealed_until_review.jsonl", identities)
    json_write(exp / "blind_review_pack_receipt.json", {"case_count": len(cases), "unique_question_ids": len({r["id"] for r in cases}),
        "seed": plan["blind_review"]["seed"], "scope": "Registered seed29 pairs where both objects succeeded. Include all discordant questions plus up to16 each common-correct and common-wrong per pair. An explicit exploratory same-driver F/U pair is listed separately if available.",
        "registered_pair_count": len(seed29_pairs), "unavailable_registered_pairs": unavailable,
        "complete_registered_scope": not unavailable,
        "exploratory_pairs": [f+"/"+c for f,c,scope,_ in pairs if scope != "registered_seed29"],
        "case_counts_by_scope": dict(Counter(r["review_scope"] for r in identities)),
        "method_names_hidden_in_review_pack": True, "pack_sha256": sha256_file(pack_path), "review_status": "pending_Agent_review_not_human"})


def main(family):
    exp = ROOT / "experiments" / family
    if not (exp / "all_model_tasks_complete.json").exists():
        raise RuntimeError("Finish all registered model tasks before this offline fresh-outcome analysis")
    plan = read(exp / "protocol_frozen.json")
    lock = read(exp / "all_training_endpoints_fixed.json")
    evaluations = read(exp / "evaluation_units.json")
    completion = read(exp / "all_model_tasks_complete.json")
    assert completion["successful_evaluation"] == sum(u["status"] == "first_success_fixed" for u in evaluations.values())
    assert all(u["status"] in {"first_success_fixed", "exhausted_failure", "missing_training_endpoint", "deadline_blocked"} for u in evaluations.values())
    missing_plan_path = exp / "missing_evaluation_analysis_plan_20261118.json"
    missing_plan = read(missing_plan_path) if any(u["status"] != "first_success_fixed" for u in evaluations.values()) else None
    physical_costs, process_ledger, failed_costs, interrupted_costs = accounting(exp, lock["training_units"], evaluations)
    assert read(exp / "fresh_development_opened.json")["endpoint_lock_sha256"] == sha256_file(exp / "all_training_endpoints_fixed.json")
    out = ROOT / "reports/comparisons" / family
    out.mkdir(parents=True, exist_ok=True)
    fresh = read(ROOT / plan["fresh_manifest_path"])
    development = jsonl_read(ROOT / fresh["rows_path"])
    assert sha256_file(ROOT / fresh["rows_path"]) == fresh["rows_sha256"]
    ids = fresh["ids"]
    assert [r["id"] for r in development] == ids and len(set(ids)) == len(ids)
    lock_hash = sha256_file(exp / "all_training_endpoints_fixed.json")
    segments_path = exp / "evaluation_driver_segments.jsonl"
    driver_segments = {r["label"]: r for r in jsonl_read(segments_path)} if segments_path.exists() else {}
    old_driver = "32.0.32015.12012"
    probe_summary = []
    candidates = read(exp / "candidate_pool.json")["ids"]
    for j in PROBE_SEEDS:
        filt = read(exp / f"prefilter_j{j}.json")
        path = ROOT / filt["predictions_path"]
        assert sha256_file(path) == filt["predictions_sha256"]
        raw = jsonl_read(path)
        for r in raw:
            assert all(r[k] == v for k, v in score_answer(r["prediction"], r["gold_answer"]).as_dict().items())
        retained, counts = raw_success_filter(candidates, raw, 4)
        assert retained == filt["retained_ids"] and counts == filt["raw_success_counts"]
        rand = read(exp / f"random_j{j}.json")
        assert rand["ids"] == matched_random_subset(candidates, len(retained), rand["random_subset_seed"])
        histogram = Counter(counts.values())
        probe_summary.append({"probe_seed": j, "candidate_n": len(candidates), "K": 4, "M": len(retained),
            **{f"raw_success_{i}_items": histogram[i] for i in range(5)},
            "full_process_seconds": filt["probe_process_wall_seconds"],
            "generated_tokens": sum(r["generated_tokens"] for r in raw), "prompt_tokens": sum(r["prompt_tokens"] for r in raw),
            "parse_failures": sum(not r["parse_ok"] for r in raw), "truncations": sum(r["truncated"] for r in raw),
            "random_subset_seed": rand["random_subset_seed"], "R_overlap_with_F": len(set(rand["ids"])&set(retained))})
    csv_write(out / "probe_summary.csv", probe_summary)
    predictions, training, endpoints, windows, coverage = {}, {}, [], [], []
    from safetensors.torch import load_file
    original_s2 = load_file(ROOT / START / "adapter_model.safetensors")
    for key in plan["evaluation_order"]:
        item = evaluations[key]
        u = lock["training_units"].get(key)
        row = {"endpoint": key, "arm": u["arm"] if u else key, "probe_seed": u["probe_seed"] if u else None,
            "training_seed": u["seed"] if u else None, "training_status": u["status"] if u else "anchor",
            "evaluation_status": item["status"], "correct": None, "n": plan["development_n"], "evaluated_n": None, "accuracy": None,
            "evaluation_attempt_count": len(item["attempts"]), "evaluation_driver_version": None, "evaluation_boot_time_unix": None,
            "training_process_seconds": u.get("training_process_seconds") if u else None,
            "evaluation_process_seconds": item.get("evaluation_process_seconds"),
            "training_parent_preparation_and_audit_seconds": u.get("parent_training_preparation_and_audit_seconds") if u else None,
            "evaluation_parent_preparation_and_audit_seconds": item.get("parent_evaluation_preparation_and_audit_seconds"),
            "charged_training_path_seconds": (u["training_process_seconds"]+u["parent_training_preparation_and_audit_seconds"]
                +u["total_probe_or_matching_debit_seconds"]) if u and u["status"] == "first_success_fixed" else None,
            "F_full_algorithm_probe_debit_seconds": u.get("probe_algorithm_debit_seconds") if u else 0,
            "R_experimental_matching_debit_seconds": u.get("R_experimental_matching_debit_seconds") if u else 0,
            "charged_end_to_end_seconds": item.get("charged_end_to_end_seconds"),
            "observed_physical_own_training_plus_evaluation_seconds": (item["charged_end_to_end_seconds"]
                - (u.get("total_probe_or_matching_debit_seconds", 0) if u else 0)) if item.get("charged_end_to_end_seconds") is not None else None,
            "C_seconds": plan["budget"]["C_seconds"], "end_to_end_cap_overshoot_seconds": item.get("common_cap_overshoot_seconds"),
            "unused_C_seconds": plan["budget"]["C_seconds"]-item["charged_end_to_end_seconds"] if item.get("charged_end_to_end_seconds") is not None else None,
            "effective_completions": None, "nonzero_gradient_steps": None, "effective_completions_per_charged_second": None,
            "nonzero_gradient_steps_per_charged_second": None, "checkpoint": u.get("checkpoint") if u else lock["objects"][key]["checkpoint"],
            "checkpoint_sha256": lock["objects"][key]["checkpoint_sha256"], "training_run": u.get("run_name") if u else None,
            "evaluation_run": item.get("run_name"),
            **{"training_"+k: None for k in ("optimizer_steps", "pool_n", "zero_variance_fraction", "all_correct_groups", "all_wrong_groups", "mixed_groups",
                "parse_failures", "format_failures", "truncations", "distinct_training_ids", "coverage_fraction", "repetitions_min_including_zero",
                "repetitions_max", "effective_top10_question_share", "adapter_delta_l2_independent", "zero_gradient_adam_changed_steps",
                "peak_allocated_bytes", "peak_reserved_bytes", "generated_tokens", "prompt_tokens", "policy_loss_input_tokens", "wall_budget_overshoot_seconds")},
            **{"evaluation_"+k: None for k in ("format_error_rate", "parse_error_rate", "truncation_rate", "mean_generated_tokens", "p50_generated_tokens", "generated_tokens", "prompt_tokens")}}
        if u and u["status"] == "first_success_fixed":
            signal, ws, cs = audited_training(u, original_s2)
            training[key] = signal
            row["effective_completions"] = signal["effective_completions"]
            row["nonzero_gradient_steps"] = signal["nonzero_gradient_steps"]
            for field in list(row):
                if field.startswith("training_") and field[len("training_"):] in signal:
                    row[field] = signal[field[len("training_"):]]
            windows.extend({"endpoint": key, "arm": u["arm"], "probe_seed": u["probe_seed"], "training_seed": u["seed"], **w} for w in ws)
            coverage.extend({"endpoint": key, "arm": u["arm"], "probe_seed": u["probe_seed"], "training_seed": u["seed"], **c} for c in cs)
        if item["status"] == "first_success_fixed":
            directory = ROOT / "runs" / item["run_name"]
            raw_path = directory / "fresh_development/predictions.jsonl"
            assert sha256_file(raw_path) == item["predictions_sha256"]
            raw = jsonl_read(raw_path)
            assert [r["id"] for r in raw] == ids and len(raw) == fresh["n"]
            for r, original in zip(raw, development, strict=True):
                assert r["gold_answer"] == original["answer"] and r["split"] == "fresh_development"
                assert all(r[k] == v for k, v in score_answer(r["prediction"], r["gold_answer"]).as_dict().items())
            correct = sum(r["correct"] for r in raw)
            metrics = read(directory / "fresh_development/metrics.json")
            assert metrics["n"] == len(ids) and metrics["decoding"] == "greedy" and metrics["thinking"] is False
            assert metrics["scorer_version"] == "numeric_final_v2"
            assert correct == item["correct"] == metrics["correct"]
            protocol = read(directory / "evaluation_protocol.json")
            assert protocol["adapter_sha256"] == lock["objects"][key]["checkpoint_sha256"]
            assert protocol["endpoint_lock_sha256"] == lock_hash and protocol["fresh_development_endpoint_key"] == key
            accepted = item["attempts"][item["selected_attempt"]]
            assert accepted["status"] == "success" and accepted["receipt"]["exit_code"] == 0
            segment = driver_segments.get(accepted["label"])
            if segment is not None:
                assert segment["process_receipt_finished_unix"] == accepted["receipt"]["finished_unix"]
            elif driver_segments:
                assert accepted["receipt"]["finished_unix"] < min(s["process_receipt_finished_unix"] for s in driver_segments.values())
            row.update(evaluation_driver_version=segment["driver"] if segment else old_driver,
                evaluation_boot_time_unix=accepted["receipt"]["host_before"]["boot_time_unix"])
            for field in list(row):
                if field.startswith("evaluation_") and field[len("evaluation_"):] in metrics:
                    row[field] = metrics[field[len("evaluation_"):]]
            predictions[key] = raw
            snapshot = read(directory / "model_precision.json")["standalone_loader"]
            assert snapshot["is_loaded_in_4bit"] is False and snapshot["attention_implementation"] == "sdpa"
            assert snapshot["input_embedding"] == snapshot["output_head"] == "torch.bfloat16"
            assert snapshot["parameter_devices"] == ["cuda:0"]
            evaluation_costs = jsonl_read(directory / "costs.jsonl")
            row.update(correct=correct, evaluated_n=len(raw), accuracy=correct/len(raw),
                evaluation_peak_allocated_bytes=max((c.get("peak_allocated_bytes") or 0 for c in evaluation_costs), default=0),
                evaluation_peak_reserved_bytes=max((c.get("peak_reserved_bytes") or 0 for c in evaluation_costs), default=0))
            if u and key in training:
                row["effective_completions_per_charged_second"] = training[key]["effective_completions"]/item["charged_end_to_end_seconds"]
                row["nonzero_gradient_steps_per_charged_second"] = training[key]["nonzero_gradient_steps"]/item["charged_end_to_end_seconds"]
        row.update(endpoint_failure_columns(row, failed_costs.get(key, {})))
        row.setdefault("evaluation_peak_allocated_bytes", None)
        row.setdefault("evaluation_peak_reserved_bytes", None)
        endpoints.append(row)
    csv_write(out / "endpoints.csv", endpoints)
    csv_write(out / "training_signal_windows.csv", windows)
    csv_write(out / "training_question_coverage.csv", coverage)
    json_write(out / "training_diagnostics.json", training)
    missing = [k for k in plan["evaluation_order"] if k not in predictions]
    summary = {"family": family, "analyzed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n": len(ids), "development_clusters": len(set(r["cluster_id"] for r in development)),
        "registered_training": 35, "successful_training": len(training), "registered_evaluation": 37,
        "successful_evaluation": len(predictions), "missing_evaluation_objects": missing, "budget": plan["budget"],
        "training_signal": training, "endpoints": endpoints, "probe_summary": probe_summary, "holdouts_opened": False}
    if not missing:
        keys = list(predictions)
        correct = np.array([[r["correct"] for r in predictions[k]] for k in keys], dtype=float)
        F = np.array([[correct[keys.index(f"F-j{j}-s{s}")] for s in TRAIN_SEEDS] for j in PROBE_SEEDS])
        R = np.array([[correct[keys.index(f"R-j{j}-s{s}")] for s in TRAIN_SEEDS] for j in PROBE_SEEDS])
        U = np.array([correct[keys.index(f"U-s{s}")] for s in TRAIN_SEEDS])
        FU = paired_difference_matrix(F, U)
        FR = (F-R).mean(axis=2)*100
        pairs = [{"probe_seed": j, "training_seed": s, "F_correct": int(F[a, b].sum()),
            "U_correct": int(U[b].sum()), "R_correct": int(R[a, b].sum()), "n": len(ids),
            "F_minus_U_pp": float(FU[a, b]), "F_minus_R_pp": float(FR[a, b]),
            "F_minus_S2_pp": float((F[a,b]-correct[keys.index('S2')]).mean()*100),
            "F_minus_A_pp": float((F[a,b]-correct[keys.index('A')]).mean()*100)}
            for a, j in enumerate(PROBE_SEEDS) for b, s in enumerate(TRAIN_SEEDS)]
        csv_write(out / "paired_differences.csv", pairs)
        joint = joint_cluster_accuracies(correct, [r["cluster_id"] for r in development],
            replicates=plan["statistics"]["conditional_cluster_bootstrap_replicates"], seed=plan["statistics"]["seed"])
        np.savez_compressed(out / "joint_cluster_bootstrap.npz", joint_accuracies=joint, keys=np.array(keys),
            seed=plan["statistics"]["seed"], replicates=len(joint))
        summary.update(F_minus_U_mean_pp=float(FU.mean()), F_minus_R_mean_pp=float(FR.mean()),
            F_minus_S2_mean_pp=float((F.mean(axis=(0, 1))-correct[keys.index("S2")]).mean()*100),
            F_minus_A_mean_pp=float((F.mean(axis=(0, 1))-correct[keys.index("A")]).mean()*100),
            paired_F_minus_U_matrix_pp=FU.tolist(), paired_F_minus_R_matrix_pp=FR.tolist(),
            conditional_intervals=intervals_from_joint(joint, keys, TRAIN_SEEDS, PROBE_SEEDS),
            crossed_exploratory_intervals=intervals_from_joint(joint, keys, TRAIN_SEEDS, PROBE_SEEDS, crossed=True),
            probe_block_sensitivity={str(j): {"F_minus_U_mean_pp": float(FU[a].mean()),
                "F_minus_U_range_pp": [float(FU[a].min()), float(FU[a].max())], "F_minus_U_sd_pp": float(FU[a].std(ddof=1)),
                "F_minus_R_mean_pp": float(FR[a].mean()), "F_minus_R_range_pp": [float(FR[a].min()), float(FR[a].max())],
                "F_minus_R_sd_pp": float(FR[a].std(ddof=1))} for a, j in enumerate(PROBE_SEEDS)},
            training_seed_sensitivity={str(s): {"F_minus_U_mean_pp": float(FU[:, b].mean()),
                "F_minus_U_range_pp": [float(FU[:, b].min()), float(FU[:, b].max())], "F_minus_U_sd_pp": float(FU[:, b].std(ddof=1)),
                "F_minus_R_mean_pp": float(FR[:,b].mean()), "F_minus_R_range_pp": [float(FR[:,b].min()),float(FR[:,b].max())],
                "F_minus_R_sd_pp": float(FR[:,b].std(ddof=1))}
                for b, s in enumerate(TRAIN_SEEDS)})
        S2 = correct[keys.index("S2")].astype(bool)
        transitions = {key: {"S2_correct_then_wrong": int((S2 & ~correct[i].astype(bool)).sum()),
            "S2_wrong_then_correct": int((~S2 & correct[i].astype(bool)).sum()),
            "net_correct_difference": int(correct[i].sum()-S2.sum())} for i, key in enumerate(keys)}
        summary["transitions_from_S2"] = transitions
        json_write(out / "transitions_from_S2.json", transitions)
        transition_ids = {key: {
            "S2_correct_then_wrong_ids": [ids[q] for q in np.flatnonzero(S2 & ~correct[i].astype(bool))],
            "S2_wrong_then_correct_ids": [ids[q] for q in np.flatnonzero(~S2 & correct[i].astype(bool))]}
            for i, key in enumerate(keys)}
        assert all(len(transition_ids[k]["S2_correct_then_wrong_ids"]) == t["S2_correct_then_wrong"]
            and len(transition_ids[k]["S2_wrong_then_correct_ids"]) == t["S2_wrong_then_correct"]
            for k, t in transitions.items())
        json_write(out / "transition_ids_from_S2.json", transition_ids)
        decomposition = []
        for j in PROBE_SEEDS:
            for seed in TRAIN_SEEDS:
                f, u = transitions[f"F-j{j}-s{seed}"], transitions[f"U-s{seed}"]
                learned = f["S2_wrong_then_correct"]-u["S2_wrong_then_correct"]
                avoided = u["S2_correct_then_wrong"]-f["S2_correct_then_wrong"]
                assert learned+avoided == f["net_correct_difference"]-u["net_correct_difference"]
                decomposition.append({"probe_seed": j, "training_seed": seed, "n": len(ids),
                    "F_S2_wrong_then_correct": f["S2_wrong_then_correct"],
                    "U_S2_wrong_then_correct": u["S2_wrong_then_correct"],
                    "F_S2_correct_then_wrong": f["S2_correct_then_wrong"],
                    "U_S2_correct_then_wrong": u["S2_correct_then_wrong"],
                    "additional_S2_wrong_then_correct_pp": learned/len(ids)*100,
                    "avoided_S2_correct_then_wrong_pp": avoided/len(ids)*100,
                    "F_minus_U_pp": (learned+avoided)/len(ids)*100})
        csv_write(out / "paired_s2_transition_decomposition.csv", decomposition)
        summary["F_minus_U_behavioral_decomposition_pp"] = {
            k: statistics.mean(r[k] for r in decomposition) for k in (
                "additional_S2_wrong_then_correct_pp", "avoided_S2_correct_then_wrong_pp", "F_minus_U_pp")}
        def aggregate_rate(arm, j=None):
            selected = [r for r in endpoints if r["arm"] == arm and (j is None or r["probe_seed"] == j)]
            denominator = sum(r["charged_end_to_end_seconds"] for r in selected)
            return {"effective_completions_per_charged_second": sum(r["effective_completions"] for r in selected)/denominator,
                "nonzero_gradient_steps_per_charged_second": sum(r["nonzero_gradient_steps"] for r in selected)/denominator,
                "endpoint_count": len(selected), "charged_seconds_sum": denominator}
        rates = {a: aggregate_rate(a) for a in ("U", "F", "R")}
        blocks = {str(j): aggregate_rate("F", j) for j in PROBE_SEEDS}
        baseline_rate = rates["U"]["effective_completions_per_charged_second"]
        improvement = rates["F"]["effective_completions_per_charged_second"]/baseline_rate-1 if baseline_rate > 0 else None
        positive = all(v["effective_completions_per_charged_second"] > rates["U"]["effective_completions_per_charged_second"] for v in blocks.values())
        summary.update(charged_signal_rates=rates, F_probe_block_signal_rates=blocks,
            F_vs_U_effective_signal_relative_improvement=improvement,
            engineering_signal_threshold_met=improvement is not None and improvement >= 0.20 and positive,
            all_three_signal_block_directions_positive=positive,
            development_evidence_threshold_met=FU.mean() >= 1.0 and summary["conditional_intervals"]["F-U"]["percentile95_pp"][0] > 0)
        operational = {arm: operational_rate_bounds(endpoints, arm) for arm in ("U", "F", "R")}
        operational_blocks = {str(j): operational_rate_bounds(endpoints, "F", j) for j in PROBE_SEEDS}
        u_low = operational["U"]["effective_completions_per_second_lower_bound"]
        u_high = operational["U"]["effective_completions_per_second_upper_bound"]
        op_improvement = [operational["F"]["effective_completions_per_second_lower_bound"]/u_high-1,
            operational["F"]["effective_completions_per_second_upper_bound"]/u_low-1] if u_low > 0 else None
        op_positive = all(v["effective_completions_per_second_lower_bound"] > u_high for v in operational_blocks.values())
        summary.update(failure_inclusive_operational_signal_rates=operational,
            failure_inclusive_F_probe_block_rates=operational_blocks,
            failure_inclusive_F_vs_U_effective_signal_relative_improvement_bounds=op_improvement,
            failure_inclusive_all_three_signal_directions_positive=op_positive,
            failure_inclusive_engineering_threshold_at_cost_lower_effect_bound=op_improvement is not None and op_improvement[0] >= .20 and op_positive,
            primary_signal_charge_scope="Retained first-success endpoint path with full P/R matching; unchanged primary definition. Failed/discarded attempts are additionally shown as bounded operational cost sensitivity; their signals are not credited to retained weights.")
    else:
        assert missing_plan is not None
        endpoint_table = {r["endpoint"]: r for r in endpoints}
        cluster_ids = [row["cluster_id"] for row in development]
        pairs, available, keys, joint, correct = differences(predictions, cluster_ids, endpoint_table,
            TRAIN_SEEDS, PROBE_SEEDS, plan["statistics"])
        csv_write(out / "paired_differences.csv", pairs)
        np.savez_compressed(out / "joint_cluster_bootstrap.npz", joint_accuracies=joint, keys=np.array(keys),
            seed=plan["statistics"]["seed"], replicates=len(joint))
        transitions, decomposition = transitions_from_s2(correct, ids, pairs)
        json_write(out / "transitions_from_S2.json", transitions)
        csv_write(out / "paired_s2_transition_decomposition.csv", decomposition)
        summary.update(original_full_matrix_inference_available=False, committed_missing_evaluation_analysis_plan_sha256=sha256_file(missing_plan_path),
            available_pair_comparisons=available, signal_sensitivity=signal_rates(endpoint_table, TRAIN_SEEDS, PROBE_SEEDS),
            transitions_from_S2=transitions, development_evidence_threshold_met=None, engineering_signal_threshold_met=None,
            across_driver_evaluation_counts=dict(Counter(r["evaluation_driver_version"] for r in endpoints if r["accuracy"] is not None)))
    blind_pack(exp, plan, predictions, development, {r["endpoint"]: r for r in endpoints}, missing_plan)
    shared = sum(r["process_wall_seconds"] or 0 for r in process_ledger if r["label"].startswith("probe-"))
    train_labels = {a["label"] for u in lock["training_units"].values() for a in u["attempts"]}
    process_by_stage = {k: {"attempts": 0, "seconds_lower_bound": 0.0, "seconds_upper_bound": 0.0}
        for k in ("calibration_and_recovery_checks", "shared_probes", "training", "fresh_development_evaluation")}
    for receipt in process_ledger:
        label = receipt["label"]
        stage = "training" if label in train_labels else "shared_probes" if label.startswith("probe-") else (
            "fresh_development_evaluation" if label.startswith("eval-") else "calibration_and_recovery_checks")
        process_by_stage[stage]["attempts"] += 1
        for bound in ("lower_bound", "upper_bound"):
            process_by_stage[stage]["seconds_"+bound] += receipt["process_seconds_"+bound]
    physical_costs.update({"shared_probe_physical_seconds_including_probe_failures": shared,
        "process_by_stage": process_by_stage,
        "registered_session_elapsed_wall_seconds_at_analysis": time.time()-read(exp / "session.json")["registered_unix"],
        "R_matching_debit_seconds_sum_not_extra_physical_cost": sum(r["R_experimental_matching_debit_seconds"] for r in endpoints),
        "F_standalone_probe_debit_seconds_sum_not_extra_shared_executions": sum(r["F_full_algorithm_probe_debit_seconds"] for r in endpoints),
        "R_subset_draw_physical_seconds": sum(read(exp / f"random_j{j}.json")["physical_subset_draw_seconds"] for j in PROBE_SEEDS)})
    summary["physical_costs"] = physical_costs
    csv_write(out / "process_cost_ledger.csv", process_ledger)
    csv_write(out / "interrupted_process_cost_bounds.csv", [{k: r[k] for k in (
        "label", "started_unix", "outer_budget_start_unix", "last_raw_activity_unix", "next_OS_boot_event_unix",
        "process_seconds_lower_bound", "process_seconds_upper_bound", "failed_endpoint_path_seconds_lower_bound",
        "failed_endpoint_path_seconds_upper_bound", "known_parent_preparation_seconds", "exact_exit_code", "raw_costs_sha256")} for r in interrupted_costs])
    json_write(out / "failed_attempt_costs_by_endpoint.json", failed_costs)
    json_write(out / "summary.json", summary)
    json_write(out / "evidence_audit.json", {"status": "passed_with_explicit_missingness" if missing else "all37_fresh_predictions_and35_training_evidence_audited",
        "successful_training_audited": len(training), "successful_evaluations_audited": len(predictions),
        "missing_objects": missing, "all_scores_recomputed_with_unchanged_scorer": True,
        "actual_tensor_masks_advantages_gradients_adam_and_parameter_deltas_checked": True,
        "no_formal_holdout_contents_read": True, "blind_review_is_separate_pending_step": True,
        "interrupted_process_costs_not_omitted": True, "primary_metrics_and_thresholds_unchanged": True})
    print(json.dumps({"family": family, "training_audited": len(training), "evaluation_audited": len(predictions), "missing": missing,
        "F_minus_U_mean_pp": summary.get("F_minus_U_mean_pp"), "F_minus_R_mean_pp": summary.get("F_minus_R_mean_pp")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
