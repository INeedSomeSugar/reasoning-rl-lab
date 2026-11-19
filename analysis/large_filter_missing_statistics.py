"""Statistics for the registered matrix with explicit unavailable evaluations."""
from collections import defaultdict, Counter
import statistics

import numpy as np

from reasoning_rl.replication_statistics import joint_cluster_accuracies


CONTRASTS = ("F-U", "F-R", "F-S2", "F-A")


def registered_pairs(training_seeds, probe_seeds):
    for probe in probe_seeds:
        for seed in training_seeds:
            f = f"F-j{probe}-s{seed}"
            yield {"probe_seed": probe, "training_seed": seed, "F": f,
                "U": f"U-s{seed}", "R": f"R-j{probe}-s{seed}", "S2": "S2", "A": "A"}


def full_matrix_bounds(rows, counts, n, right):
    # Each shared U outcome is one unknown with its total weight across the entire matrix.
    coefficients = Counter()
    for pair in rows:
        coefficients[pair["F"]] += 1
        coefficients[pair[right]] -= 1
    known = sum(coeff*counts[key] for key, coeff in coefficients.items() if key in counts)
    unknown = {key: coeff for key, coeff in coefficients.items() if key not in counts and coeff}
    lower = known + n*sum(coeff for coeff in unknown.values() if coeff < 0)
    upper = known + n*sum(coeff for coeff in unknown.values() if coeff > 0)
    scale = 100/(n*len(rows))
    return {"bounds_pp": [lower*scale, upper*scale], "missing_endpoint_cell_weights": unknown,
        "known_weighted_correct_count": known, "registered_cells": len(rows),
        "scope": "Worst/best full-matrix differences under unrestricted missing accuracies; each missing model reused consistently. No imputation."}


def stats(values):
    return {"pairs": len(values), "mean_pp": statistics.mean(values) if values else None,
        "range_pp": [min(values), max(values)] if values else None,
        "sd_pp": statistics.stdev(values) if len(values) > 1 else None}


def differences(predictions, cluster_ids, endpoints, training_seeds, probe_seeds, protocol):
    keys = list(predictions)
    correct = {key: np.array([r["correct"] for r in predictions[key]], dtype=bool) for key in keys}
    counts = {key: int(value.sum()) for key, value in correct.items()}
    n = len(cluster_ids)
    assert keys and all(len(r) == n for r in correct.values())
    index = {key: i for i, key in enumerate(keys)}
    joint = joint_cluster_accuracies(np.stack(list(correct.values())), cluster_ids,
        replicates=protocol["conditional_cluster_bootstrap_replicates"], seed=protocol["seed"])
    registered = list(registered_pairs(training_seeds, probe_seeds))
    rows = []
    for pair in registered:
        row = {"probe_seed": pair["probe_seed"], "training_seed": pair["training_seed"], "n": n}
        for arm in ("F", "U", "R", "S2", "A"):
            key = pair[arm]
            row.update({f"{arm}_endpoint": key, f"{arm}_correct": counts.get(key),
                f"{arm}_evaluation_status": endpoints[key]["evaluation_status"],
                f"{arm}_driver": endpoints[key]["evaluation_driver_version"]})
        for comparison in CONTRASTS:
            right = comparison.split("-")[1]
            f, c = pair["F"], pair[right]
            missing = [k for k in (f, c) if k not in counts]
            base = comparison.replace("-", "_minus_")
            row[f"{base}_pp"] = None if missing else (counts[f]-counts[c])/n*100
            row[f"{base}_missing_objects"] = ",".join(missing)
            f_driver, c_driver = row["F_driver"], row[f"{right}_driver"]
            row[f"{base}_same_driver"] = None if missing or not f_driver or not c_driver else f_driver == c_driver
        rows.append(row)
    summary = {}
    for comparison in CONTRASTS:
        right = comparison.split("-")[1]
        base = comparison.replace("-", "_minus_")
        actual = [r for r in rows if r[f"{base}_pp"] is not None]
        values = [r[f"{base}_pp"] for r in actual]
        available = stats(values)
        if actual:
            # For all pairs and bootstrap replicates, index the very same joint draw of each U.
            left_indices = [index[r["F_endpoint"]] for r in actual]
            right_indices = [index[r[f"{right}_endpoint"]] for r in actual]
            draws = (joint[:, left_indices]-joint[:, right_indices]).mean(axis=1)*100
            available["conditional_percentile95_pp"] = np.quantile(draws, [0.025, 0.975]).tolist()
        else:
            available["conditional_percentile95_pp"] = None
        available["fixed_training_probe_and_observed_cells"] = True
        available["joint_cluster_replicates"] = len(joint)
        summary[comparison] = {"registered_pairs": len(registered), "available": available,
            "original_point_estimate_pp": available["mean_pp"] if len(actual) == len(registered) else None,
            "full_matrix_sensitivity": full_matrix_bounds(registered, counts, n, right),
            "by_probe_seed": {str(probe): stats([r[f"{base}_pp"] for r in actual if r["probe_seed"] == probe]) for probe in probe_seeds},
            "by_training_seed": {str(seed): stats([r[f"{base}_pp"] for r in actual if r["training_seed"] == seed]) for seed in training_seeds},
            "same_driver": stats([r[f"{base}_pp"] for r in actual if r[f"{base}_same_driver"] is True]),
            "different_driver": stats([r[f"{base}_pp"] for r in actual if r[f"{base}_same_driver"] is False]),
            "unknown_driver_pairs": sum(r[f"{base}_same_driver"] is None for r in actual),
            "matrix_pp_probe_by_training": [[r[f"{base}_pp"] for r in rows if r["probe_seed"] == probe] for probe in probe_seeds]}
    return rows, summary, keys, joint, correct


def transitions_from_s2(correct, ids, paired_rows):
    anchor = correct["S2"]
    transitions = {}
    for key, outcome in correct.items():
        regressions = [ids[q] for q in np.flatnonzero(anchor & ~outcome)]
        acquisitions = [ids[q] for q in np.flatnonzero(~anchor & outcome)]
        net = len(acquisitions)-len(regressions)
        assert net == int(outcome.sum()-anchor.sum())
        transitions[key] = {"S2_correct_then_wrong": len(regressions), "S2_wrong_then_correct": len(acquisitions),
            "net_correct_difference": net, "S2_correct_then_wrong_ids": regressions, "S2_wrong_then_correct_ids": acquisitions}
    decomposition = []
    for pair in paired_rows:
        if pair["F_minus_U_pp"] is None:
            continue
        f, u = (transitions[pair[f"{arm}_endpoint"]] for arm in ("F", "U"))
        learned = f["S2_wrong_then_correct"]-u["S2_wrong_then_correct"]
        avoided = u["S2_correct_then_wrong"]-f["S2_correct_then_wrong"]
        assert learned+avoided == f["net_correct_difference"]-u["net_correct_difference"]
        decomposition.append({"probe_seed": pair["probe_seed"], "training_seed": pair["training_seed"],
            "F_S2_wrong_then_correct": f["S2_wrong_then_correct"], "U_S2_wrong_then_correct": u["S2_wrong_then_correct"],
            "F_S2_correct_then_wrong": f["S2_correct_then_wrong"], "U_S2_correct_then_wrong": u["S2_correct_then_wrong"],
            "additional_S2_wrong_then_correct_pp": learned/len(ids)*100,
            "avoided_S2_correct_then_wrong_pp": avoided/len(ids)*100, "F_minus_U_pp": pair["F_minus_U_pp"]})
    return transitions, decomposition


def pooled_rate(rows, denominator):
    if not rows:
        return None
    assert all(r[denominator] is not None and r[denominator] > 0 for r in rows)
    seconds = sum(r[denominator] for r in rows)
    completions = sum(r["effective_completions"] for r in rows)
    gradients = sum(r["nonzero_gradient_steps"] for r in rows)
    return {"endpoints": len(rows), "seconds": seconds, "effective_completions": completions, "nonzero_gradient_steps": gradients,
        "effective_completions_per_second": completions/seconds, "nonzero_gradient_steps_per_second": gradients/seconds}


def signal_rates(endpoints, training_seeds, probe_seeds):
    arms = defaultdict(list)
    for row in endpoints.values():
        if row["arm"] in ("F", "R", "U"):
            arms[row["arm"]].append(row)
    pre_eval = {arm: pooled_rate(rows, "charged_training_path_seconds") for arm, rows in arms.items()}
    pre_blocks = {str(probe): {arm: pooled_rate([r for r in arms[arm] if r["probe_seed"] == probe], "charged_training_path_seconds")
        for arm in ("F", "R")} for probe in probe_seeds}
    end_to_end = "charged_end_to_end_seconds"
    complete_seed_blocks = [seed for seed in training_seeds if all(
        row[end_to_end] is not None for group in arms.values() for row in group if row["training_seed"] == seed)]
    complete_rates = {arm: pooled_rate([r for r in rows if r["training_seed"] in complete_seed_blocks], end_to_end)
        for arm, rows in arms.items()}
    matched = {}
    for probe in probe_seeds:
        block = {}
        for comparator in ("U", "R"):
            pairs = [(endpoints[f"F-j{probe}-s{seed}"], endpoints[f"U-s{seed}" if comparator == "U" else f"R-j{probe}-s{seed}"])
                for seed in training_seeds]
            actual = [(l, r) for l, r in pairs if l[end_to_end] is not None and r[end_to_end] is not None]
            block[f"F-{comparator}"] = {"training_seeds": [l["training_seed"] for l, _ in actual],
                "F": pooled_rate([l for l, _ in actual], end_to_end), comparator: pooled_rate([r for _, r in actual], end_to_end)}
        matched[str(probe)] = block
    return {"all35_charged_training_only": pre_eval, "all35_training_only_by_probe_seed": pre_blocks,
        "complete_end_to_end_training_seed_blocks": complete_seed_blocks, "complete_seed_block_end_to_end": complete_rates,
        "matched_end_to_end_by_probe": matched,
        "original_end_to_end_engineering_decision": None,
        "original_decision_unavailable_reason": "One or more registered F/U endpoints lack a successful full evaluation. No zero denominator or fake evaluation duration assigned."}
