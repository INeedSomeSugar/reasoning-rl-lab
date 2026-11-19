"""Audit real training-pool exposure and probe uncertainty without model/holdout access."""
import argparse
import json
import math
from collections import Counter, defaultdict

from reasoning_rl.data import load_split
from reasoning_rl.reporting import _plot_style, write_csv
from reasoning_rl.runtime import ROOT, bootstrap, json_write, jsonl_read


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    bootstrap()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="localbudget_20261105")
    parser.add_argument("--runs", nargs="*")
    args = parser.parse_args()
    directories = [ROOT / "runs" / name for name in args.runs] if args.runs else sorted((ROOT / "runs").glob(args.prefix + "-*-s*"))
    train = {r["id"]: r for r in load_split("train", "develop")}
    audit_path = ROOT / "reports/training_label_audit.json"
    flagged = {r["id"] for r in read(audit_path)["records"]} if audit_path.exists() else set()
    output = ROOT / "reports/comparisons" / args.prefix
    output.mkdir(parents=True, exist_ok=True)
    summaries, exposures, probe_rows, reward_points = [], [], [], []
    for directory in directories:
        if not (directory / "training_result.json").exists():
            continue
        result = read(directory / "training_result.json")
        contract = read(directory / "comparison_contract.json")
        ids = contract["pool_ids"]
        if not set(ids) <= train.keys():
            raise ValueError("Sampling pool contains a non-training ID")
        selections = jsonl_read(directory / "selections.jsonl")
        draws = [i for event in selections for i in event["ids"]]
        if not set(draws) <= set(ids):
            raise ValueError("Selection escaped the declared pool")
        counts = Counter(draws)
        expected_counts = {i: sum(event["weights"][i] * len(event["ids"]) for event in selections) for i in ids}
        groups = defaultdict(list)
        rollouts = jsonl_read(directory / "rollouts.jsonl")
        for row in rollouts:
            groups[(row["generation_batch"], row["group_within_batch"])].append(row)
        if len(groups) != result["groups"]:
            raise ValueError("Saved group count does not reconcile")
        mixed_by_id, groups_by_id = Counter(), Counter()
        for group in groups.values():
            if len({r["id"] for r in group}) != 1:
                raise ValueError("Group mixes questions")
            item = group[0]["id"]
            groups_by_id[item] += 1
            mixed_by_id[item] += int(len({r["reward"] for r in group}) > 1)
        events = []
        previous_weights = None
        for path in sorted((directory / "probes").glob("step_*_state.json")):
            state = read(path)
            step = int(path.stem.split("_")[1])
            predictions = jsonl_read(path.with_name(path.name.replace("_state.json", ".jsonl")))
            by_id = defaultdict(list)
            for prediction in predictions:
                by_id[prediction["id"]].append(prediction)
            if set(by_id) != set(ids):
                raise ValueError("Complete probe lacks a pool item or contains a foreign item")
            for item in ids:
                observed = by_id[item]
                n, successes = len(observed), sum(r["correct"] for r in observed)
                prior = state["prior"]
                a, b = successes + prior, n - successes + prior
                p = a / (a + b)
                if not math.isclose(p, state["p_success"][item], abs_tol=1e-12):
                    raise ValueError("Probe success estimate differs from saved sampler state")
                probe_rows.append({"run": directory.name, "strategy": contract["strategy"], "seed": contract["protocol"]["seed"],
                    "optimizer_step": step, "id": item, "successes": successes, "trials": n,
                    "posterior_mean": p, "beta_posterior_std": math.sqrt(a*b / ((a+b)**2 * (a+b+1))),
                    "weight": state["weights"][item], "format_error_rate": sum(not r["format_ok"] for r in observed)/n,
                    "truncation_rate": sum(r["truncated"] for r in observed)/n,
                    "interpretation": "Descriptive Beta posterior under iid Bernoulli assumptions; not a calibrated model-skill confidence interval"})
            weights = state["weights"]
            tv = None if previous_weights is None else sum(abs(weights[i] - previous_weights[i]) for i in ids)/2
            events.append({"step": step, "update_index": state["updates"],
                           "draws_using_this_update": sum(len(s["ids"]) for s in selections if s["sampler_updates"] == state["updates"]),
                           "effective_pool_size": state["effective_pool_size"],
                           "max_min_weight_ratio": max(weights.values())/min(weights.values()),
                           "weight_total_variation_from_previous": tv})
            previous_weights = weights
        costs = jsonl_read(directory / "costs.jsonl")
        total = next(c for c in costs if c["phase"] == "grpo_total")
        probes = [c for c in costs if c["phase"] == "difficulty_probe"]
        validations = [c for c in costs if c["phase"] == "validation"]
        mixed = sum(mixed_by_id.values())
        summaries.append({"run": directory.name, "strategy": contract["strategy"], "seed": contract["protocol"]["seed"],
            "pool_n": len(ids), "draws": len(draws), "unique_selected": len(counts), "unseen_pool_items": [i for i in ids if not counts[i]],
            "max_exposure_share": max(counts.values())/len(draws) if draws else None,
            "empirical_vs_expected_selection_tv": sum(abs(counts[i]/len(draws) - expected_counts[i]/len(draws)) for i in ids)/2 if draws else None,
            "groups": len(groups), "mixed_reward_groups": mixed, "mixed_groups_per_grpo_minute": mixed/(total["wall_seconds"]/60),
            "probe_fraction_of_grpo_time": sum(c["wall_seconds"] for c in probes)/total["wall_seconds"],
            "validation_fraction_of_grpo_time": sum(c["wall_seconds"] for c in validations)/total["wall_seconds"],
            "complete_probe_events": events, "incomplete_probe_calls": sum(not c.get("probe_complete", True) for c in probes),
            "incomplete_probe_items": sum(c.get("probe_items", 0) for c in probes if not c.get("probe_complete", True)),
            "weight_change_events": result["sampler_weight_change_events"],
            "suspected_label_issue_draws": sum(counts[i] for i in flagged),
            "suspected_label_issue_rollouts": sum(r["id"] in flagged for r in rollouts),
            "suspected_label_issue_positive_rewards": sum(r["reward"] for r in rollouts if r["id"] in flagged),
            "note": "Mixed reward can still yield no gradient after truncated-completion masking; no learning or accuracy claim from this statistic alone."})
        for item in ids:
            exposures.append({"run": directory.name, "strategy": contract["strategy"], "seed": contract["protocol"]["seed"], "id": item,
                "selected_count": counts[item], "expected_selected_count": expected_counts[item],
                "observed_groups": groups_by_id[item], "mixed_reward_groups": mixed_by_id[item],
                "suspected_label_issue": item in flagged,
                "question_words": len(train[item]["question"].split()),
                "solution_contains_division": "/" in train[item]["solution"],
                "solution_contains_multiplication": "*" in train[item]["solution"],
                "topic_label_status": "Only overlapping text features; no human math-topic annotation"})
        progress = {c["optimizer_step"]: c for c in costs if c["phase"] == "budget_progress"}
        for event in costs:
            if event["phase"] == "grpo_log" and "reward" in event and event["optimizer_step"] in progress:
                reward_points.append({"run": directory.name, "step": event["optimizer_step"],
                    "cumulative_wall_seconds": progress[event["optimizer_step"]]["cumulative_training_wall_seconds"],
                    "reward": event["reward"], "zero_variance_fraction": progress[event["optimizer_step"]]["zero_variance_group_fraction"]})
    if not summaries:
        raise ValueError("No completed GRPO run to audit")
    json_write(output / "sampling_diagnostics.json", {"scope": "Training-only descriptive diagnostics; no holdout read", "runs": summaries,
        "uncertainty_note": "With K=2 and prior=0.5, success estimates take only 1/6, 1/2, 5/6. Changes can be Monte Carlo noise.",
        "coverage_note": "A positive sampling floor ensures nonzero draw probability, not exposure to every item in a short run."})
    write_csv(output / "item_exposure.csv", exposures)
    write_csv(output / "probe_uncertainty.csv", probe_rows)
    write_csv(output / "reward_compute_points.csv", reward_points)
    plt = _plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    colors = {"uniform": "#2563a6", "dynamic": "#c07810", "static": "#657d39"}
    for summary in summaries:
        label = f"{summary['strategy']} s{summary['seed']}"
        points = [p for p in reward_points if p["run"] == summary["run"]]
        if points:
            axes[0].scatter([p["cumulative_wall_seconds"]/60 for p in points], [p["reward"] for p in points],
                            label=label, color=colors[summary["strategy"]], s=18, alpha=0.7)
        axes[1].bar(label, summary["mixed_groups_per_grpo_minute"], color=colors[summary["strategy"]])
    axes[0].set(title="Observed training correctness rewards", xlabel="Cumulative GRPO elapsed time (min)", ylabel="Group reward (0/1 mean)", ylim=(-0.04, 1.04))
    axes[0].legend(fontsize=7)
    axes[1].set(title="Mixed groups per total GRPO minute", ylabel="Mixed groups / min (validation included)")
    axes[1].tick_params(axis="x", labelrotation=35)
    fig.suptitle("Training diagnostics only | finite probes, truncated-loss masking, and actual costs", fontsize=11)
    fig.savefig(output / "sampling_diagnostics.png", dpi=160)
    fig.savefig(output / "sampling_diagnostics.svg")
    plt.close(fig)
    print(output / "sampling_diagnostics.json")


if __name__ == "__main__":
    main()
