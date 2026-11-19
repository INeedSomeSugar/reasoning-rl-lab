"""Recompute group/step signal denominators from immutable historical evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, checkpoint_hash
from reasoning_rl.scoring import score_answer, group_advantages
from filter_common import EXP, FAMILY


def audit_run(path):
    rollouts = jsonl_read(path / "rollouts.jsonl")
    groups = defaultdict(list)
    mismatches = []
    counts = Counter()
    for r in rollouts:
        s = score_answer(r["prediction"], r["gold_answer"])
        if float(s.correct) != r["reward"] or any(s.as_dict()[k] != r[k] for k in s.as_dict()):
            mismatches.append(r["id"])
        groups[(r["generation_batch"], r["group_within_batch"])].append(r)
        counts["completions"] += 1
        counts["truncated_completions"] += int(r["truncated"])
        counts["parse_failures"] += int(not r["parse_ok"])
        counts["format_failures"] += int(not r["format_ok"])
        counts["generated_tokens"] += r["generated_tokens"]
    signal_by_batch = defaultdict(int)
    for (batch, _), rs in groups.items():
        assert len(rs) == 4 and len({r["id"] for r in rs}) == 1
        n = sum(r["reward"] for r in rs)
        kind = "all_correct" if n == 4 else "all_wrong" if n == 0 else "mixed"
        counts[kind + "_groups"] += 1
        advs = group_advantages([r["reward"] for r in rs])
        for r, a in zip(rs, advs, strict=True):
            counts["zero_advantage_completions"] += int(a == 0)
            counts["nonzero_advantage_completions"] += int(a != 0)
            counts["advantage_masked_by_truncation"] += int(a != 0 and r["truncated"])
            effective = a != 0 and not r["truncated"]
            counts["effective_advantage_completions"] += int(effective)
            signal_by_batch[batch] += int(effective)
    costs = jsonl_read(path / "costs.jsonl")
    grads = [r for r in costs if r["phase"] == "gradient_diagnostic"]
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    result = json.loads((path / "training_result.json").read_text(encoding="utf-8"))
    selections = jsonl_read(path / "selections.jsonl")
    assert len(selections) == len(groups) == len(grads) == result["completed_optimizer_steps"]
    assert all(len(r["ids"]) == 1 for r in selections)  # one G4 group per optimizer step here
    unexpected = []
    for sel, g in zip(selections, grads, strict=True):
        assert sel["optimizer_step"] + 1 == g["optimizer_step"]
        effective = signal_by_batch[sel["generation_batch"]] > 0
        nonzero = g["nonzero_gradient_elements"] > 0
        if effective != nonzero:
            unexpected.append(g["optimizer_step"])
    assert not mismatches and not unexpected
    assert counts["nonzero_advantage_completions"] == result["nonzero_advantage_completions"]
    counts["groups"] = len(groups)
    counts["optimizer_steps"] = len(grads)
    counts["nonzero_gradient_steps"] = sum(r["nonzero_gradient_elements"] > 0 for r in grads)
    counts["zero_gradient_steps"] = len(grads) - counts["nonzero_gradient_steps"]
    probe = sum(r.get("wall_seconds", 0) for r in costs if r["phase"] == "difficulty_probe")
    total = result["arm_total_through_checkpoint_seconds"]
    return {"run": path.name, "arm": path.name.split("-")[-2], "seed": cfg["seed"],
        **counts, "all_correct_group_fraction": counts["all_correct_groups"] / len(groups),
        "all_wrong_group_fraction": counts["all_wrong_groups"] / len(groups),
        "mixed_group_fraction": counts["mixed_groups"] / len(groups),
        "zero_advantage_completion_fraction": counts["zero_advantage_completions"] / len(rollouts),
        "truncation_completion_fraction": counts["truncated_completions"] / len(rollouts),
        "zero_gradient_step_fraction": counts["zero_gradient_steps"] / len(grads),
        "training_entry_to_saved_checkpoint_seconds": total, "initial_probe_seconds": probe,
        "effective_completions_per_training_entry_second": counts["effective_advantage_completions"] / total,
        "nonzero_gradient_steps_per_training_entry_second": counts["nonzero_gradient_steps"] / total,
        "reward_recomputation_mismatches": mismatches, "signal_gradient_disagreements": unexpected,
        "final_checkpoint_matches_result": checkpoint_hash(result["checkpoint"]) == result["checkpoint_sha256"],
        "endpoint_adapter_delta_l2": result["adapter_delta_l2"],
        "historical_per_step_weight_deltas": "not recorded; do not infer zero update from zero gradient"}


def main():
    runs = [audit_run(ROOT / "runs" / f"desktoprepair_20261106_152712-{arm}-s{seed}")
        for arm in ("C", "D", "E") for seed in (13, 29, 47)]
    totals = []
    fields = ["all_correct_groups", "all_wrong_groups", "mixed_groups", "groups", "completions",
        "zero_advantage_completions", "nonzero_advantage_completions", "effective_advantage_completions",
        "truncated_completions", "advantage_masked_by_truncation", "parse_failures", "optimizer_steps",
        "nonzero_gradient_steps", "zero_gradient_steps", "generated_tokens"]
    for arm in ("C", "D", "E"):
        rs = [r for r in runs if r["arm"] == arm]
        t = {"arm": arm, **{k: sum(r[k] for r in rs) for k in fields}}
        seconds = sum(r["training_entry_to_saved_checkpoint_seconds"] for r in rs)
        t.update(training_entry_seconds=seconds,
            effective_completions_per_second=t["effective_advantage_completions"] / seconds,
            nonzero_gradient_steps_per_second=t["nonzero_gradient_steps"] / seconds)
        totals.append(t)
    payload = {"historical_family": "desktoprepair_20261106_152712", "runs": runs, "by_arm": totals,
        "denominators": "groups are sampled prompt x4 rollouts; completions are generated answers; optimizer steps are actual updates. One group/step was verified, not assumed.",
        "effective_advantage_definition": "nonzero raw group-standardized advantage AND nonempty loss mask (EOS-terminated)",
        "mask_evidence_scope": "Historical masks reconstructed from recorded EOS/truncation plus pinned TRL 0.23.1; new study logs actual tensors.",
        "interpretation": "All recomputed rewards and signal-to-gradient alignments match. Zero scalar GRPO loss at ratio=1 does not imply zero derivative. Adam momentum can update weights on zero-gradient steps."}
    json_write(EXP / "historical_signal_audit.json", payload)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
