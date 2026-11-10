"""Recompute complete U/F outcomes, tensor signal, optimizer drift and paid costs."""
from collections import Counter, defaultdict
import csv
import json
import math
import statistics
import numpy as np
from reasoning_rl.runtime import ROOT, json_write, jsonl_write, jsonl_read, checkpoint_hash, sha256_file
from reasoning_rl.scoring import score_answer, group_advantages
from reasoning_rl.data import load_split
from filter_common import EXP, FAMILY, read

OUT = ROOT / "reports/comparisons" / FAMILY


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint(label, cost, probe):
    directory = ROOT / "runs" / cost["training_run"]
    result = read(directory / "training_result.json")
    rollouts = jsonl_read(directory / "rollouts.jsonl")
    groups = defaultdict(list)
    counts = Counter({k: 0 for k in ("all_correct_groups", "all_wrong_groups", "mixed_groups")})
    for r in rollouts:
        score = score_answer(r["prediction"], r["gold_answer"])
        assert score.as_dict() == {k: r[k] for k in score.as_dict()}
        assert float(score.correct) == r["reward"]
        assert len(r["completion_token_ids"]) == r["generated_tokens"]
        groups[(r["generation_batch"], r["group_within_batch"])].append(r)
        counts["parse_failures"] += int(not r["parse_ok"])
        counts["format_failures"] += int(not r["format_ok"])
        counts["truncated_completions"] += int(r["truncated"])
    signals = jsonl_read(directory / "rollout_group_signal.jsonl")
    for key, rs in groups.items():
        assert len(rs) == 4 and len({r["id"] for r in rs}) == 1
        n = sum(r["reward"] for r in rs)
        counts["all_correct_groups" if n == 4 else "all_wrong_groups" if n == 0 else "mixed_groups"] += 1
        adv = group_advantages([r["reward"] for r in rs])
        counts["zero_advantage_completions"] += sum(a == 0 for a in adv)
        counts["nonzero_advantage_completions"] += sum(a != 0 for a in adv)
        counts["effective_advantage_completions"] += sum(a != 0 and not r["truncated"] for a,r in zip(adv,rs,strict=True))
        actual = signals[key[0]]
        assert actual["generation_batch"] == key[0] and actual["mask_and_advantage_checks_passed"]
        assert actual["raw_rewards"] == [r["reward"] for r in rs]
        assert all(abs(a-b) < 2e-6 for a,b in zip(actual["actual_advantages"],adv,strict=True))
        assert actual["actual_loss_mask_tokens_per_completion"] == [0 if r["truncated"] else r["generated_tokens"] for r in rs]
    ledger = jsonl_read(directory / "costs.jsonl")
    grads = [r for r in ledger if r["phase"] == "gradient_diagnostic"]
    updates = [r for r in ledger if r["phase"] == "optimizer_weight_update"]
    steps = result["completed_optimizer_steps"]
    assert len(groups) == len(signals) == len(grads) == len(updates) == steps
    assert len(jsonl_read(directory / "loss_signal.jsonl")) == steps*4
    assert all(g["finite_gradients"] and not g["frozen_parameter_gradients"] for g in grads)
    for g, u in zip(grads, updates, strict=True):
        assert g["optimizer_step"] == u["optimizer_step"]
        assert u["zero_gradient_step"] == (g["nonzero_gradient_elements"] == 0)
    counts["nonzero_gradient_steps"] = sum(g["nonzero_gradient_elements"] > 0 for g in grads)
    counts["zero_gradient_steps"] = steps-counts["nonzero_gradient_steps"]
    counts["zero_gradient_steps_with_actual_weight_change"] = sum(u["zero_gradient_step"] and u["actual_changed_weight_elements"] > 0 for u in updates)
    counts["all_steps_with_actual_weight_change"] = sum(u["actual_changed_weight_elements"] > 0 for u in updates)
    assert counts["effective_advantage_completions"] == result["actual_effective_advantage_completions"]
    assert counts["nonzero_advantage_completions"] == result["nonzero_advantage_completions"]
    assert checkpoint_hash(ROOT / "checkpoints" / cost["training_run"]) == result["checkpoint_sha256"]
    evaluation = ROOT / "runs" / (cost["training_run"]+"-development")
    preds = jsonl_read(evaluation / "validation/predictions.jsonl")
    metrics = read(evaluation / "validation/metrics.json")
    assert read(evaluation / "evaluation_protocol.json")["adapter_sha256"] == result["checkpoint_sha256"]
    assert metrics["correct"] == sum(r["correct"] for r in preds) == cost["correct"]
    for r in preds:
        assert score_answer(r["prediction"], r["gold_answer"]).correct == r["correct"]
    counts["groups"] = len(groups)
    counts["completions"] = len(rollouts)
    counts["optimizer_steps"] = steps
    paid_train = cost["training_process_seconds"] + cost["initial_probe_standalone_debit_seconds"] + cost["parent_preparation_and_training_audit_seconds"]
    paid_end = cost["charged_end_to_end_seconds"]
    assigned_probe_tokens = probe["generated_tokens"] if cost["arm"] == "F" else 0
    row = {"label": label, "arm": cost["arm"], "seed": cost["seed"], "correct": metrics["correct"],
        "n": 128, "accuracy": metrics["accuracy"], **counts,
        "zero_variance_group_fraction": (counts["all_correct_groups"]+counts["all_wrong_groups"])/len(groups),
        "training_parse_failure_fraction": counts["parse_failures"]/len(rollouts),
        "training_truncation_fraction": counts["truncated_completions"]/len(rollouts),
        "training_process_seconds": cost["training_process_seconds"],
        "probe_standalone_debit_seconds": cost["initial_probe_standalone_debit_seconds"],
        "paid_training_through_checkpoint_seconds": paid_train,
        "evaluation_process_seconds": cost["evaluation_process_seconds"],
        "parent_bookkeeping_seconds": cost["parent_preparation_and_training_audit_seconds"]+cost["parent_evaluation_preparation_and_audit_seconds"],
        "charged_end_to_end_seconds": paid_end, "common_cap_seconds": cost["common_cap_seconds"],
        "unused_common_cap_seconds": cost["unused_common_cap_seconds"],
        "effective_completions_per_paid_training_second": counts["effective_advantage_completions"]/paid_train,
        "effective_completions_per_paid_end_to_end_second": counts["effective_advantage_completions"]/paid_end,
        "nonzero_gradient_steps_per_paid_training_second": counts["nonzero_gradient_steps"]/paid_train,
        "nonzero_gradient_steps_per_paid_end_to_end_second": counts["nonzero_gradient_steps"]/paid_end,
        "training_generated_tokens": sum(r["generated_tokens"] for r in rollouts),
        "assigned_probe_generated_tokens": assigned_probe_tokens,
        "evaluation_generated_tokens": metrics["generated_tokens"],
        "standalone_total_generated_tokens": result["generated_tokens"]+assigned_probe_tokens+metrics["generated_tokens"],
        "eval_parse_failures": sum(not r["parse_ok"] for r in preds),
        "eval_truncations": sum(r["truncated"] for r in preds),
        "training_unique_items": len({r["id"] for r in rollouts}),
        "adapter_delta_l2": result["adapter_delta_l2"], "checkpoint_sha256": result["checkpoint_sha256"],
        "atomic_training_budget_overshoot_seconds": result["wall_budget_overshoot_seconds"],
        "peak_allocated_bytes": max(r.get("peak_allocated_bytes") or 0 for r in ledger),
        "peak_reserved_bytes": max(r.get("peak_reserved_bytes") or 0 for r in ledger)}
    assert row["training_generated_tokens"] == result["generated_tokens"]
    exposure = Counter(r["id"] for r in rollouts)
    return row, preds, [{"label": label, "id": i, "training_completions": n, "raw_s2_probe_successes": probe["raw_success_counts"][i]} for i,n in exposure.items()]


def plots(rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    colors = {"U": "#2874A6", "F": "#14866D"}
    fig, ax = plt.subplots(figsize=(8.3, 4.6), layout="constrained")
    for j, seed in enumerate((13,29,47)):
        values = [next(r for r in rows if r["seed"] == seed and r["arm"] == a)["accuracy"]*100 for a in ("U","F")]
        offset = (j-1)*.07
        ax.plot([0+offset,1+offset],values,"o-",color=["#366C92","#9B6A2F","#7163A1"][j],label=f"seed {seed}",lw=1.8,ms=7)
        for x, y, a in zip((offset,1+offset),values,("U","F"),strict=True):
            correct = next(r for r in rows if r["seed"] == seed and r["arm"] == a)["correct"]
            ax.annotate(f"{correct}/128",(x,y),xytext=(7,(j-1)*10+4),textcoords="offset points",fontsize=9)
    ax.axhline(summary["s2_baseline"]["accuracy"]*100,color="#777777",ls="--",lw=1,label="fixed S2 baseline")
    ax.set_xticks([0,1],["U: uniform full pool","F: uniform prefiltered pool"])
    ax.set_xlim(-.3,1.5)
    ys=[r["accuracy"]*100 for r in rows]+[summary["s2_baseline"]["accuracy"]*100]
    ax.set_ylim(math.floor(min(ys))-2, math.ceil(max(ys))+3)
    ax.set_ylabel("Development accuracy (%)")
    ax.set_title("Fixed-budget U/F endpoints: all three paired seeds\nReused development128; independent process reload; no formal test")
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="upper left",ncol=2,fontsize=9)
    fig.savefig(OUT/"accuracy_paired.png",dpi=170)
    plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    arms=summary["by_arm"]
    bottom=np.zeros(2)
    for key,label,color in (("all_correct_groups","all correct","#9BB5CA"),("all_wrong_groups","all wrong","#D5D9DF"),("mixed_groups","mixed: nonzero advantage","#D48D36")):
        values=np.array([a[key]/a["groups"]*100 for a in arms])
        axes[0].bar([0,1],values,bottom=bottom,label=label,color=color,width=.6)
        for x,v,b in zip((0,1),values,bottom,strict=True):
            if v>4:
                axes[0].text(x,b+v/2,f"{v:.1f}%",ha="center",va="center",fontsize=9)
        bottom+=values
    axes[0].set(xticks=[0,1],xticklabels=["U","F"],ylabel="Rollout groups (%)",ylim=(0,100),title="Actual training reward groups")
    axes[0].legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.08))
    rates=[a["effective_completions_per_paid_end_to_end_second"] for a in arms]
    axes[1].bar([0,1],rates,color=[colors[a] for a in ("U","F")],width=.6)
    for x,v in enumerate(rates):
        axes[1].text(x,v,f"{v:.4f}",ha="center",va="bottom")
    axes[1].set(xticks=[0,1],xticklabels=["U","F"],ylabel="Effective advantage completions / second",title="Includes full probe cost and reload evaluation",ylim=(0,max(rates)*1.25))
    bottom=np.zeros(2)
    for field,label,color in (("training_process_seconds","training process","#729DBB"),("probe_standalone_debit_seconds","paid probe debit","#D48D36"),("evaluation_process_seconds","reload evaluation","#75AD9A"),("parent_bookkeeping_seconds","parent bookkeeping","#B7B5B4")):
        values=np.array([a[field]/3 for a in arms])
        axes[2].bar([0,1],values,bottom=bottom,label=label,color=color,width=.6)
        bottom+=values
    cap=rows[0]["common_cap_seconds"]
    axes[2].axhline(cap,color="#333333",ls="--",lw=1)
    axes[2].set(xticks=[0,1],xticklabels=["U","F"],ylabel="Mean charged endpoint seconds",ylim=(0,cap*1.1),title=f"Common upper cap: {cap:.0f} s")
    axes[2].legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.08))
    fig.savefig(OUT/"mechanism_and_cost.png",dpi=170)
    plt.close(fig)


def main():
    assert read(ROOT / "environment" / (FAMILY+"-status.json"))["status"] == "completed_three_paired_seed_development"
    OUT.mkdir(parents=True, exist_ok=True)
    plan=read(EXP/"protocol_frozen.json")
    costs=read(EXP/"endpoint_costs.json")
    probe=read(ROOT/"runs"/(FAMILY+"-probe")/"probe_result.json")
    rows, predictions, exposure=[],{},[]
    for seed in (13,29,47):
        for arm in ("U","F"):
            label=f"{arm}-s{seed}"
            row,preds,items=endpoint(label,costs[label],probe)
            rows.append(row)
            predictions[label]={r["id"]:r for r in preds}
            exposure.extend(items)
    dev_ids=read(EXP/"development_ids.json")["comparison"]
    assert all(list(p)==dev_ids for p in predictions.values())
    s2=read(ROOT/"runs"/(FAMILY+"-S2-development")/"validation/metrics.json")
    paired=[]
    deltas=[]
    for seed in (13,29,47):
        u=next(r for r in rows if r["seed"]==seed and r["arm"]=="U")
        f=next(r for r in rows if r["seed"]==seed and r["arm"]=="F")
        paired.append({"seed":seed,"U_correct":u["correct"],"F_correct":f["correct"],"F_minus_U_correct":f["correct"]-u["correct"],
            "F_minus_U_pp":(f["correct"]-u["correct"])/128*100})
        deltas.append([int(predictions[f"F-s{seed}"][i]["correct"])-int(predictions[f"U-s{seed}"][i]["correct"]) for i in dev_ids])
    by_arm=[]
    for arm in ("U","F"):
        rs=[r for r in rows if r["arm"]==arm]
        sum_keys=["groups","completions","optimizer_steps","all_correct_groups","all_wrong_groups","mixed_groups",
            "zero_advantage_completions","nonzero_advantage_completions","effective_advantage_completions",
            "nonzero_gradient_steps","zero_gradient_steps","zero_gradient_steps_with_actual_weight_change",
            "all_steps_with_actual_weight_change","truncated_completions","parse_failures","format_failures",
            "training_generated_tokens","assigned_probe_generated_tokens","evaluation_generated_tokens","standalone_total_generated_tokens",
            "training_process_seconds","probe_standalone_debit_seconds","evaluation_process_seconds","parent_bookkeeping_seconds",
            "paid_training_through_checkpoint_seconds","charged_end_to_end_seconds"]
        a={"arm":arm,**{k:sum(r[k] for r in rs) for k in sum_keys}}
        a.update(mean_accuracy=statistics.mean(r["accuracy"] for r in rs),
            seed_accuracy_sample_sd=statistics.stdev(r["accuracy"] for r in rs),
            zero_variance_group_fraction=(a["all_correct_groups"]+a["all_wrong_groups"])/a["groups"],
            effective_completions_per_paid_training_second=a["effective_advantage_completions"]/a["paid_training_through_checkpoint_seconds"],
            effective_completions_per_paid_end_to_end_second=a["effective_advantage_completions"]/a["charged_end_to_end_seconds"],
            nonzero_gradient_steps_per_paid_end_to_end_second=a["nonzero_gradient_steps"]/a["charged_end_to_end_seconds"])
        by_arm.append(a)
    qdiff=np.mean(np.array(deltas,dtype=float),axis=0)
    rng=np.random.default_rng(20261110)
    boot=np.array([qdiff[rng.integers(0,128,128)].mean()*100 for _ in range(10000)])
    errors=Counter()
    cases=[]
    val={r["id"]:r for r in load_split("validation","evaluate_validation",seed=20261105)}
    for seed in (13,29,47):
        for i in dev_ids:
            u,f=predictions[f"U-s{seed}"][i],predictions[f"F-s{seed}"][i]
            if u["correct"]==f["correct"]:
                continue
            direction="U_correct_F_wrong" if u["correct"] else "F_correct_U_wrong"
            wrong=f if u["correct"] else u
            category="truncated" if wrong["truncated"] else "parse_failure" if not wrong["parse_ok"] else "untruncated_wrong_numeric_requires_review"
            errors[direction]+=1
            errors[direction+"__"+category]+=1
            cases.append({"seed":seed,"id":i,"direction":direction,"automatic_category":category,
                "question":val[i]["question"],"gold_answer":val[i]["answer"],
                "U_prediction":u["prediction"],"F_prediction":f["prediction"],"human_reviewed":False})
    summary={"family":FAMILY,"scope":"completed three paired seeds on reused DEVELOPMENT128; GSM8K test/SVAMP sealed",
        "s2_baseline":s2,"endpoints":rows,"by_arm":by_arm,"paired_differences":paired,
        "mean_paired_F_minus_U_pp":statistics.mean(p["F_minus_U_pp"] for p in paired),
        "paired_difference_seed_sample_sd_pp":statistics.stdev(p["F_minus_U_pp"] for p in paired),
        "conditional_question_bootstrap_95_pp":np.quantile(boot,[.025,.975]).tolist(),
        "bootstrap_scope":"Question resampling jointly over the same three fixed seeds; NOT seed uncertainty, NOT independent test generalization; validation near similarity can violate iid.",
        "probe":{"n_candidates":len(probe["candidate_ids"]),"n_retained":len(probe["retained_ids"]),"raw_success_histogram":probe["raw_success_histogram"],
            "generated_tokens":probe["generated_tokens"],"parse_failures":probe["parse_failures"],"truncations":probe["truncations"],
            "cost":plan["probe_cost"],"single_fixed_probe_seed":20261110},
        "paired_error_counts":dict(errors),"protocol_sha256":sha256_file(EXP/"protocol_frozen.json"),
        "cost_notes":"Full shared probe process debited per F in primary endpoint cost. Physical shared probe is counted once in session total; amortization is descriptive, not a separately trained equal-budget result."}
    csv_write(OUT/"standalone_endpoints.csv",rows)
    csv_write(OUT/"paired_differences.csv",paired)
    csv_write(OUT/"item_exposure.csv",exposure)
    jsonl_write(OUT/"paired_error_cases.jsonl",cases)
    json_write(OUT/"summary.json",summary)
    all_preds=[{"label":label,**r} for label,p in predictions.items() for r in p.values()]
    jsonl_write(OUT/"all_endpoint_predictions.jsonl",all_preds)
    plots(rows,summary)
    print(json.dumps({"paired":paired,"mean_difference_pp":summary["mean_paired_F_minus_U_pp"],
        "conditional_bootstrap95_pp":summary["conditional_question_bootstrap_95_pp"],"by_arm":by_arm},indent=2))


if __name__=="__main__":
    main()
