"""Exact endpoint, budget, distribution and paired-question summaries; no model calls."""
import csv
import datetime as dt
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
import numpy as np
from reasoning_rl.runtime import ROOT,bootstrap,json_write,jsonl_read,sha256_file
from reasoning_rl.data import load_split

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"reports/comparisons"/FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_write(path,rows):
    if not rows:
        return
    with Path(path).open("w",encoding="utf-8-sig",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def metrics(label,subset):
    p=ROOT/"runs"/f"{FAMILY}-{label}-{subset}"/"validation/metrics.json"
    return read(p) if p.exists() else None


def predictions(label,subset):
    p=ROOT/"runs"/f"{FAMILY}-{label}-{subset}"/"validation/predictions.jsonl"
    return jsonl_read(p) if p.exists() and p.with_name("metrics.json").exists() else None


def main():
    bootstrap()
    OUT.mkdir(parents=True,exist_ok=True)
    plan=read(EXP/"sampling_frozen.json")
    decision=read(EXP/"sft_decision.json")
    cost_records=jsonl_read(EXP/"process_costs.jsonl")
    assert len({r["phase"] for r in cost_records})==len(cost_records)
    cost_map={r["phase"]:r for r in cost_records}
    rows,exposure,distributions,per_question=[],[],[],[]
    train={r["id"]:r for r in load_split("train","develop")}
    for seed in (13,29,47):
        for arm in ("C","D","E"):
            label=f"{arm}-s{seed}"
            directory=ROOT/"runs"/(FAMILY+"-"+label)
            result=read(directory/"training_result.json")
            contract=read(directory/"comparison_contract.json")
            config=contract["protocol"]
            costs=jsonl_read(directory/"costs.jsonl")
            grpo=next(r for r in costs if r["phase"]=="grpo_total")
            probes=[r for r in costs if r["phase"]=="difficulty_probe"]
            gradients=[r for r in costs if r["phase"]=="gradient_diagnostic"]
            probe_seconds=sum(r["wall_seconds"] for r in probes)
            maintenance=result["sampler_maintenance_seconds"]
            selections=jsonl_read(directory/"selections.jsonl")
            counts=Counter(i for event in selections for i in event["ids"])
            eff=[1/sum(v*v for v in event["weights"].values()) for event in selections]
            sampler=read(directory/"sampler_final.json")
            used_changes=sum(a["weights"]!=b["weights"] for a,b in zip(selections,selections[1:]))
            train_process=cost_map[label]["process_wall_seconds"]
            evaluate_process=cost_map[label+"-comparison"]["process_wall_seconds"]
            actual_total=train_process+evaluate_process
            cap=plan["cost_based_capacity"]["end_to_end_common_cap_seconds"]
            compare=metrics(label,"comparison")
            confirm=metrics(label,"confirmation")
            assert compare and compare["n"]==128
            eval_costs=jsonl_read(ROOT/"runs"/(FAMILY+"-"+label+"-comparison")/"costs.jsonl")
            rows.append({"run":directory.name,"arm":arm,"strategy":plan["arms"][arm],"seed":seed,
                "common_start_sha256":plan["common_start"]["checkpoint_sha256"],"pool_n":len(contract["pool_ids"]),
                "comparison_n":compare["n"],"comparison_correct":compare["correct"],"comparison_accuracy":compare["accuracy"],
                "comparison_parse_error_rate":compare["parse_error_rate"],"comparison_truncation_rate":compare["truncation_rate"],
                "comparison_mean_generated_tokens":compare["mean_generated_tokens"],
                "confirmation_n":confirm["n"] if confirm else None,"confirmation_correct":confirm["correct"] if confirm else None,
                "confirmation_accuracy":confirm["accuracy"] if confirm else None,
                "confirmation_parse_error_rate":confirm["parse_error_rate"] if confirm else None,
                "confirmation_truncation_rate":confirm["truncation_rate"] if confirm else None,
                "steps":result["completed_optimizer_steps"],"finite_gradient_steps":len(gradients),
                "nonzero_gradient_steps":sum(r["nonzero_gradient_elements"]>0 for r in gradients),
                "adapter_delta_l2":result["adapter_delta_l2"],"checkpoint_sha256":result["checkpoint_sha256"],
                "training_sampling_cap_seconds":config["grpo"]["max_wall_seconds"],
                "arm_internal_total_seconds":result["arm_total_through_checkpoint_seconds"],
                "arm_internal_unused_seconds":max(0,config["grpo"]["max_wall_seconds"]-result["arm_total_through_checkpoint_seconds"]),
                "arm_internal_overshoot_seconds":max(0,result["arm_total_through_checkpoint_seconds"]-config["grpo"]["max_wall_seconds"]),
                "training_process_seconds":train_process,"grpo_nested_seconds":grpo["wall_seconds"],
                "initial_and_refresh_probe_seconds":probe_seconds,"sampler_maintenance_seconds":maintenance,
                "rollout_optimization_other_inner_seconds":grpo["wall_seconds"]-probe_seconds-maintenance,
                "setup_serialization_process_overhead_seconds":train_process-grpo["wall_seconds"],
                "independent_comparison_process_seconds":evaluate_process,
                "independent_comparison_load_seconds":sum(r["wall_seconds"] for r in eval_costs if r["phase"]=="load_model"),
                "independent_comparison_generation_seconds":sum(r["wall_seconds"] for r in eval_costs if r["phase"]=="validation"),
                "actual_end_to_end_seconds":actual_total,"common_end_to_end_cap_seconds":cap,
                "end_to_end_unused_seconds":max(0,cap-actual_total),"end_to_end_overshoot_seconds":max(0,actual_total-cap),
                "complete_initial_probes":sum(r.get("probe_complete",False) and r.get("optimizer_step")==0 for r in probes),
                "complete_post_initial_full_probes":result["full_post_initial_probes"],
                "incomplete_probes":result["incomplete_probes"],"cache_rollout_update_events":result["cache_rollout_update_events"],
                "cache_expiry_events":sampler.get("expiry_events",0),
                "sampler_any_weight_change_events":result["sampler_weight_change_events"],
                "consecutive_used_distribution_change_events":used_changes,
                "initial_effective_pool_size":eff[0],"last_used_effective_pool_size":eff[-1],
                "mean_used_effective_pool_size":statistics.mean(eff),"min_used_effective_pool_size":min(eff),"max_used_effective_pool_size":max(eff),
                "final_unused_sampler_state_effective_pool_size":sampler["effective_pool_size"],
                "optimization_draws":sum(counts.values()),"optimization_unique_selected":len(counts),
                "optimization_coverage":len(counts)/len(contract["pool_ids"]),
                "zero_variance_rate":result["zero_variance_group_fraction"],"mean_train_reward":result["mean_reward"],
                "training_truncation_rate":result["training_truncation_rate"],"training_parse_error_rate":result["training_parse_error_rate"],
                "policy_generated_tokens":result["generated_tokens"],"policy_loss_input_tokens":result["processed_tokens"],
                "probe_generated_tokens":result["probe_generated_tokens"],"probe_prompt_tokens":result["probe_prompt_tokens"],
                "peak_allocated_gib":grpo["peak_allocated_bytes"]/2**30,"peak_reserved_gib":grpo["peak_reserved_bytes"]/2**30,
                "device_elapsed_seconds":grpo["device_elapsed_seconds"]})
            for identifier in contract["pool_ids"]:
                exposure.append({"run":directory.name,"arm":arm,"seed":seed,"id":identifier,
                    "selected_count":counts[identifier],
                    "expected_selected_count":sum(event["weights"][identifier]*len(event["ids"]) for event in selections),
                    "question_words":len(train[identifier]["question"].split()),
                    "solution_contains_division_symbol":"/" in train[identifier]["solution"],
                    "solution_contains_multiplication_symbol":"*" in train[identifier]["solution"],
                    "feature_scope":"Overlapping literal text features, not human-reviewed math-topic labels"})
            for event,effn in zip(selections,eff,strict=True):
                distributions.append({"run":directory.name,"arm":arm,"seed":seed,"optimizer_step":event["optimizer_step"],
                    "generation_batch":event["generation_batch"],"effective_pool_size":effn,
                    "sampler_updates":event["sampler_updates"],"max_weight":max(event["weights"].values()),"min_weight":min(event["weights"].values())})
    labels=["A",decision["selected_candidate"]] if decision["selected_candidate"] else ["A"]
    labels += [f"{arm}-s{seed}" for seed in (13,29,47) for arm in ("C","D","E")]
    by_subset={}
    for subset in ("comparison","confirmation"):
        by_subset[subset]={}
        for label in labels:
            observed=predictions(label,subset)
            if observed is None:
                continue
            by_subset[subset][label]=observed
            for record in observed:
                per_question.append({"subset":subset,"endpoint":label,**record})
    effects=[]
    rng=np.random.default_rng(20261106)
    for subset,observed in by_subset.items():
        for left,right in (("D","C"),("D","E"),("E","C")):
            if not all(f"{arm}-s{seed}" in observed for arm in (left,right) for seed in (13,29,47)):
                continue
            deltas=[]
            for seed in (13,29,47):
                l,r=observed[f"{left}-s{seed}"],observed[f"{right}-s{seed}"]
                assert [p["id"] for p in l]==[p["id"] for p in r]
                deltas.append(np.array([float(a["correct"])-float(b["correct"]) for a,b in zip(l,r,strict=True)]))
            matrix=np.stack(deltas)
            per_seed=matrix.mean(axis=1)*100
            # Resample questions jointly across the three fixed paired seeds.
            question_effect=matrix.mean(axis=0)
            draws=rng.integers(0,len(question_effect),size=(20000,len(question_effect)))
            boot=question_effect[draws].mean(axis=1)*100
            effects.append({"subset":subset,"contrast":left+"-"+right,"seeds":[13,29,47],
                "paired_seed_differences_pp":per_seed.tolist(),"mean_difference_pp":float(per_seed.mean()),
                "sample_seed_sd_pp":float(per_seed.std(ddof=1)),
                "question_paired_bootstrap95_pp":np.quantile(boot,[0.025,0.975]).tolist(),
                "interval_scope":"IID-question resampling conditional on these3 fixed seeds and this already developer-accessible validation subset; within-validation similar questions may violate IID; not seed/population/generalization uncertainty"})
    aggregates={}
    for subset,observed in by_subset.items():
        aggregates[subset]={}
        for arm in ("C","D","E"):
            scores=[sum(r["correct"] for r in observed[f"{arm}-s{seed}"])/128*100
                    for seed in (13,29,47) if f"{arm}-s{seed}" in observed]
            aggregates[subset][arm]={"n_seeds":len(scores),"seed_accuracy_pct":scores,
                "mean_accuracy_pct":statistics.mean(scores) if scores else None,
                "sample_seed_sd_pct":statistics.stdev(scores) if len(scores)>1 else None}
    session=read(EXP/"session.json")
    started=dt.datetime.fromisoformat(session["started_utc"]).timestamp()
    audit_dir=ROOT/"runs"/(FAMILY+"-sft-contract")
    audit_seconds=(audit_dir/"audit_result.json").stat().st_mtime-read(audit_dir/"provenance.json")["created_unix"]
    summary={"family":FAMILY,"generated_unix":time.time(),"scope":"Development comparison128 and one-stage disjoint confirmation128; formal holdouts sealed",
        "baseline":{subset:{label:metrics(label,subset) for label in labels[:len(labels)-9]} for subset in ("comparison","confirmation")},
        "aggregates":aggregates,"paired_effects":effects,"endpoint_rows":rows,
        "cost_accounting":{"session_started_utc":session["started_utc"],"session_deadline_utc":session["deadline_utc"],
            "session_elapsed_to_summary_seconds":time.time()-started,"all_recorded_child_process_seconds":sum(r["process_wall_seconds"] for r in cost_records),
            "sft_contract_from_run_creation_seconds":audit_seconds,"sum_nine_training_process_seconds":sum(r["training_process_seconds"] for r in rows),
            "sum_nine_independent_comparison_seconds":sum(r["independent_comparison_process_seconds"] for r in rows),
            "sum_nine_probe_seconds_nested_in_training":sum(r["initial_and_refresh_probe_seconds"] for r in rows),
            "sum_nine_cache_maintenance_seconds_nested_in_training":sum(r["sampler_maintenance_seconds"] for r in rows),
            "confirmation_child_process_seconds":sum(r["process_wall_seconds"] for r in cost_records if r["phase"].endswith("-confirmation")),
            "accounting_note":"Probe/maintenance are nested in training, not added again. Child process time includes interpreter/setup/serialization; session wall also includes diagnosis, engineering, analysis and handoff. GPU event elapsed is not active-kernel time or FLOPs."},
        "source":{"frozen_code_sha256":plan["code_sha256"],"frozen_source_manifest_sha256":plan["frozen_source_manifest_sha256"],
            "common_start":plan["common_start"],"data_manifest_sha256":plan["data_manifest_sha256"],
            "development_ids_sha256":plan["development_ids_sha256"],"public_model_verification_sha256":plan["public_model_verification_sha256"]},
        "all_nine_comparison_complete":all(r["comparison_n"]==128 for r in rows),
        "all_nine_confirmation_complete":all(r["confirmation_n"]==128 for r in rows),
        "additional_full_pool_dynamic_baseline_run":False,"formal_test_or_svamp_opened":False}
    csv_write(OUT/"standalone_endpoints.csv",rows)
    csv_write(OUT/"item_exposure.csv",exposure)
    csv_write(OUT/"observed_sampler_points.csv",distributions)
    csv_write(OUT/"all_endpoint_predictions.csv",per_question)
    json_write(OUT/"summary.json",summary)
    plot(summary,distributions)
    print(json.dumps({"aggregates":aggregates,"paired_effects":effects,"cost_accounting":summary["cost_accounting"]},ensure_ascii=False,indent=2))


def plot(summary,points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":0.18,"axes.axisbelow":True,"figure.facecolor":"white","axes.facecolor":"white"})
    colors={"C":"#2166ac","D":"#d06c00","E":"#7154a1"}
    markers={13:"o",29:"s",47:"^"}
    offsets={13:-0.16,29:0,47:0.16}
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout="constrained")
    base_labels=list(summary["baseline"]["comparison"])
    for ax,subset,title in zip(axes,("comparison","confirmation"),("Comparison128 (used for development)","Confirmation128 (one stage; not a test)"),strict=True):
        for x,label in enumerate(base_labels):
            m=summary["baseline"][subset][label]
            if m:
                ax.scatter([x],[100*m["accuracy"]],color="#252525" if label=="A" else "#28784b",marker="D",s=70,zorder=5)
                ax.annotate(f"{m['correct']}/128",(x,100*m["accuracy"]),xytext=(0,9),textcoords="offset points",ha="center",fontsize=9)
        for x,arm in enumerate(("C","D","E"),start=len(base_labels)):
            values=[]
            for row in summary["endpoint_rows"]:
                if row["arm"]!=arm or row[subset+"_accuracy"] is None:
                    continue
                value=100*row[subset+"_accuracy"]
                ax.scatter([x+offsets[row["seed"]]],[value],color=colors[arm],marker=markers[row["seed"]],s=65,zorder=5)
                ax.annotate(str(row[subset+"_correct"]),(x+offsets[row["seed"]],value),xytext=(0,10 if row["seed"]!=29 else -16),textcoords="offset points",ha="center",fontsize=8)
                values.append(value)
            if values:
                ax.scatter([x],[statistics.mean(values)],color="black",marker="_",s=320,linewidth=2,zorder=6)
        ax.set(title=title,ylabel="Accuracy (%)",ylim=(0,100),xticks=list(range(len(base_labels)+3)),xticklabels=base_labels+["C uniform","D cache","E static"])
        ax.set_xlim(-0.5,len(base_labels)+2.5)
    handles=[plt.Line2D([],[],color="black",marker=markers[s],linestyle="None",label=f"seed{s}") for s in (13,29,47)]
    handles += [plt.Line2D([],[],color="black",marker="_",linestyle="None",markersize=16,label="mean of3 seeds")]
    axes[1].legend(handles=handles,loc="lower right",fontsize=9)
    fig.suptitle("Independent reload endpoints | pinned model and data; 600 s training/sampling allocation",fontsize=12)
    fig.savefig(OUT/"standalone_comparison.png",dpi=180);fig.savefig(OUT/"standalone_comparison.svg");plt.close(fig)
    rows=summary["endpoint_rows"]
    fig,ax=plt.subplots(figsize=(13,7.5),layout="constrained")
    fields=[("setup_serialization_process_overhead_seconds","Process/setup/save","#bcc4cc"),
            ("initial_and_refresh_probe_seconds","Initial probe","#d58b22"),
            ("sampler_maintenance_seconds","Sampler maintenance","#7154a1"),
            ("rollout_optimization_other_inner_seconds","Rollout/optimization/other inner cost","#3977b3"),
            ("independent_comparison_process_seconds","Independent reload+evaluation","#43886f")]
    left=np.zeros(len(rows))
    for key,label,color in fields:
        values=np.array([r[key] for r in rows])
        assert (values>=-1e-6).all(),key
        ax.barh(range(len(rows)),values,left=left,color=color,label=label,height=.72)
        left+=values
    for y,row in enumerate(rows):
        assert abs(left[y]-row["actual_end_to_end_seconds"])<1e-6
        ax.text(left[y]+8,y,f"{left[y]:.1f}s; unused {row['end_to_end_unused_seconds']:.1f}s",va="center",fontsize=8)
    cap=rows[0]["common_end_to_end_cap_seconds"]
    ax.axvline(cap,color="#333333",linestyle="--",linewidth=1,label=f"common total cap {cap} s")
    ax.set(yticks=range(len(rows)),yticklabels=[f"{r['arm']} seed{r['seed']}" for r in rows],xlabel="Measured seconds (nonoverlapping components)",
           title="Common 600 s training/sampling + 540 s evaluation reserve + 30 s process reserve",xlim=(0,cap+90))
    ax.invert_yaxis();ax.legend(loc="upper left",bbox_to_anchor=(0,-0.12),ncol=3,fontsize=9)
    fig.savefig(OUT/"cost_ledger.png",dpi=180);fig.savefig(OUT/"cost_ledger.svg");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout="constrained")
    pool=rows[0]["pool_n"]
    for ax,seed in zip(axes,(13,29,47),strict=True):
        for arm in ("C","D","E"):
            observed=[p for p in points if p["seed"]==seed and p["arm"]==arm]
            ax.scatter([p["optimizer_step"] for p in observed],[p["effective_pool_size"] for p in observed],s=11,alpha=.7,color=colors[arm],label=arm)
        ax.set(title=f"seed{seed}",xlabel="Optimizer step at actual selection",ylabel="Effective pool 1/sum(q^2)",ylim=(0,pool+2))
        ax.legend(loc="lower right")
    fig.suptitle("Recorded sampling distributions only | no interpolated validation trajectory",fontsize=12)
    fig.savefig(OUT/"effective_pool_observed.png",dpi=180);fig.savefig(OUT/"effective_pool_observed.svg");plt.close(fig)
    sft=read(OUT/"sft_attribution.json")
    baseline=summary["baseline"]["comparison"]["A"]
    names=["A"]+[r["candidate"] for r in sft["candidates"]]
    acc=[baseline["accuracy"]*100]+[r["correct"]/128*100 for r in sft["candidates"]]
    trunc=[baseline["truncation_rate"]*100]+[r["truncated"]/128*100 for r in sft["candidates"]]
    fig,ax=plt.subplots(figsize=(10,5.6),layout="constrained")
    x=np.arange(len(names))
    ax.bar(x-.18,acc,width=.35,color="#3977b3",label="Correct/128")
    ax.bar(x+.18,trunc,width=.35,color="#d58b22",label="Truncated/128")
    for i,(a,t) in enumerate(zip(acc,trunc,strict=True)):
        ax.text(i-.18,a+1.3,f"{round(a*128/100)}/128",ha="center",fontsize=9)
        ax.text(i+.18,t+1.3,f"{round(t*128/100)}",ha="center",fontsize=9)
    ax.set(xticks=x,xticklabels=names,ylim=(0,100),ylabel="Percent of fixed 128 questions",title="Bounded SFT diagnosis: identical masks, prompt, full targets and decoding")
    ax.legend(loc="upper left",bbox_to_anchor=(0,-0.12),ncol=2,fontsize=10)
    fig.savefig(OUT/"sft_strength.png",dpi=180);fig.savefig(OUT/"sft_strength.svg");plt.close(fig)


if __name__=="__main__":
    main()
