"""Summarize every fixed training endpoint without consuming development scores."""
from collections import defaultdict
from pathlib import Path
import json
import sys
import time
from reasoning_rl.runtime import ROOT,json_write,jsonl_read
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"analysis"))
from filter_common import EXP,FAMILY,read
from audit_filter_artifacts import window


def main():
    assert (EXP/"all_training_endpoints_fixed.json").exists()
    costs=read(EXP/"endpoint_costs.json")
    rows=[]
    for seed in (13,29,47):
        for arm in ("U","F"):
            label=f"{arm}-s{seed}"
            cost=costs[label]
            directory=ROOT/"runs"/cost["training_run"]
            groups=defaultdict(list)
            for r in jsonl_read(directory/"rollouts.jsonl"):
                groups[r["generation_batch"]].append(r)
            ledger=jsonl_read(directory/"costs.jsonl")
            grads=[r for r in ledger if r["phase"]=="gradient_diagnostic"]
            updates=[r for r in ledger if r["phase"]=="optimizer_weight_update"]
            paid=cost["training_process_seconds"]+cost["initial_probe_standalone_debit_seconds"]+cost["parent_preparation_and_training_audit_seconds"]
            metrics=window(groups.values())
            rows.append({"label":label,"arm":arm,"seed":seed,**metrics,
                "nonzero_gradient_steps":sum(g["nonzero_gradient_elements"]>0 for g in grads),
                "zero_gradient_weight_change_steps":sum(u["zero_gradient_step"] and u["actual_changed_weight_elements"]>0 for u in updates),
                "paid_training_seconds":paid,
                "effective_advantage_completions_per_paid_training_second":metrics["effective_advantage_completions"]/paid,
                "unique_training_items":len({g[0]["id"] for g in groups.values()})})
    arms=[]
    for arm in ("U","F"):
        rs=[r for r in rows if r["arm"]==arm]
        n=sum(r["groups"] for r in rs)
        mixed=sum(r["mixed_groups"] for r in rs)
        effective=sum(r["effective_advantage_completions"] for r in rs)
        seconds=sum(r["paid_training_seconds"] for r in rs)
        arms.append({"arm":arm,"groups":n,"mixed_groups":mixed,"zero_variance_group_fraction":1-mixed/n,
            "nonzero_gradient_steps":sum(r["nonzero_gradient_steps"] for r in rs),
            "effective_advantage_completions":effective,"paid_training_seconds":seconds,
            "effective_advantage_completions_per_paid_training_second":effective/seconds,
            "stochastic_training_success_fraction":sum(r["training_stochastic_success_fraction"]*r["groups"] for r in rs)/n})
    record={"recorded_unix":time.time(),"all_six_training_endpoints_fixed":True,
        "completed_main_endpoint_evaluations_at_collection":len(list((ROOT/"runs").glob(FAMILY+"-r1-*-development/validation/metrics.json"))),
        "development_scores_read":False,"scope":"Training/probe-cost mechanism only; no inference about held-out accuracy",
        "endpoints":rows,"by_arm":arms}
    json_write(EXP/"training_mechanism_before_evaluation.json",record)
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
