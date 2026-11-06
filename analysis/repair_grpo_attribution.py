"""Paired endpoint error accounting, without inferring arithmetic causes from a score."""
import json
from collections import Counter
from pathlib import Path
from reasoning_rl.runtime import ROOT,json_write,jsonl_write,jsonl_read

FAMILY="desktoprepair_20261106_152712"
OUT=ROOT/"reports/comparisons"/FAMILY


def category(row):
    return "truncated" if row["truncated"] else "nontruncated_parse_failure" if not row["parse_ok"] else "nontruncated_wrong_numeric_requires_review"


def main():
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    records,cases=[],[]
    for subset in ("comparison","confirmation"):
        for left,right in (("D","C"),("D","E")):
            for seed in (13,29,47):
                lpath=ROOT/"runs"/f"{FAMILY}-{left}-s{seed}-{subset}/validation/predictions.jsonl"
                rpath=ROOT/"runs"/f"{FAMILY}-{right}-s{seed}-{subset}/validation/predictions.jsonl"
                l,r=jsonl_read(lpath),jsonl_read(rpath)
                assert [x["id"] for x in l]==[x["id"] for x in r]
                losses=[(a,b) for a,b in zip(l,r,strict=True) if b["correct"] and not a["correct"]]
                gains=[(a,b) for a,b in zip(l,r,strict=True) if a["correct"] and not b["correct"]]
                breakdown=Counter(category(a) for a,b in losses)
                assert sum(breakdown.values())==len(losses)
                records.append({"subset":subset,"contrast":left+"-"+right,"seed":seed,"n":len(l),
                    "right_correct_left_wrong":len(losses),"right_wrong_left_correct":len(gains),
                    "net_correct_difference":len(gains)-len(losses),
                    "left_regression_surface_partition":dict(breakdown),
                    "regression_ids":[a["id"] for a,b in losses],"gain_ids":[a["id"] for a,b in gains]})
                if subset=="confirmation" and right=="C":
                    selected=[]
                    for key in ("truncated","nontruncated_parse_failure","nontruncated_wrong_numeric_requires_review"):
                        selected += [(a,b) for a,b in losses if category(a)==key][:1]
                    for a,b in selected:
                        cases.append({"subset":subset,"seed":seed,"id":a["id"],"D_prediction":a["prediction"],
                            "C_prediction":b["prediction"],"gold_answer":a["gold_answer"],"D_surface_category":category(a),
                            "D_source":lpath.relative_to(ROOT).as_posix(),"C_source":rpath.relative_to(ROOT).as_posix(),
                            "human_reviewed":False,"semantic_error_not_automatically_assigned":True})
    by_arm={arm:{"steps":sum(r["steps"] for r in summary["endpoint_rows"] if r["arm"]==arm),
                 "nonzero_gradient_steps":sum(r["nonzero_gradient_steps"] for r in summary["endpoint_rows"] if r["arm"]==arm),
                 "draw_weighted_zero_variance_rate":sum(r["zero_variance_rate"]*r["steps"] for r in summary["endpoint_rows"] if r["arm"]==arm)/sum(r["steps"] for r in summary["endpoint_rows"] if r["arm"]==arm)} for arm in ("C","D","E")}
    json_write(OUT/"grpo_error_attribution.json",{"records":records,"learning_signal_by_arm":by_arm,
        "scope":"Same-question surface outcome accounting, conditional on fixed development subsets; partitions are not causal/semantic diagnoses",
        "interpretation":"More optimizer steps need not yield more nonzero learning steps. Sampling, generated rewards, actual time and endpoints must be interpreted together; no matched-step extra experiment was performed."})
    jsonl_write(OUT/"grpo_confirmation_case_pack.jsonl",cases)
    print(json.dumps({"learning_signal":by_arm,"confirmation_DC":[r for r in records if r["subset"]=="confirmation" and r["contrast"]=="D-C"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
