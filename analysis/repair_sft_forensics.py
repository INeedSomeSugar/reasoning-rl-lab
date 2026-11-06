"""Recompute SFT diagnostic attribution from saved predictions; no new model calls."""
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from reasoning_rl.runtime import ROOT,json_write,jsonl_write,jsonl_read,sha256_file
from reasoning_rl.data import load_split
from reasoning_rl.scoring import score_answer

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"reports/comparisons"/FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def function_digest(path,name):
    module=ast.parse(path.read_text(encoding="utf-8"))
    function=next(n for n in module.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    decision=read(EXP/"sft_decision.json")
    val={r["id"]:r for r in load_split("validation","evaluate_validation",seed=20261105)}
    baseline=jsonl_read(ROOT/"runs"/(FAMILY+"-A-comparison")/"validation/predictions.jsonl")
    by_a={r["id"]:r for r in baseline}
    rows,details,cases=[],[],[]
    fingerprints=[]
    for candidate in decision["candidates"]:
        label=candidate["name"]
        directory=ROOT/"runs"/(FAMILY+"-"+label)
        predictions=jsonl_read(ROOT/"runs"/(FAMILY+"-"+label+"-comparison")/"validation/predictions.jsonl")
        assert [r["id"] for r in baseline]==[r["id"] for r in predictions]
        assert all(score_answer(r["prediction"],r["gold_answer"]).correct==r["correct"] for r in predictions)
        lost=[r for r in predictions if by_a[r["id"]]["correct"] and not r["correct"]]
        gained=[r for r in predictions if not by_a[r["id"]]["correct"] and r["correct"]]
        def bucket(r):
            return "truncated" if r["truncated"] else "nontruncated_parse_failure" if not r["parse_ok"] else "nontruncated_wrong_numeric_requires_review"
        breakdown=Counter(bucket(r) for r in lost)
        assert sum(breakdown.values())==len(lost)
        batches=jsonl_read(directory/"supervision_batches.jsonl")
        result=read(directory/"training_result.json")
        assert sum(r["supervised_tokens"] for r in batches)==result["supervised_tokens"]
        assert sum(r["processed_tokens"] for r in batches)==result["processed_tokens"]
        gradients=[r for r in jsonl_read(directory/"costs.jsonl") if r["phase"]=="gradient_diagnostic"]
        rows.append({"candidate":label,"learning_rate":candidate["learning_rate"],"steps":candidate["steps"],
            "pool_n":128,"actual_unique_supervised_examples":len({r["id"] for r in batches}),
            "actual_supervised_tokens":result["supervised_tokens"],"actual_processed_tokens":result["processed_tokens"],
            "correct":candidate["metrics"]["correct"],"n":128,
            "truncated":sum(r["truncated"] for r in predictions),"parse_failed":sum(not r["parse_ok"] for r in predictions),
            "repetition_suspected":sum("repetition_suspected" in r["diagnostic_tags"] for r in predictions),
            "A_correct_candidate_wrong":len(lost),"A_wrong_candidate_correct":len(gained),
            "lost_truncated":breakdown["truncated"],"lost_nontruncated_parse_failure":breakdown["nontruncated_parse_failure"],
            "lost_nontruncated_wrong_numeric":breakdown["nontruncated_wrong_numeric_requires_review"],
            "adapter_delta_l2":result["adapter_delta_l2"],"finite_gradient_checks":sum(r["finite_gradients"] for r in gradients),
            "nonzero_gradient_checks":sum(r["nonzero_gradient_elements"]>0 for r in gradients),
            "frozen_gradient_checks_failed":sum(r["frozen_parameter_gradients"]>0 for r in gradients),
            "max_clipped_gradient_l2":max(r["clipped_gradient_l2"] for r in gradients),
            "reload_logits_max_abs_difference":result["reload_logits_max_absolute_difference"],
            "eligible":candidate["eligible"]})
        details.append({"candidate":label,"regressions":[r["id"] for r in lost],"fixes":[r["id"] for r in gained],
            "regression_partition":dict(breakdown),"partition_is_mutually_exclusive":True,
            "repetition_is_overlapping_surface_tag_not_causal_label":True})
        fingerprints.append({"candidate":label,
            "modeling_module_sha256":sha256_file(directory/"source_snapshot/src/reasoning_rl/modeling.py"),
            "run_sft_ast_sha256":function_digest(directory/"source_snapshot/src/reasoning_rl/training.py","run_sft"),
            "gradient_check_ast_sha256":function_digest(directory/"source_snapshot/src/reasoning_rl/training.py","_gradient_evidence"),
            "encode_sft_ast_sha256":function_digest(directory/"source_snapshot/src/reasoning_rl/modeling.py","encode_sft"),
            "collator_ast_sha256":function_digest(directory/"source_snapshot/src/reasoning_rl/modeling.py","CompletionCollator"),
            "generate_one_ast_sha256":function_digest(directory/"source_snapshot/src/reasoning_rl/evaluation.py","generate_one")})
        # Deterministic case pack for agent review, not outcome selection.
        if label in {"S1",decision["selected_candidate"]}:
            selected=[]
            for key in ("truncated","nontruncated_parse_failure","nontruncated_wrong_numeric_requires_review"):
                selected += [r for r in lost if bucket(r)==key][:2]
            selected += gained[:2]
            for r in selected:
                cases.append({"candidate":label,"id":r["id"],"question":val[r["id"]]["question"],
                    "reference_solution":val[r["id"]]["solution"],"gold_answer":r["gold_answer"],
                    "baseline_prediction":by_a[r["id"]]["prediction"],"candidate_prediction":r["prediction"],
                    "baseline_correct":by_a[r["id"]]["correct"],"candidate_correct":r["correct"],
                    "candidate_truncated":r["truncated"],"automatic_tags":r["diagnostic_tags"],
                    "human_reviewed":False,"agent_reviewed":False,"review_notes":""})
    old_weights=ROOT/"checkpoints/desktoprocm10pilot_20261105-B/adapter_model.safetensors"
    s1_weights=ROOT/"checkpoints"/(FAMILY+"-S1")/"adapter_model.safetensors"
    old_pred=jsonl_read(ROOT/"runs/desktoprocm10pilot_20261105-B-val/validation/predictions.jsonl")
    new_pred=jsonl_read(ROOT/"runs"/(FAMILY+"-S1-regression")/"validation/predictions.jsonl")
    exact=sha256_file(old_weights)==sha256_file(s1_weights)
    raw_equal=[(r["id"],r["prediction"]) for r in old_pred]==[(r["id"],r["prediction"]) for r in new_pred]
    assert exact and raw_equal
    relevant_functions_identical=all(len({r[key] for r in fingerprints})==1 for key in fingerprints[0] if key!="candidate")
    assert relevant_functions_identical
    source=load_split("train","train",limit=128,seed=20261105)
    assert all(score_answer(r["solution"],r["answer"]).correct for r in source)
    audit={"scope":"Fixed comparison128, old32 regression only; no holdout access",
        "baseline_correct":sum(r["correct"] for r in baseline),"candidates":rows,"paired_attribution":details,
        "exact_reproduction":{"old_and_S1_adapter_weight_bytes_identical":exact,"adapter_weight_sha256":sha256_file(s1_weights),
            "old32_all_raw_predictions_identical":raw_equal},
        "sft_relevant_implementation_ast_identical_across_candidates":relevant_functions_identical,
        "implementation_fingerprints":fingerprints,
        "gold_final_numeric_matches_manifest_all128":True,
        "gold_quality_limit":"Numeric consistency and complete serialization do not prove every reference reasoning statement is mathematically correct",
        "interpretation":"Strong optimization intervention reproduces degradation. Masks/shift/unchanged-LoRA/reload controls rule out the checked implementation defects. The specific internal mechanism behind repetitive generation remains unresolved."}
    json_write(OUT/"sft_attribution.json",audit)
    jsonl_write(OUT/"sft_case_review.jsonl",cases)
    with (OUT/"sft_candidates.csv").open("w",encoding="utf-8-sig",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"candidates":rows,"exact_reproduction":audit["exact_reproduction"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
