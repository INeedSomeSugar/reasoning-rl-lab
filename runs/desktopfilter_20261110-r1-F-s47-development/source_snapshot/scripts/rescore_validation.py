"""Validation-only parser calibration; immutable originals and an explicit audit trail."""
import argparse
import shutil
from reasoning_rl.runtime import ROOT, bootstrap, jsonl_read, jsonl_write, json_write, sha256_file, code_hash

bootstrap()
from reasoning_rl.scoring import score_answer, SCORER_VERSION
from reasoning_rl.evaluation import aggregate, diagnostic_tags

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
args = parser.parse_args()
if any((ROOT/"experiments/final_claims").glob("*.json")):
    raise ValueError("Parser calibration must finish before opening final holdouts")
for name in args.runs:
    directory = ROOT/"runs"/name/"validation"
    path = directory/"predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = jsonl_read(path)
    original = directory/"predictions_original_scorer_v1.jsonl"
    if original.exists():
        raise ValueError("Already rescored; preserve the audit rather than overwriting")
    before = sha256_file(path)
    shutil.copyfile(path, original)
    shutil.copyfile(directory/"metrics.json", directory/"metrics_original_scorer_v1.json")
    for row in rows:
        score = score_answer(row["prediction"], row["gold_answer"])
        row.update(score.as_dict())
        row["diagnostic_tags"] = diagnostic_tags(row["prediction"], score, row["truncated"])
        row["scorer_version"] = SCORER_VERSION
    jsonl_write(path, rows)
    metrics = {**aggregate(rows), "phase": "validation", "optimizer_step": None,
               "decoding": "greedy", "thinking": False, "scorer_version": SCORER_VERSION,
               "rescored_from_immutable_predictions": True}
    json_write(directory/"metrics.json", metrics)
    json_write(directory/"rescoring_audit.json", {"original_sha256": before,
        "rescored_sha256": sha256_file(path), "scorer_version": SCORER_VERSION,
        "code_sha256": code_hash(), "reason": "Accept a single unambiguous number in the final #### section across newlines/Markdown; keep format independent",
        "generation_repeated": False, "split": "validation"})
    jsonl_write(directory/"error_review.jsonl", [{"id":r["id"], "prediction":r["prediction"],
        "gold_answer":r["gold_answer"], "automatic_tags":r["diagnostic_tags"],
        "human_reviewed":False, "arithmetic_error":None, "condition_omission":None, "review_notes":""} for r in rows if not r["correct"]])
    print(name, metrics["correct"], "/", metrics["n"], SCORER_VERSION)
