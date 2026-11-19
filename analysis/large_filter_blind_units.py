"""Exact-text deduplication for actual Agent review, without reading method identities.

Every original case and alias remains covered. This program never assigns semantic
categories: each unique question/gold/reference/prediction/truncation tuple needs
an explicit Agent annotation before judgments can be copied to exact duplicates.
"""
import argparse
import datetime as dt
import hashlib
import json
import random
import sys

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_write, sha256_file
from finalize_large_filter_reviews import CATEGORIES
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read


def deduplicate_cases(pack):
    units, crosswalk = {}, []
    seen_cases = set()
    for case in pack:
        if case["case_id"] in seen_cases:
            raise ValueError("Duplicate source case")
        seen_cases.add(case["case_id"])
        aliases = set()
        for response in case["responses"]:
            if response["alias"] in aliases:
                raise ValueError("Duplicate alias in a case")
            aliases.add(response["alias"])
            payload = {k: case[k] for k in ("id", "question", "gold_answer", "original_reference_solution")}
            payload.update(prediction=response["prediction"], truncated=response["truncated"])
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            unit = {"review_unit_id": digest, **payload}
            if digest in units and units[digest] != unit:
                raise AssertionError("Canonical hash collision")
            units[digest] = unit
            crosswalk.append({"case_id": case["case_id"], "alias": response["alias"], "review_unit_id": digest})
        if len(aliases) != 2:
            raise ValueError("Each original pair must have exactly two responses")
    return list(units.values()), crosswalk


def materialize_cases(pack, annotations):
    units, crosswalk = deduplicate_cases(pack)
    expected = {u["review_unit_id"] for u in units}
    by_unit = {}
    for row in annotations:
        key = row["review_unit_id"]
        if key not in expected or key in by_unit:
            raise ValueError("Unknown or duplicate unique-response annotation")
        if row.get("reviewer_kind") != "Agent" or row.get("human_reviewed") is not False:
            raise ValueError("Annotations must identify an Agent, never a human review")
        if row.get("method_identity_hidden_during_review") is not True:
            raise ValueError("Unblinded annotations are not accepted")
        if row.get("category") not in CATEGORIES or not row.get("notes", "").strip():
            raise ValueError("Every unique response needs an explicit category and substantive notes")
        by_unit[key] = row
    if set(by_unit) != expected:
        raise RuntimeError(f"Unique-response review incomplete: {len(expected-set(by_unit))} pending")
    by_alias = {(r["case_id"], r["alias"]): r["review_unit_id"] for r in crosswalk}
    result = []
    for case in pack:
        responses, issues = [], set()
        for response in case["responses"]:
            key = by_alias[(case["case_id"], response["alias"])]
            row = by_unit[key]
            responses.append({"alias": response["alias"], "review_unit_id": key,
                "category": row["category"], "notes": row["notes"]})
            if row.get("question_or_gold_issue"):
                issues.add(row["question_or_gold_issue"])
        result.append({"case_id": case["case_id"], "reviewer_kind": "Agent", "human_reviewed": False,
            "method_identity_hidden_during_review": True, "exact_text_deduplication_used": True,
            "responses": responses, "question_or_gold_issue": " | ".join(sorted(issues)) if issues else None})
    assert len(result) == len(pack) and sum(len(c["responses"]) for c in result) == len(crosswalk)
    return result


def json_new(path, value):
    if path.exists():
        if read(path) != value:
            raise FileExistsError(f"Refuse to overwrite different evidence: {path}")
    else:
        json_write(path, value)


def jsonl_new(path, rows):
    if path.exists():
        if jsonl_read(path) != rows:
            raise FileExistsError(f"Refuse to overwrite different annotations: {path}")
    else:
        jsonl_write(path, rows)


def selfcheck():
    base = {"id": "synthetic-only", "question": "1+1?", "gold_answer": "2", "original_reference_solution": "1+1=2"}
    cases = [{**base, "case_id": "c1", "responses": [
        {"alias": "a", "prediction": "2", "truncated": False},
        {"alias": "b", "prediction": "2", "truncated": False}]},
        {**base, "case_id": "c2", "responses": [
        {"alias": "c", "prediction": "2", "truncated": False},
        {"alias": "d", "prediction": "2", "truncated": True}]},
        {**base, "gold_answer": "3", "case_id": "c3", "responses": [
        {"alias": "e", "prediction": "2", "truncated": False},
        {"alias": "f", "prediction": "2", "truncated": False}]}]
    units, crosswalk = deduplicate_cases(cases)
    assert len(units) == 3 and len(crosswalk) == 6
    annotations = [{"review_unit_id": u["review_unit_id"], "category": "uncertain",
        "notes": "Synthetic smoke fixture only, no real response has been reviewed.",
        "reviewer_kind": "Agent", "human_reviewed": False, "method_identity_hidden_during_review": True} for u in units]
    filled = materialize_cases(cases, annotations)
    assert len(filled) == 3 and sum(len(c["responses"]) for c in filled) == 6
    for bad in (annotations[:-1], annotations+annotations[:1],
                [{**annotations[0], "human_reviewed": True}]+annotations[1:]):
        try:
            materialize_cases(cases, bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Incomplete, duplicate or misleading reviewer metadata was accepted")
    return {"status": "passed", "synthetic_only": True, "real_cases_read": False,
        "exact_duplicates_reused": True, "gold_and_truncation_differences_not_merged": True,
        "all_case_aliases_preserved": True, "missing_duplicate_and_human_annotations_rejected": True}


def main(family, mode):
    exp = ROOT / "experiments" / family
    if mode == "selfcheck":
        result = selfcheck()
        json_write(exp / "blind_unique_units_synthetic_selfcheck.json", result)
        print(result)
        return
    if not (exp / "all_model_tasks_complete.json").exists():
        raise RuntimeError("No fresh response review before all registered model tasks finish")
    pack_path = exp / "blind_review_pack.jsonl"
    pack_receipt = read(exp / "blind_review_pack_receipt.json")
    assert sha256_file(pack_path) == pack_receipt["pack_sha256"]
    pack = jsonl_read(pack_path)
    units_path = exp / "blind_review_unique_units.jsonl"
    map_path = exp / "blind_review_unit_crosswalk_do_not_inspect_until_annotations_fixed.jsonl"
    manifest_path = exp / "blind_review_unique_units_manifest.json"
    if mode == "prepare":
        if manifest_path.exists():
            manifest = read(manifest_path)
            assert manifest["source_pack_sha256"] == sha256_file(pack_path)
            assert manifest["units_sha256"] == sha256_file(units_path)
            assert manifest["crosswalk_sha256"] == sha256_file(map_path)
            print({"status": "existing_exact_units_verified", "unique_response_units": manifest["unique_response_units"]})
            return
        units, crosswalk = deduplicate_cases(pack)
        random.Random(read(exp / "protocol_frozen.json")["blind_review"]["seed"]).shuffle(units)
        jsonl_new(units_path, units)
        jsonl_new(map_path, crosswalk)
        json_write(manifest_path, {"prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_pack_sha256": sha256_file(pack_path), "units_sha256": sha256_file(units_path),
            "crosswalk_sha256": sha256_file(map_path), "original_cases": len(pack),
            "original_response_presentations": len(crosswalk), "unique_response_units": len(units),
            "rule": "Exact same ID, question, gold, reference, prediction text and truncation only; no normalization or semantic merging",
            "method_names_and_occurrence_counts_hidden_in_review_units": True,
            "identity_map_read": False, "semantic_categories_assigned_by_program": False})
        print({"cases": len(pack), "response_presentations": len(crosswalk), "unique_units_requiring_actual_Agent_review": len(units)})
        return
    manifest = read(manifest_path)
    assert manifest["source_pack_sha256"] == sha256_file(pack_path)
    assert manifest["units_sha256"] == sha256_file(units_path)
    assert manifest["crosswalk_sha256"] == sha256_file(map_path)
    batches = sorted(exp.glob("blind_unit_review_batch_*.jsonl"))
    annotations = [r for path in batches for r in jsonl_read(path)]
    cases = materialize_cases(pack, annotations)
    fixed_path = exp / "all_blind_unit_annotations_fixed_before_materialization.jsonl"
    jsonl_new(fixed_path, sorted(annotations, key=lambda r: r["review_unit_id"]))
    target = exp / "blind_review_batch_from_exact_units.jsonl"
    if [p for p in exp.glob("blind_review_batch_*.jsonl") if p != target]:
        raise FileExistsError("Other case annotations already exist; never silently replace or duplicate them")
    jsonl_new(target, cases)
    receipt_path = exp / "blind_review_unit_materialization_receipt.json"
    receipt = {"source_pack_sha256": sha256_file(pack_path), "original_cases": len(pack),
        "original_response_presentations": sum(len(c["responses"]) for c in cases),
        "unique_response_units_actually_annotated": len(annotations),
        "unit_annotations_fixed_sha256": sha256_file(fixed_path), "derived_case_annotations_sha256": sha256_file(target),
        "unit_annotation_batches": {p.name: sha256_file(p) for p in batches},
        "all_original_case_aliases_preserved": True, "identity_map_read": False,
        "semantic_categories_assigned_by_program": False, "human_reviewed": False}
    json_new(receipt_path, receipt)
    print({"status": "every_case_materialized_from_complete_explicit_Agent_annotations", **receipt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--mode", choices=("prepare", "materialize", "selfcheck"), required=True)
    args = parser.parse_args()
    main(args.family, args.mode)
