"""Unblind only after every preregistered case has an explicit Agent annotation."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import sys

from reasoning_rl.runtime import ROOT, jsonl_read, jsonl_write, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read

CATEGORIES = {"supported_correct", "correct_without_checkable_reasoning", "read_or_modeling_error", "calculation_error", "parsing_error",
              "truncated", "lucky_numeric_correct", "uncertain", "other"}


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    receipt = read(exp / "blind_review_pack_receipt.json")
    pack = jsonl_read(exp / "blind_review_pack.jsonl")
    assert sha256_file(exp / "blind_review_pack.jsonl") == receipt["pack_sha256"]
    originals = {r["case_id"]: r for r in pack}
    annotations = []
    for path in sorted(exp.glob("blind_review_batch_*.jsonl")):
        annotations.extend(jsonl_read(path))
    by_case = {}
    for row in annotations:
        case = row["case_id"]
        if case in by_case:
            raise ValueError("Duplicate case annotation; preserve it and adjudicate explicitly")
        if case not in originals or row["reviewer_kind"] != "Agent" or row["human_reviewed"] is not False:
            raise ValueError("Unknown case or misleading review identity")
        assert row["method_identity_hidden_during_review"] is True
        expected = {r["alias"] for r in originals[case]["responses"]}
        assert {r["alias"] for r in row["responses"]} == expected and len(row["responses"]) == 2
        assert all(r["category"] in CATEGORIES and r["notes"].strip() for r in row["responses"])
        by_case[case] = row
    pending = set(originals)-set(by_case)
    if pending:
        json_write(out / "blind_review_progress.json", {"completed_cases": len(by_case), "pending_cases": len(pending)})
        raise RuntimeError(f"Blinded Agent review still has {len(pending)} pending cases; do not unblind")
    exact_units = None
    exact_receipt_path = exp / "blind_review_unit_materialization_receipt.json"
    if exact_receipt_path.exists():
        exact_units = read(exact_receipt_path)
        assert exact_units["source_pack_sha256"] == receipt["pack_sha256"]
        assert exact_units["derived_case_annotations_sha256"] == sha256_file(exp / "blind_review_batch_from_exact_units.jsonl")
        fixed_units_path = exp / "all_blind_unit_annotations_fixed_before_materialization.jsonl"
        assert exact_units["unit_annotations_fixed_sha256"] == sha256_file(fixed_units_path)
        unit_annotations = {r["review_unit_id"]: r for r in jsonl_read(fixed_units_path)}
        assert len(unit_annotations) == exact_units["unique_response_units_actually_annotated"]
        used = set()
        for case in by_case.values():
            for response in case["responses"]:
                unit = unit_annotations[response["review_unit_id"]]
                assert response["category"] == unit["category"] and response["notes"] == unit["notes"]
                used.add(response["review_unit_id"])
        assert used == set(unit_annotations)
    # Annotation hash is fixed before this program reads the identity map.
    jsonl_write(exp / "all_blind_annotations_fixed_before_unblinding.jsonl", [by_case[r["case_id"]] for r in pack])
    annotation_hash = sha256_file(exp / "all_blind_annotations_fixed_before_unblinding.jsonl")
    identities = jsonl_read(exp / "blind_review_identity_sealed_until_review.jsonl")
    identity = {r["case_id"]: r for r in identities}
    assert set(identity) == set(originals)
    responses, strata, unique, flags = [], Counter(), defaultdict(list), []
    for case in pack:
        key = case["case_id"]
        labels = identity[key]
        anno = by_case[key]
        strata[(labels["comparison"], labels["stratum"])] += 1
        if anno.get("question_or_gold_issue"):
            flags.append({"case_id": key, "id": case["id"], "notes": anno["question_or_gold_issue"], "gold_changed": False})
        label_by_alias = {r["alias"]: r for r in labels["responses"]}
        for r in anno["responses"]:
            label = label_by_alias[r["alias"]]
            if r["category"] == "lucky_numeric_correct" and not label["correct_under_frozen_scorer"]:
                raise ValueError("A scorer-wrong answer cannot be labeled lucky numeric correct")
            item = {"case_id": key, "id": case["id"], "comparison": labels["comparison"], "stratum": labels["stratum"],
                "review_scope": labels.get("review_scope", "registered_seed29"),
                **label, **r, "human_reviewed": False, "reviewer_kind": "Agent"}
            responses.append(item)
            unique[(label["endpoint_key"], case["id"])].append(item)
    counts, conflicts = Counter(), []
    for (endpoint, identifier), entries in unique.items():
        categories = {r["category"] for r in entries}
        category = next(iter(categories)) if len(categories) == 1 else "uncertain"
        if len(categories) > 1:
            conflicts.append({"endpoint": endpoint, "id": identifier, "categories": sorted(categories),
                "case_ids": [r["case_id"] for r in entries], "summary_category": "uncertain",
                "note": "Original blinded judgments retained; no favorable post-unblinding resolution"})
        counts[(endpoint, category)] += 1
    jsonl_write(out / "blind_review_unblinded_responses.jsonl", responses)
    jsonl_write(out / "blind_review_question_gold_flags.jsonl", flags)
    json_write(out / "blind_review_summary.json", {"unblinded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_cases": len(pack), "pending_cases": 0, "reviewed_responses": len(responses),
        "unique_question_ids": len({r["id"] for r in pack}), "unique_endpoint_question_responses": len(unique),
        "annotation_sha256_fixed_before_unblinding": annotation_hash, "pack_sha256": receipt["pack_sha256"],
        "human_reviewed": False, "reviewer_kind": "Agent", "method_names_hidden_in_annotation_pack": True,
        "scope": receipt["scope"], "stratum_counts": [{"comparison": comp, "stratum": kind, "n": n} for (comp, kind), n in strata.items()],
        "registered_pair_count": receipt.get("registered_pair_count"), "unavailable_registered_pairs": receipt.get("unavailable_registered_pairs", []),
        "complete_registered_scope": receipt.get("complete_registered_scope", True),
        "exploratory_pairs": receipt.get("exploratory_pairs", []), "case_counts_by_scope": receipt.get("case_counts_by_scope", {}),
        "unique_response_category_counts": [{"endpoint": ep, "category": category, "n": n} for (ep, category), n in sorted(counts.items())],
        "repeated_response_judgment_conflicts": conflicts, "question_or_gold_flags": flags,
        "exact_text_deduplication": exact_units,
        "unreviewed_other_seed_numeric_errors_automatically_arithmetic": False,
        "scorer_gold_and_primary_outcomes_changed": False})
    print({"cases_reviewed": len(pack), "responses": len(responses), "conflicts_preserved": len(conflicts), "gold_flags_not_applied": len(flags)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
