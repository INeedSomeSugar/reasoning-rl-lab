"""CPU regression checks for the missing evaluation branch."""
import json
import sys

import numpy as np
import pytest

from reasoning_rl.runtime import ROOT
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_large_filter import blind_pack
from large_filter_missing_statistics import differences, full_matrix_bounds, registered_pairs


def test_full_matrix_bounds_use_missing_u_once_at_its_total_weight():
    rows = list(registered_pairs([13, 29], [11, 12, 13]))
    counts = {"U-s13": 50, "F-j11-s13": 60, "F-j12-s13": 60, "F-j13-s13": 60,
              "F-j12-s29": 70, "F-j13-s29": 70}
    result = full_matrix_bounds(rows, counts, 100, "U")
    assert result["missing_endpoint_cell_weights"] == {"U-s29": -3, "F-j11-s29": 1}
    assert result["bounds_pp"] == pytest.approx([-130/6, 45])


def test_observed_pair_bootstrap_keeps_missing_slots_and_reuses_u():
    pairs = list(registered_pairs([13, 29], [11, 12, 13]))
    keys = {row[arm] for row in pairs for arm in ("F", "U", "R", "S2", "A")}
    absent = {"U-s29", "F-j11-s29"}
    fields = {key: {"evaluation_status": "exhausted_failure" if key in absent else "first_success_fixed",
                    "evaluation_driver_version": "old" if key == "S2" else "new"} for key in keys}
    predictions = {key: [{"correct": bool(value)} for value in ([1,1,0,1] if key.startswith("F") else [1,0,1,0])]
                   for key in sorted(keys-absent)}
    rows, comparisons, models, draws, _ = differences(predictions, ["a", "a", "b", "c"], fields, [13, 29], [11, 12, 13],
        {"conditional_cluster_bootstrap_replicates": 10000, "seed": 20261116})
    assert len(rows) == 6 and draws.shape == (10000, len(models))
    stats = comparisons["F-U"]
    assert stats["original_point_estimate_pp"] is None
    assert stats["matrix_pp_probe_by_training"] == [[25.0, None], [25.0, None], [25.0, None]]
    assert stats["available"]["pairs"] == 3 and stats["available"]["mean_pp"] == 25.0
    reference = np.quantile((draws[:, models.index("F-j11-s13")]-draws[:, models.index("U-s13")])*100, [.025, .975])
    assert np.allclose(stats["available"]["conditional_percentile95_pp"], reference)
    assert comparisons["F-S2"]["different_driver"]["pairs"] == 5


def test_blind_pack_preserves_missing_registered_pairs_and_labels_supplement(tmp_path):
    ids = [f"id{i}" for i in range(48)]
    records = [{"id": key, "question": key, "answer": "1", "solution": "1"} for key in ids]
    keys = ["F-j20261111-s29", "R-j20261111-s29", "S2", "F-j20261111-s13", "U-s13"]
    predictions = {}
    for k,key in enumerate(keys):
        flags = [i < 24 for i in range(len(ids))]
        flags[k] = not flags[k]
        predictions[key] = [{"id": question, "prediction": f"response {k} {i}", "truncated": False, "correct": flags[i]}
            for i, question in enumerate(ids)]
    scope = {"review": {"supplement": {"left": "F-j20261111-s13", "right": "U-s13", "seed": 20261117}}}
    endpoints = {k: {"evaluation_driver_version": "new"} for k in keys}
    blind_pack(tmp_path, {"blind_review": {"seed": 20261117}}, predictions, records, endpoints, scope)
    receipt = json.loads((tmp_path / "blind_review_pack_receipt.json").read_text(encoding="utf-8"))
    assert receipt["registered_pair_count"] == 9 and len(receipt["unavailable_registered_pairs"]) == 7
    assert receipt["exploratory_pairs"] == ["F-j20261111-s13/U-s13"]
    assert receipt["case_counts_by_scope"] == {"registered_seed29": 68, "exploratory_same_driver_seed13": 34}
    cases = [json.loads(line) for line in (tmp_path / "blind_review_pack.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {tuple(sorted(r)) for r in cases} == {("case_id", "gold_answer", "id", "original_reference_solution", "question", "responses")}
    assert all(set(r) == {"alias", "prediction", "truncated"} for c in cases for r in c["responses"])
