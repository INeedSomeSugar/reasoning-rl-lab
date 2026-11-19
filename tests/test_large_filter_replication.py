import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from reasoning_rl import data, fresh_development as fresh, prefiltering as pf
from reasoning_rl.runtime import json_write, jsonl_write, sha256_file
from reasoning_rl.replication_statistics import paired_difference_matrix, joint_cluster_accuracies, intervals_from_joint


def isolated_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    rows = [{"id": "gsm8k-train-00001", "split": "train", "official_split": "train"},
            {"id": "gsm8k-train-00002", "split": "train", "official_split": "train"}]
    jsonl_write(tmp_path / "data/train.jsonl", rows)
    json_write(tmp_path / "data/manifest.json", {"splits": {"train": {"path": "data/train.jsonl", "count": 2,
        "sha256": sha256_file(tmp_path / "data/train.jsonl")}}})
    relative = "data/fresh_development/study/manifest.json"
    manifest = {"kind": fresh.KIND, "family": "study", "ids": [rows[0]["id"]], "n": 1,
        "id_sha256": hashlib.sha256(rows[0]["id"].encode()).hexdigest(),
        "original_data_manifest_sha256": sha256_file(tmp_path / "data/manifest.json")}
    json_write(tmp_path / relative, manifest)
    json_write(tmp_path / fresh.REGISTRY, {"entries": [{"manifest_path": relative,
        "manifest_sha256": sha256_file(tmp_path / relative)}]})
    return relative, manifest


@pytest.mark.parametrize("intent", ["train", "select", "develop"])
def test_registered_fresh_items_never_return_from_ordinary_train(tmp_path, monkeypatch, intent):
    isolated_fixture(tmp_path, monkeypatch)
    assert [r["id"] for r in data.load_split("train", intent)] == ["gsm8k-train-00002"]


@pytest.mark.parametrize("intent", ["train", "select", "develop", "final", "evaluate_validation"])
def test_ordinary_permissions_never_open_fresh(tmp_path, monkeypatch, intent):
    with pytest.raises(ValueError, match="isolation"):
        data.load_split("fresh_development", intent)
    with pytest.raises(ValueError, match="dedicated"):
        fresh.load_fresh_development("does-not-exist", intent=intent)


def test_tampered_reservation_fails_closed(tmp_path, monkeypatch):
    relative, manifest = isolated_fixture(tmp_path, monkeypatch)
    manifest["ids"] = ["gsm8k-train-00002"]
    json_write(tmp_path / relative, manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        data.load_split("train", "train")


def test_fresh_evaluation_gate_requires_all35_terminal_and_registered37(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    json_write(tmp_path / "lock.json", {"stage": "training_in_progress"})
    with pytest.raises(ValueError, match="sealed"):
        fresh.authorize_endpoint("none", "lock.json", "A", None)
    json_write(tmp_path / "lock.json", {"stage": "all_training_endpoints_fixed_before_fresh_evaluation",
        "registered_training_keys": list(map(str, range(34))), "training_units": {}})
    with pytest.raises(ValueError, match="incomplete"):
        fresh.authorize_endpoint("none", "lock.json", "A", None)
    units = {str(i): {"status": "exhausted_failure"} for i in range(35)}
    units["12"] = {"status": "pending_retry"}
    json_write(tmp_path / "lock.json", {"stage": "all_training_endpoints_fixed_before_fresh_evaluation",
        "registered_training_keys": list(units), "training_units": units})
    with pytest.raises(ValueError, match="terminal"):
        fresh.authorize_endpoint("none", "lock.json", "A", None)


def test_even_one_changed_fixed_weight_keeps_new_development_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "checkpoint_hash", lambda path: "different_actual_hash")
    units = {str(i): {"status": "exhausted_failure"} for i in range(35)}
    units["12"] = {"status": "first_success_fixed", "checkpoint": "checkpoints/success", "checkpoint_sha256": "frozen_hash"}
    json_write(tmp_path / "lock.json", {"stage": "all_training_endpoints_fixed_before_fresh_evaluation",
        "registered_training_keys": list(units), "training_units": units})
    with pytest.raises(ValueError, match="endpoint changed"):
        fresh.authorize_endpoint("none", "lock.json", "A", None)


def test_random_subset_uses_full_pool_allows_overlap_and_never_picks_by_score():
    ids = list(map(str, range(100)))
    F = set(ids[:30])
    chosen = pf.matched_random_subset(ids, 30, 3101)
    assert len(chosen) == len(set(chosen)) == 30
    assert set(chosen)&F and set(chosen)-F
    assert chosen == pf.matched_random_subset(ids, 30, 3101)
    assert set(pf.matched_random_subset(ids, 100, 3101)) == set(ids)
    with pytest.raises(ValueError):
        pf.matched_random_subset(ids, 0, 3101)


def test_R_matches_actual_M_and_full_probe_debit_but_labels_artificial_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "ROOT", tmp_path)
    json_write(tmp_path / "data/manifest.json", {})
    candidates = {"ids": ["a", "b", "c"], "data_manifest_sha256": sha256_file(tmp_path / "data/manifest.json")}
    json_write(tmp_path / "pool.json", candidates)
    raw = [{"id": i, "correct": n < success} for i, success in zip(candidates["ids"], [0, 2, 4]) for n in range(4)]
    jsonl_write(tmp_path / "probe.jsonl", raw)
    kept, successes = pf.raw_success_filter(candidates["ids"], raw)
    json_write(tmp_path / "filter.json", {"candidate_pool_sha256": sha256_file(tmp_path / "pool.json"),
        "s2_checkpoint_sha256": "S2", "predictions_path": "probe.jsonl", "predictions_sha256": sha256_file(tmp_path / "probe.jsonl"),
        "k": 4, "retained_ids": kept, "raw_success_counts": successes, "probe_process_wall_seconds": 1234, "F_endpoint_count": 5})
    chosen = pf.matched_random_subset(candidates["ids"], len(kept), 3101)
    json_write(tmp_path / "random.json", {"ids": chosen, "candidate_pool_sha256": sha256_file(tmp_path / "pool.json"),
        "matched_prefilter_manifest_path": "filter.json", "matched_prefilter_manifest_sha256": sha256_file(tmp_path / "filter.json"),
        "random_subset_seed": 3101, "physical_subset_draw_seconds": 0.001})
    pool, _, debit, evidence = pf.resolve_frozen_pool({"candidate_pool_path": "pool.json", "random_subset_manifest_path": "random.json"},
        [{"id": i, "split": "train"} for i in candidates["ids"]], "S2")
    assert [r["id"] for r in pool] == chosen and len(pool) == len(kept)
    assert debit == 1234 and evidence["physical_probe_required_by_R"] is False
    assert evidence["accounting_kind"] == "R_experimental_matching_debit"
    assert evidence["F_endpoint_count"] == 5 and evidence["secondary_amortized_debit_per_F_endpoint_seconds"] == pytest.approx(1234/5)


def test_U_reuse_and_joint_bootstrap_keep_shared_endpoint_dependency():
    U = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    F = np.stack([U, U, U])
    assert np.array_equal(paired_difference_matrix(F, U), np.zeros((3, 2)))
    assert np.array_equal(paired_difference_matrix(F.astype(bool), U.astype(bool)), np.zeros((3, 2)))
    keys = ["A", "S2", "U-s13", "U-s29"]+[f"{a}-j{j}-s{s}" for a in ("F", "R") for j in (11, 12, 13) for s in (13, 29)]
    correct = np.stack([np.ones(3), np.ones(3)]+[U[0], U[1]]+[U[t] for _a in range(2) for _j in range(3) for t in range(2)])
    joint = joint_cluster_accuracies(correct, ["same", "same", "other"], replicates=10000)
    for crossed in (False, True):
        intervals = intervals_from_joint(joint, keys, [13, 29], [11, 12, 13], crossed=crossed)
        assert intervals["F-U"]["percentile95_pp"] == pytest.approx([0, 0])
        assert intervals["F-R"]["percentile95_pp"] == pytest.approx([0, 0])
    # Both questions of the same cluster always move together, never get split draws.
    assert set(np.round(joint[:, 2], 10)) <= {0.5, round(2/3, 10), 1.0}


def test_failed_phase_is_retained_and_same_label_cannot_overwrite(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import large_filter_common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "host_snapshot", lambda: {"test": True})
    (tmp_path / "logs").mkdir()
    (tmp_path / "experiments/study").mkdir(parents=True)
    failed = common.run_phase("study", "unit-a0", ["-c", "import sys; print('failed evidence'); sys.exit(1)"], 20, gpu=False)
    assert failed["exit_code"] == 1 and (tmp_path / "logs/study-unit-a0.log").exists()
    before = sha256_file(tmp_path / "logs/study-unit-a0.log")
    with pytest.raises(FileExistsError):
        common.run_phase("study", "unit-a0", ["-c", "pass"], 20, gpu=False)
    assert sha256_file(tmp_path / "logs/study-unit-a0.log") == before
    good = common.run_phase("study", "unit-a1", ["-c", "print('new full attempt')"], 20, gpu=False)
    assert good["exit_code"] == 0


def test_registered_matrix_contains_five_unique_U_and_fifteen_F_R_each():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from large_filter_common import matrix_order
    units = matrix_order()
    assert len(units) == len({u["key"] for u in units}) == 35
    assert [sum(u["arm"] == a for u in units) for a in ("U", "F", "R")] == [5, 15, 15]


def test_parent_verification_and_interpreter_startup_consume_B(monkeypatch):
    from reasoning_rl.training import _accounted_arm_start
    assert _accounted_arm_start({}, now_unix=100, now_perf=50) == 50
    assert _accounted_arm_start({"account_outer_process_start": True}, outer_started=90, now_unix=100, now_perf=50) == 40
    # At the same current process time a ten-second startup leaves ten fewer seconds.
    assert pf.remaining_training_seconds(3600, 50-40, 1080) == 2510
    monkeypatch.delenv("RRL_BUDGET_START_UNIX", raising=False)
    with pytest.raises(ValueError, match="controlled runner"):
        _accounted_arm_start({"account_outer_process_start": True}, now_unix=100, now_perf=50)
