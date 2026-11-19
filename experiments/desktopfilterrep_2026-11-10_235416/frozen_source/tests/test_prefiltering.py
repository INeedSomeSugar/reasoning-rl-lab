import json
import math
import pytest
from reasoning_rl import prefiltering as pf


def records(successes):
    return [{"id": str(i), "correct": j < n} for i, n in enumerate(successes) for j in range(4)]


def test_raw_counts_exclude_all_correct_and_wrong_despite_smoothing():
    ids = list(map(str, range(5)))
    kept, counts = pf.raw_success_filter(ids, records([0, 1, 2, 3, 4]))
    assert kept == ["1", "2", "3"]
    assert counts == {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    assert all(0 < (n+.5)/5 < 1 for n in counts.values())


@pytest.mark.parametrize("bad", [records([1])[:-1], records([1])+records([1])[:1], [{"id": "x", "correct": True}],
    [{"id": "0", "correct": 0.5}]*4])
def test_partial_duplicate_extra_or_nonbinary_probe_is_rejected(bad):
    with pytest.raises(ValueError):
        pf.raw_success_filter(["0"], bad)


def test_cache_is_never_free_in_primary_or_amortized_comparison():
    charge = pf.charge_probe(123, 3)
    assert charge["primary_standalone_debit_per_F_endpoint_seconds"] == 123
    assert charge["secondary_amortized_debit_per_F_endpoint_seconds"] == 41
    assert pf.remaining_training_seconds(600, 10, 123) == 467
    assert pf.remaining_training_seconds(600, 10, 0) == 590
    assert pf.remaining_training_seconds(600, 590, 123) == 0


@pytest.mark.parametrize("value", [-1, 0, float("inf"), float("nan")])
def test_invalid_probe_cost_rejected(value):
    with pytest.raises(ValueError):
        pf.charge_probe(value)


def test_end_to_end_pool_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text("{}")
    candidate = {"ids": ["0", "1", "2"], "data_manifest_sha256": pf.sha256_file(tmp_path / "data/manifest.json")}
    (tmp_path / "pool.json").write_text(json.dumps(candidate))
    raw = records([0, 2, 4])
    raw_path = tmp_path / "probe.jsonl"
    raw_path.write_text("\n".join(json.dumps(r) for r in raw)+"\n")
    subset, counts = pf.raw_success_filter(candidate["ids"], raw)
    manifest = {"candidate_pool_sha256": pf.sha256_file(tmp_path / "pool.json"), "s2_checkpoint_sha256": "S2",
        "predictions_path": "probe.jsonl", "predictions_sha256": pf.sha256_file(raw_path),
        "retained_ids": subset, "raw_success_counts": counts, "k": 4, "probe_process_wall_seconds": 20}
    (tmp_path / "filter.json").write_text(json.dumps(manifest))
    train = [{"id": str(i), "split": "train"} for i in range(3)]
    opts = {"candidate_pool_path": "pool.json"}
    pool, ids, debit, _ = pf.resolve_frozen_pool(opts, train, "S2")
    assert [r["id"] for r in pool] == ids == candidate["ids"] and debit == 0
    opts["prefilter_manifest_path"] = "filter.json"
    pool, ids, debit, evidence = pf.resolve_frozen_pool(opts, train, "S2")
    assert [r["id"] for r in pool] == ["1"] and ids == candidate["ids"] and debit == 20
    with pytest.raises(ValueError, match="S2 differ"):
        pf.resolve_frozen_pool(opts, train, "wrong_start")
    raw_path.write_text(raw_path.read_text()+"\n")
    with pytest.raises(ValueError, match="output changed"):
        pf.resolve_frozen_pool(opts, train, "S2")


def test_paths_cannot_escape_project(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="project-local"):
        pf.local_file("../outside.json")
