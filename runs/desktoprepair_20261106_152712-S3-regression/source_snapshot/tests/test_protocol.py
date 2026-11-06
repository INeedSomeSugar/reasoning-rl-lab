import math
import pytest
from reasoning_rl.scoring import canonical_number, score_answer, correctness_reward, group_advantages
from reasoning_rl.sampling import DifficultySampler
from reasoning_rl.data import load_split, normalize_question


@pytest.mark.parametrize("text,gold,correct,format_ok", [
    ("Steps: 9-3=6.\n#### 6", "6", True, True),
    ("I computed 6 but cannot solve.", "6", False, False),
    ("Earlier result 9. Final \\boxed{6}", "6", True, False),
    ("#### 1,234", "1234.0", True, True),
    ("#### -0.25", "-1/4", True, True),
    ("#### 7\n#### 6", "6", True, True),
    ("\\boxed{6}\n#### unclear", "6", False, False),
    ("<think>#### 2</think>\n#### 6", "6", True, True),
    ("#### 6 apples", "6", True, False),
    ("####\n\n**Answer:** 6", "6", True, False),
    ("#### 7 or 6", "6", False, False),
    ("#### 6/0", "6", False, True),
    ("Final answer: **6**", "6", True, False),
    ("#### x6", "6", False, False),
    ("#### NaN", "6", False, False),
    ("The cost is $60.\n60", "60", True, False),
    ("#### 6.1", "6", False, True),
])
def test_answer_contract(text, gold, correct, format_ok):
    result = score_answer(text, gold)
    assert result.correct is correct
    assert result.format_ok is format_ok


def test_reward_does_not_reward_format_or_incidental_numbers():
    assert correctness_reward(["#### 7", "The quantity is 6 but no answer.", "\\boxed{6}"], ["6"]*3) == [0, 0, 1]


def test_advantage_zero_variance_and_group_invariance():
    assert group_advantages([0, 0, 0]) == [0, 0, 0]
    assert group_advantages([1, 1, 1]) == [0, 0, 0]
    a = group_advantages([0, 1, 1, 0])
    assert sum(a) == pytest.approx(0)
    assert a[0] < 0 < a[1]
    assert all(math.isfinite(x) for x in a)


def test_uniform_dynamic_static_controls_and_coverage():
    ids = ["hard", "medium", "easy"]
    sampler = DifficultySampler(ids, "dynamic", 13)
    with pytest.raises(ValueError):
        sampler.draw(1)
    sampler.refresh({"hard": (0, 8), "medium": (4, 8), "easy": (8, 8)})
    w = sampler.weights()
    assert w[1] > w[0]
    assert w[0] == pytest.approx(w[2])
    assert sum(w) == pytest.approx(1)
    assert min(w) >= 0.25/3
    first = list(w)
    sampler.refresh({"hard": (4, 8), "medium": (8, 8), "easy": (8, 8)})
    assert sampler.weights()[0] > sampler.weights()[1]
    assert sampler.weights() != first
    static = DifficultySampler(ids, "static", 13)
    static.refresh({i: (1, 2) for i in ids})
    with pytest.raises(ValueError):
        static.refresh({i: (1, 2) for i in ids})
    with pytest.raises(ValueError):
        sampler.refresh({"validation_item": (1, 2)})
    a = DifficultySampler(ids, "uniform", 13)
    b = DifficultySampler(ids, "uniform", 13)
    assert a.draw(100) == b.draw(100)


@pytest.mark.parametrize("split,intent", [("test", "train"), ("external", "select"), ("validation", "select"), ("test", "develop")])
def test_holdout_rejected_before_any_file_read(split, intent):
    with pytest.raises(ValueError, match="isolation"):
        load_split(split, intent)


def test_near_duplicate_normalization():
    assert normalize_question(" A  2-apple problem！ ") == normalize_question("a 2 apple problem")


def test_prepared_data_reuse_is_offline_and_rejects_tampering(tmp_path, monkeypatch):
    import json
    import reasoning_rl.data as data
    from reasoning_rl.runtime import sha256_file
    lock = {"gsm8k": {"revision": "pinned"}, "svamp": {"revision": "pinned"}}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/data.sources.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    (tmp_path / "data").mkdir()
    splits = {}
    for split in ("train", "validation", "test", "external"):
        path = tmp_path / "data" / f"{split}.jsonl"
        path.write_text("opaque bytes; no need to deserialize held-out answers", encoding="utf-8")
        splits[split] = {"path": f"data/{split}.jsonl", "sha256": sha256_file(path)}
    manifest = {"sources": lock, "seed": 20261105, "validation_target": 512,
                "near_threshold": 0.9, "splits": splits}
    (tmp_path / "data/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing prepared splits must not be re-downloaded or deserialized")
    monkeypatch.setattr(data.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(data, "jsonl_read", forbidden)
    assert data.prepare() == manifest
    (tmp_path / "data/train.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        data.prepare()


def test_mask_only_supervises_completion_and_padding_is_ignored():
    torch = pytest.importorskip("torch")
    from reasoning_rl.modeling import CompletionCollator
    rows = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]},
            {"input_ids": [1, 4], "attention_mask": [1, 1], "labels": [-100, 4]}]
    batch = CompletionCollator(99)(rows)
    assert batch["labels"].tolist() == [[-100, -100, 3], [-100, 4, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_prepared_splits_and_near_pairs_are_isolated():
    import json
    from reasoning_rl.runtime import ROOT, jsonl_read
    manifest_path = ROOT / "data/manifest.json"
    if not manifest_path.exists():
        pytest.skip("Run prepare-data for integration audit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development = {s: load_split(s, "develop") for s in ("train", "validation")}
    origins = {r["id"]: s for s, rows in development.items() for r in rows}
    # Holdout IDs encode the official split; no need to deserialize holdout answers in an engineering test.
    def origin(identifier):
        if identifier in origins:
            return origins[identifier]
        if identifier.startswith("gsm8k-test-"):
            return "test"
        if identifier.startswith("svamp-"):
            return "external"
        return None
    assert not ({r["id"] for r in development["train"]} & {r["id"] for r in development["validation"]})
    for pair in jsonl_read(ROOT / "data/audit/near_pairs.jsonl"):
        a, b = origin(pair["a"]), origin(pair["b"])
        if a is not None and b is not None and (a in {"train", "validation"} or b in {"train", "validation"}):
            assert a == b
