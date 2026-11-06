import json
import pytest
from reasoning_rl import evaluation


def test_explicit_validation_ids_cannot_select_train_or_duplicate(tmp_path, monkeypatch):
    from reasoning_rl import data
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(data, "load_split", lambda split, intent, seed: [
        {"id": "val-a", "split": "validation"}, {"id": "val-b", "split": "validation"}])
    config = {"data_seed": 1, "evaluation": {"validation_size": 2, "validation_ids_path": "ids.json"}}
    (tmp_path / "ids.json").write_text(json.dumps({"comparison": ["val-b", "val-a"]}))
    assert [r["id"] for r in evaluation.validation_rows(config)] == ["val-b", "val-a"]
    for ids in (["val-a", "train-x"], ["val-a", "val-a"]):
        (tmp_path / "ids.json").write_text(json.dumps({"comparison": ids}))
        with pytest.raises(ValueError, match="unique members"):
            evaluation.validation_rows(config)
