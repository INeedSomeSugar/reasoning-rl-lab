import pytest

from reasoning_rl.training import _exclude_training_items


def test_audited_exclusion_keeps_order_and_never_mutates_original_pool():
    rows = [{"id": "train-a", "split": "train"}, {"id": "flagged", "split": "train"}, {"id": "train-b", "split": "train"}]
    assert [r["id"] for r in _exclude_training_items(rows, ["flagged"])] == ["train-a", "train-b"]
    assert len(rows) == 3
    with pytest.raises(ValueError, match="candidate pool"):
        _exclude_training_items(rows, ["holdout-item"])
    with pytest.raises(ValueError, match="Duplicate"):
        _exclude_training_items(rows, ["flagged", "flagged"])


def test_nontraining_rows_cannot_enter_audited_selection_pool():
    with pytest.raises(ValueError, match="training candidate"):
        _exclude_training_items([{"id": "val-a", "split": "validation"}], [])
