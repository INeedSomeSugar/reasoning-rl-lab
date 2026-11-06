"""Failed GPU work must remain auditable and retain its original error."""
import pytest
from reasoning_rl.runtime import Ledger, jsonl_read


def test_failed_stage_retains_wall_time_and_counts(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "costs.jsonl"
    with pytest.raises(ValueError, match="primary failure"):
        with Ledger(path).measure("partial") as cost:
            cost["generated_tokens"] = 11
            raise ValueError("primary failure")
    row = jsonl_read(path)[0]
    assert not row["success"]
    assert row["generated_tokens"] == 11
    assert row["wall_seconds"] >= 0


def test_cuda_cleanup_does_not_mask_primary_error(tmp_path, monkeypatch):
    import torch
    events = []
    class Event:
        def __init__(self, **kwargs):
            self.index = len(events)
            events.append(self)
        def record(self):
            if self.index == 1:
                raise RuntimeError("secondary CUDA event failure")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake GPU")
    monkeypatch.setattr(torch.cuda, "Event", Event)
    path = tmp_path / "costs.jsonl"
    with pytest.raises(ValueError, match="primary computation failure"):
        with Ledger(path).measure("partial"):
            raise ValueError("primary computation failure")
    row = jsonl_read(path)[0]
    assert row["device_timing_error"] == "secondary CUDA event failure"
    assert row["device_elapsed_seconds"] is None
    assert not row["success"]
