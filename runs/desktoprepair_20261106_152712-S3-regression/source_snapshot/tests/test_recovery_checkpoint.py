import contextlib
import json

import pytest

import reasoning_rl.training as training


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_recovery_weights_survive_validation_failure(tmp_path, monkeypatch, evaluation_fails):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text("{}", encoding="utf-8")
    run = tmp_path / "runs/example"
    calls = []

    class Ledger:
        @contextlib.contextmanager
        def measure(self, *args, **kwargs):
            yield {}

    def fake_save(model, tokenizer, path, config, stage, **metadata):
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(b"test serialization boundary")
        calls.append(("saved", stage, metadata["completed_optimizer_steps"]))

    def fake_evaluate(*args, **kwargs):
        receipt = json.loads((run / "recovery_checkpoints/step_00008.json").read_text())
        assert not receipt["evaluation_completed"]
        assert not receipt["exact_optimizer_resume_supported"]
        assert (tmp_path / "checkpoints/example-recovery/step_00008/adapter_model.safetensors").exists()
        calls.append(("evaluated",))
        if evaluation_fails:
            raise RuntimeError("simulated generation failure")
        return {"n": 1}

    monkeypatch.setattr(training, "save_adapter", fake_save)
    monkeypatch.setattr(training, "evaluate_rows", fake_evaluate)
    invoke = lambda: training._evaluate_with_recovery(None, None, [], {}, run, Ledger(), 8, "fixed-sft-hash")
    if evaluation_fails:
        with pytest.raises(RuntimeError, match="simulated generation failure"):
            invoke()
    else:
        assert invoke() == {"n": 1}
    receipt = json.loads((run / "recovery_checkpoints/step_00008.json").read_text())
    assert receipt["evaluation_completed"] is (not evaluation_fails)
    assert receipt["not_a_completed_run"]
    assert calls == [("saved", "grpo_recovery", 8), ("evaluated",)]
    with pytest.raises(FileExistsError):
        invoke()
