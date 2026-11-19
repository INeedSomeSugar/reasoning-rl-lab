"""Post-reinstall readiness using tensors and the already-used old64 development set."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import traceback

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file

sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import START, read, run_phase, write_new, utc_now
from run_large_filter import verify_frozen


def tensor_child(record):
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    name = torch.cuda.get_device_name(0)
    assert "9070 XT" in name and torch.version.hip
    torch.manual_seed(13)
    q, k, v = [torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    # The adapter cast is the operation at which rapid post-abort failures surfaced.
    cast = q.detach().to(torch.float32)
    assert torch.isfinite(cast).all().item()
    result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    loss = result.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(result).all().item()
    assert all(t.grad is not None and torch.isfinite(t.grad).all().item() and t.grad.abs().sum().item() > 0 for t in (q, k, v))
    receipt = {"checked_utc": utc_now(), "status": "tensor_BF16_SDPA_forward_backward_and_cast_passed",
        "torch": torch.__version__, "hip": torch.version.hip, "device": name, "visible_device_count": 1,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "no_model_or_dataset_loaded": True,
        "no_optimizer_or_saved_model_modified": True}
    write_new(ROOT / record, receipt)
    print(json.dumps(receipt), flush=True)


def main(family):
    exp = ROOT / "experiments" / family
    assert read(exp / "runner_state.json")["status"] == "diagnostic_hold"
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    directory = exp / ("driver_recovery_readiness_"+stamp)
    directory.mkdir(exist_ok=False)
    (directory / "source.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    write_new(directory / "plan_before_diagnostics.json", {"registered_utc": utc_now(),
        "source_sha256": sha256_file(Path(__file__)), "tensor_acceptance": "same9070XT,onevisibleGPU,finiteBF16SDPAforward/backward,finiteBF16tofloat32cast",
        "old_development_diagnostic": "Replay frozen original64 IDs and greedy S2, original512token cap; no fresh/test/SVAMP",
        "driver_change_scope": "User-performed recovery; record environment stratum, no numerical configuration change",
        "parity_comparison_is_diagnostic_only": True, "no_adjustment_of_B_E_O_C_or_session_deadline": True})
    phases = []
    try:
        tensor = run_phase(family, "driver-recovery-tensor-"+stamp,
            ["analysis/check_large_filter_driver_recovery.py", "--tensor-child", "--record", (directory / "tensor.json").relative_to(ROOT).as_posix()], 180)
        phases.append(tensor)
        if tensor["exit_code"] or tensor["timeout"]:
            raise RuntimeError("Recovered-device tensor check failed; no model diagnostic launched")
        label = "driver-recovery-old64-"+stamp
        name = family+"-"+label
        replay = run_phase(family, label, ["-m", "reasoning_rl", "evaluate", "--name", name,
            "--config", f"configs/{family}_calibration.yaml", "--hardware", "rocm", "--adapter", START], 1200)
        phases.append(replay)
        if replay["exit_code"] or replay["timeout"]:
            raise RuntimeError("Recovered-device S2 old64 diagnostic failed; no fresh evaluation launched")
        old_path = ROOT / "runs" / (family+"-old-development64-a0") / "validation/predictions.jsonl"
        new_path = ROOT / "runs" / name / "validation/predictions.jsonl"
        old, new = jsonl_read(old_path), jsonl_read(new_path)
        assert len(old) == len(new) == 64 and [r["id"] for r in old] == [r["id"] for r in new]
        parity = {"n": 64, "prediction_text_identical": sum(a["prediction"] == b["prediction"] for a,b in zip(old,new,strict=True)),
            "strict_correct_flags_identical": sum(a["correct"] == b["correct"] for a,b in zip(old,new,strict=True)),
            "original_correct": sum(a["correct"] for a in old), "recovered_driver_correct": sum(a["correct"] for a in new),
            "original_predictions_sha256": sha256_file(old_path), "recovery_predictions_sha256": sha256_file(new_path),
            "ID_order_matches_original64": True, "does_not_prove_full_new_development_driver_equivalence": True}
        verify_frozen(family, plan)
        write_new(directory / "completed.json", {"completed_utc": utc_now(), "status": "passed", "process_receipts": phases,
            "tensor": read(directory / "tensor.json"), "old64_parity": parity,
            "frozen_source_data_models_unchanged": True, "fresh_development_test_SVAMP_not_accessed": True,
            "no_training_or_primary_evaluation_attempts_consumed": True})
        print(json.dumps({"readiness": "passed", "record": directory.relative_to(ROOT).as_posix(), "old64_parity": parity}, ensure_ascii=False), flush=True)
    except BaseException as error:
        json_write(directory / "failed.json", {"recorded_utc": utc_now(), "error": repr(error), "traceback": traceback.format_exc(),
            "process_receipts": phases, "fresh_development_not_launched": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family")
    parser.add_argument("--tensor-child", action="store_true")
    parser.add_argument("--record")
    args = parser.parse_args()
    if args.tensor_child:
        tensor_child(args.record)
    else:
        main(args.family)
