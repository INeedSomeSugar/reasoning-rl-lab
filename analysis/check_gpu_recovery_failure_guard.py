"""CPU-only scheduling regression: protect retries after a shared GPU failure."""
import datetime as dt
from unittest.mock import patch

from reasoning_rl.runtime import ROOT, json_write, sha256_file
import resume_large_filter_after_gpu_recovery as recovery


def main():
    calls = []
    gpu = {"driver": {"DriverVersion": "test-driver"}}
    def phase(*args, **kwargs):
        calls.append((args, kwargs))
        return {"exit_code": 1, "timeout": False, "finished_unix": 123}
    guard = recovery.Guard("synthetic", {"prior-a0"}, "test-driver", {"eval-synthetic-a0"},
        lambda *_: calls.append("verify"), phase, lambda: gpu)
    guard.last_gpu = gpu
    args = ["-m", "reasoning_rl", "evaluate", "--split", "fresh_development"]
    with patch.object(recovery, "jsonl_append"):
        guard.phase("synthetic", "eval-synthetic-a0", args, 7650.0)
    assert calls == [(("synthetic", "eval-synthetic-a0", args, 7650.0), {})]
    try:
        guard.verify("synthetic", {})
    except recovery.RecoveryHold:
        pass
    else:
        raise AssertionError("Failed child did not fence the next attempt")
    assert len(calls) == 1
    try:
        guard.phase("synthetic", "eval-synthetic-a0", ["-m", "reasoning_rl", "grpo"], 7650.0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Evaluation-only guard permitted training")
    assert len(calls) == 1
    baseline = {"x": {"attempts": [{"label": "prior-a0", "status": "failed"}]}}
    validating = recovery.Guard("synthetic", {"prior-a0"}, "test-driver", set(),
        lambda *_: calls.append("verified-baseline"), phase, lambda: gpu)
    with patch.object(recovery, "read", return_value=baseline):
        validating.verify("synthetic", {})
    assert calls[-1] == "verified-baseline"
    broken = {"x": {"attempts": baseline["x"]["attempts"]+[{"label": "new-a1", "status": "failed"}]}}
    with patch.object(recovery, "read", return_value=broken), patch.object(validating, "gpu_query") as query:
        try:
            validating.verify("synthetic", {})
        except recovery.RecoveryHold:
            pass
        else:
            raise AssertionError("New validation failure permitted a further attempt")
        query.assert_not_called()
    receipt = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "passed",
        "checks": ["child failure stops before next attempt", "evaluation args and budget passed unchanged",
            "training rejected before process launch", "historical failures retained and tolerated", "new validation failure stops before GPU check"],
        "CPU_only": True, "no_actual_model_generation_or_training": True,
        "guard_source_sha256": sha256_file(ROOT / "analysis/resume_large_filter_after_gpu_recovery.py"),
        "test_source_sha256": sha256_file(ROOT / "analysis/check_gpu_recovery_failure_guard.py")}
    path = ROOT / "experiments/desktopfilterrep_2026-11-10_235416/gpu_recovery_failure_guard_regression_20261113.json"
    assert not path.exists()
    json_write(path, receipt)
    print(receipt)


if __name__ == "__main__":
    main()
