"""Cost-only calibration without selecting any model or hyperparameter by accuracy."""
import json
import yaml
from reasoning_rl.runtime import ROOT, json_write, sha256_file
from filter_common import EXP, FAMILY, START, read, run_phase


def main():
    reg = read(EXP / "calibration_preregistration.json")
    json_write(EXP / "calibration_pool.json", {"ids": reg["calibration_train_ids"],
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json")})
    config = yaml.safe_load((ROOT / "configs" / f"{FAMILY}_calibration.yaml").read_text())
    config["grpo"].update(candidate_pool_path=f"experiments/{FAMILY}/calibration_pool.json",
        steps=reg["calibration_train_optimizer_steps"], max_wall_seconds=240,
        standalone_endpoints_only=True, experiment_arm="calibration_U", save_optimizer_state=True)
    path = ROOT / "configs" / f"{FAMILY}_calibration_train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    run_phase("calibration-probe", ["scripts/run_filter_probe.py", "--calibration"], timeout=300)
    run_phase("calibration-train", ["-m", "reasoning_rl", "grpo", "--name", FAMILY+"-calibration-train",
        "--config", str(path.relative_to(ROOT)), "--hardware", "rocm", "--strategy", "uniform",
        "--sft-checkpoint", START], timeout=300)
    result = read(ROOT / "runs" / (FAMILY+"-calibration-train") / "training_result.json")
    assert result["completed_optimizer_steps"] == reg["calibration_train_optimizer_steps"]
    assert result["actual_signal_checked_groups"] == result["groups"]
    json_write(EXP / "calibration_complete.json", {"completed": True,
        "actual_optimizer_steps": result["completed_optimizer_steps"],
        "mask_and_advantage_tensor_checks": result["actual_signal_checked_groups"],
        "accuracy_not_used_for_capacity": True})


if __name__ == "__main__":
    main()
