"""Two old-data GRPO steps verify the new full-process clock before main freeze."""
import argparse
import sys
import time
import yaml
from reasoning_rl.runtime import ROOT, sha256_file, json_write
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, run_phase, START, START_HASH

p = argparse.ArgumentParser()
p.add_argument("--family", required=True)
args = p.parse_args()
family = args.family
exp = ROOT / "experiments" / family
target = exp / "full_process_clock_smoke.json"
if target.exists():
    raise FileExistsError(target)
cfg = yaml.safe_load((ROOT / "configs" / f"{family}_calibration_train.yaml").read_text())
cfg["grpo"].update(steps=2, max_wall_seconds=90, account_outer_process_start=True,
    experiment_arm="old_data_full_clock_smoke")
path = ROOT / "configs" / f"{family}_accounting_clock_smoke.yaml"
if path.exists():
    raise FileExistsError(path)
path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
outer = time.time()
for item in read(exp / "preflight.json")["public_model_files"]:
    assert sha256_file(ROOT / item["path"]) == item["sha256"]
name = family+"-accounting-clock-smoke-a0"
receipt = run_phase(family, "accounting-clock-smoke-a0", ["-m", "reasoning_rl", "grpo", "--name", name,
    "--hardware", "rocm", "--config", path.relative_to(ROOT).as_posix(), "--strategy", "uniform",
    "--sft-checkpoint", START], 150, budget_start_unix=outer)
assert receipt["exit_code"] == 0 and not receipt["timeout"]
contract = read(ROOT / "runs" / name / "comparison_contract.json")
result = read(ROOT / "runs" / name / "training_result.json")
assert contract["outer_budget_start_unix"] == outer
assert result["completed_optimizer_steps"] == 2 and result["initial_checkpoint_sha256"] == START_HASH
assert result["arm_total_through_checkpoint_seconds"] > receipt["started_unix"]-outer
json_write(target, {"passed": True, "outer_budget_start_unix": outer,
    "parent_model_cache_hash_verification_seconds": receipt["started_unix"]-outer,
    "receipt": receipt, "steps": 2, "recorded_by_child": contract["outer_budget_start_unix"],
    "fresh_development_inference": False, "only_historical_training_items": True})
print({"full_process_clock_smoke": "passed", "steps": 2})
