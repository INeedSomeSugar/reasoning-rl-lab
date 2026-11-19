"""Independent engineering diagnostic after the frozen pilot; not a new quality comparison."""
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml
from reasoning_rl.runtime import ROOT, json_write

pilot = "desktoprocm10pilot_20261105"
status = json.loads((ROOT / f"environment/{pilot}-status.json").read_text(encoding="utf-8-sig"))
if status["status"] != "completed_development_pilot":
    raise RuntimeError("Finish and preserve the frozen paired pilot before a new diagnostic configuration")
name = "desktoprocm10refresh_20261105-D-s13"
config_path = ROOT / "configs/desktop_rocm10_bf16_refresh_check.yaml"
receipt = ROOT / "experiments/desktop_dynamic_refresh_verification_20261105.json"
if config_path.exists() or receipt.exists() or (ROOT / "runs" / name).exists():
    raise FileExistsError("Preserve the original diagnostic evidence")
config = yaml.safe_load((ROOT / "configs/desktop_rocm10_bf16_smoke.yaml").read_text(encoding="utf-8"))
config["stage"] = "desktop_rocm10_bf16_dynamic_refresh_engineering"
review = json.loads((ROOT / "experiments/desktop_expanded_train_review_20261105.json").read_text(encoding="utf-8"))
config["grpo"].update(pool_size=32, pool_offset=128,
    excluded_train_ids=[r["id"] for r in review["items"] if r["decision"] == "exclude"],
    steps=12, refresh_interval=4, validation_interval=12, max_wall_seconds=600)
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
json_write(receipt, {"status": "running", "scope": "Engineering only: verify live sampling-weight refresh, not a matched-cost or model-quality comparison",
    "reason": "The fixed 300-second pilot allowance could not finish a refreshed 28-item probe; smoke probes had identical empirical outcomes",
    "source_sft": "checkpoints/desktoprocm10bf16_20261105-B", "source_sft_steps": 4,
    "config": str(config_path.relative_to(ROOT)), "holdout_evaluated": False,
    "interpretation_limit": "Weight changes can reflect stochastic success-rate estimation; they do not establish a learning benefit."})
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
def execute(label, arguments):
    log = ROOT / f"logs/desktoprocm10refresh_20261105-{label}.log"
    if log.exists():
        raise FileExistsError(log)
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        code = subprocess.call([sys.executable, "-u", "-m", "reasoning_rl", *arguments], cwd=ROOT,
                               stdout=stream, stderr=subprocess.STDOUT)
    if code:
        failed = json.loads(receipt.read_text(encoding="utf-8"))
        failed.update(status="failed", failed_phase=label, exit_code=code, failure_log=str(log.relative_to(ROOT)))
        json_write(receipt, failed)
        raise RuntimeError(f"Diagnostic {label} failed, exit={code}; retain the full log")
execute("train", ["grpo", "--hardware", "rocm", "--config", str(config_path), "--name", name,
    "--sft-checkpoint", "checkpoints/desktoprocm10bf16_20261105-B", "--strategy", "dynamic"])
execute("reloadval", ["evaluate", "--hardware", "rocm", "--config", str(config_path),
    "--name", name+"-reloadval", "--adapter", "checkpoints/"+name])
result = json.loads((ROOT / "runs" / name / "training_result.json").read_text(encoding="utf-8"))
record = json.loads(receipt.read_text(encoding="utf-8"))
record.update(status="verified" if result["sampler_weight_change_events"] > 0 and result["adapter_updated"] and result["nonzero_advantage_completions"] > 0 else "completed_without_full_signal",
    probe_calls=result["probe_calls"], weight_change_events=result["sampler_weight_change_events"],
    completed_optimizer_steps=result["completed_optimizer_steps"], adapter_updated=result["adapter_updated"],
    checkpoint_sha256=result["checkpoint_sha256"], result=f"runs/{name}/training_result.json")
json_write(receipt, record)
print(json.dumps(record, indent=2), flush=True)
