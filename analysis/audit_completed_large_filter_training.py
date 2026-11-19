"""CPU-only independent raw-evidence audit of completed training, never fresh evaluation."""
import argparse
import sys
from reasoning_rl.runtime import ROOT, json_write, sha256_file
from safetensors.torch import load_file
sys.path.insert(0, str(ROOT / "analysis"))
import analyze_large_filter as analysis

p = argparse.ArgumentParser()
p.add_argument("--family", required=True)
args = p.parse_args()
exp = ROOT / "experiments" / args.family
units = analysis.read(exp / "training_units.json")
s2 = load_file(ROOT / analysis.START / "adapter_model.safetensors")
done = []
for key, unit in units.items():
    if unit["status"] != "first_success_fixed":
        continue
    target = exp / "completed_training_audits" / (key+".json")
    if target.exists():
        continue
    result, windows, coverage = analysis.audited_training(unit, s2)
    json_write(target, {"endpoint": key, "status": "independent_CPU_audit_passed", "diagnostics": result,
        "signal_windows": windows, "coverage_records": len(coverage), "fresh_development_predictions_read": False,
        "training_result_sha256": sha256_file(ROOT / "runs" / unit["run_name"] / "training_result.json"),
        "checkpoint_sha256": unit["checkpoint_sha256"]})
    done.append({"endpoint": key, "steps": result["optimizer_steps"], "effective_completions": result["effective_completions"],
        "nonzero_gradient_steps": result["nonzero_gradient_steps"], "delta_l2": result["adapter_delta_l2_independent"]})
print(done)
