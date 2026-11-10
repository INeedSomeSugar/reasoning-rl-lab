"""Compact read-only progress; never uses partial development accuracy to tune."""
import json
import sys
import time
from pathlib import Path
from reasoning_rl.runtime import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from filter_common import EXP, FAMILY, read


def complete_lines(path):
    rows=[]
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # last append may still be in progress; never treat it as an observed row
    return rows


def main():
    state=read(ROOT/"environment"/(FAMILY+"-status.json"))
    phase=state.get("phase")
    result={"status":state["status"],"phase":phase,"completed_training":state["completed_training"],
        "completed_evaluation":state["completed_evaluation"],"session_remaining_seconds":read(EXP/"session.json")["deadline_unix"]-time.time()}
    directory=ROOT/"runs"/(state.get("run_prefix", FAMILY)+"-"+phase)
    if phase and phase.endswith("-development"):
        result["development_predictions_completed"]=len(complete_lines(directory/"validation/predictions.jsonl"))
    elif directory.exists():
        costs=complete_lines(directory/"costs.jsonl")
        grads=[r for r in costs if r["phase"] == "gradient_diagnostic"]
        updates=[r for r in costs if r["phase"] == "optimizer_weight_update"]
        signals=complete_lines(directory/"rollout_group_signal.jsonl")
        result.update(optimizer_steps_completed=len(updates),nonzero_gradient_steps=sum(r["nonzero_gradient_elements"] > 0 for r in grads),
            finite_gradient_checks=all(r["finite_gradients"] and r["frozen_parameter_gradients"] == 0 for r in grads),
            actual_loss_mask_and_advantage_checks=len(signals),
            zero_gradient_steps_with_actual_weight_changes=sum(r["zero_gradient_step"] and r["actual_changed_weight_elements"] > 0 for r in updates))
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
