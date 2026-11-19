"""Read-only compact progress for the multi-day controller; never loads a model."""
import argparse
import json
from pathlib import Path
import time
import psutil
from reasoning_rl.runtime import ROOT


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.05)  # A live controller may be atomically between truncation and write.


def tail_json(path, max_bytes=131072):
    if not path.exists():
        return []
    with path.open("rb") as f:
        size = path.stat().st_size
        f.seek(max(0, size-max_bytes))
        content = f.read().decode("utf-8", errors="replace").splitlines()
    result = []
    for line in content:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def status(family):
    exp = ROOT / "experiments" / family
    state = read(exp / "runner_state.json", {})
    heartbeat = read(exp / "heartbeat.json", {})
    units = read(exp / "training_units.json", {})
    evaluations = read(exp / "evaluation_units.json", {})
    heartbeat_age = None if not heartbeat else time.time()-__import__("datetime").datetime.fromisoformat(heartbeat["heartbeat_utc"]).timestamp()
    visible = psutil.pid_exists(state["runner_pid"]) if state.get("runner_pid") else False
    result = {"family": family, "stage": state.get("stage"), "phase": state.get("phase"), "status": state.get("status"),
        "runner_pid": state.get("runner_pid"), "runner_pid_visible_to_this_reader": visible,
        "runner_liveness": "PID visible" if visible else "recent heartbeat; host PID may be outside reader namespace" if heartbeat_age is not None and heartbeat_age < 120 else "unconfirmed; inspect host PID before concluding exit",
        "successful_training": sum(r["status"] == "first_success_fixed" for r in units.values()),
        "failed_training_units": [k for k,r in units.items() if r["status"] == "exhausted_failure"],
        "successful_evaluation": sum(r["status"] == "first_success_fixed" for r in evaluations.values()),
        "failed_evaluation_units": [k for k,r in evaluations.items() if r["status"] in {"exhausted_failure", "missing_training_endpoint"}],
        "current_phase_process_seconds": heartbeat.get("elapsed_seconds"), "current_phase_timeout_seconds": heartbeat.get("timeout_seconds"),
        "heartbeat_age_seconds": heartbeat_age,
        "all_training_fixed": (exp / "all_training_endpoints_fixed.json").exists(),
        "fresh_development_opened": (exp / "fresh_development_opened.json").exists(),
        "all_model_tasks_complete": read(exp / "all_model_tasks_complete.json"), "error": state.get("error")}
    result["controller_suspended_for_user_boundary_pause"] = heartbeat.get("controller_suspended_for_user_boundary_pause", False)
    result["pause_observer_pid"] = state.get("pause_observer_pid", heartbeat.get("pause_observer_pid"))
    result["historical_training_pause_completed"] = read(exp / "pause_after_endpoint_completed.json")
    result["historical_evaluation_pause_completed"] = read(exp / "pause_after_evaluation_completed.json")
    result["historical_driver_recovery_pause_completed"] = read(exp / "pause_after_driver_recovery_evaluation_completed.json")
    continuations = [read(p) for p in exp.glob("evaluation_continuation_*/pause_handoff.json")]
    pauses = [item for item in (result["historical_training_pause_completed"],
        result["historical_evaluation_pause_completed"], result["historical_driver_recovery_pause_completed"], *continuations) if item]
    result["pause_completed"] = max(pauses, key=lambda item: item["paused_utc"]) if pauses else None
    result["pause_currently_active"] = state.get("status") == "paused_by_user"
    pause_errors = sorted(list(exp.glob("user_boundary_pause_*/observer_error.json"))
        +list(exp.glob("user_evaluation_boundary_pause_*/observer_error.json"))
        +list(exp.glob("user_driver_recovery_pause_*/observer_error.json")))
    result["boundary_pause_error"] = read(pause_errors[-1]) if pause_errors else None
    if state.get("status") == "paused_by_user":
        result["runner_liveness"] = "Paused by explicit user request; no model task expected"
    phase = state.get("phase")
    if phase:
        directory = ROOT / "runs" / f"{family}-{phase}"
        costs = tail_json(directory / "costs.jsonl")
        progress = [r for r in costs if r.get("phase") == "budget_progress"]
        if progress:
            last = progress[-1]
            result.update(current_optimizer_steps=last["optimizer_step"],
                current_generated_tokens=last.get("generated_tokens"),
                current_zero_variance_fraction=last.get("zero_variance_group_fraction"))
        if phase.startswith("eval-"):
            path = directory / "fresh_development/predictions.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    result["current_evaluation_predictions"] = sum(1 for line in f if line.strip())
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    args = p.parse_args()
    print(json.dumps(status(args.family), ensure_ascii=False))
