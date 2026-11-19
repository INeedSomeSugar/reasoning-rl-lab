"""Execute only the four preregistered candidates, retaining failures and process wall time."""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_append, sha256_file, checkpoint_hash

FAMILY = "desktoprepair_20261106_152712"
EXP = ROOT / "experiments" / FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if not (ROOT / "runs" / (FAMILY + "-sft-contract") / "audit_result.json").exists():
        raise ValueError("Real-model contract audit must finish before SFT")
    plan = read(EXP / "sft_preregistration.json")
    deadline = dt.datetime.fromisoformat(read(EXP / "session.json")["deadline_utc"]).timestamp()
    status_path = ROOT / "environment" / (FAMILY + "-sft-status.json")
    if status_path.exists():
        raise FileExistsError("Preserve phase status; use explicit recovery for any failure")
    state = {"status": "running", "started_unix": time.time(), "completed": [], "holdout_opened": False}

    def phase(label, args):
        log = ROOT / "logs" / f"{FAMILY}-{label}.log"
        if log.exists():
            raise FileExistsError(log)
        state.update(phase=label)
        json_write(status_path, state)
        started = time.time()
        command = [sys.executable, "-u", *args]
        jsonl_append(EXP / "commands.jsonl", [{"phase": label, "argv": command, "started_unix": started}])
        with log.open("x", encoding="utf-8") as out:
            process = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                     timeout=max(1, deadline-time.time()-900), check=False)
        jsonl_append(EXP / "process_costs.jsonl", [{"phase": label, "process_wall_seconds": time.time()-started,
            "exit_code": process.returncode, "started_unix": started, "finished_unix": time.time()}])
        if process.returncode:
            raise RuntimeError(f"{label}: exit {process.returncode}; preserve {log}")
        state["completed"].append(label)
        json_write(status_path, state)

    try:
        phase("A-comparison", ["-m", "reasoning_rl", "evaluate", "--name", FAMILY+"-A-comparison",
            "--hardware", "rocm", "--config", plan["configs"]["eval"]])
        for candidate in plan["candidates"]:
            label = candidate["name"]
            name = FAMILY + "-" + label
            cfg = plan["configs"][label]
            phase(label, ["-m", "reasoning_rl", "sft", "--name", name, "--hardware", "rocm", "--config", cfg])
            for subset, key in (("comparison", "eval"), ("regression", "regression")):
                phase(label+"-"+subset, ["-m", "reasoning_rl", "evaluate", "--name", name+"-"+subset,
                    "--hardware", "rocm", "--config", plan["configs"][key], "--adapter", "checkpoints/"+name])
        base = read(ROOT / "runs" / (FAMILY+"-A-comparison") / "validation/metrics.json")
        records = []
        for candidate in plan["candidates"]:
            label = candidate["name"]
            directory = ROOT / "runs" / (FAMILY + "-" + label)
            result = read(directory / "training_result.json")
            metrics = read(ROOT / "runs" / (FAMILY+"-"+label+"-comparison") / "validation/metrics.json")
            n = metrics["n"]
            gradients = [json.loads(line) for line in (directory / "costs.jsonl").read_text().splitlines()
                         if json.loads(line).get("phase") == "gradient_diagnostic"]
            eligible = metrics["correct"] >= base["correct"]-2 and (
                metrics["truncation_rate"]*n <= base["truncation_rate"]*n+2) and (
                metrics["parse_error_rate"]*n <= base["parse_error_rate"]*n+2) and result["adapter_updated"] and (
                all(r["finite_gradients"] and not r["frozen_parameter_gradients"] for r in gradients))
            records.append({**candidate, "metrics": metrics, "eligible": eligible,
                "adapter_delta_l2": result["adapter_delta_l2"], "checkpoint_sha256": result["checkpoint_sha256"],
                "reload_logits_max_abs_difference": result["reload_logits_max_absolute_difference"]})
        eligible = [r for r in records if r["eligible"]]
        selected = min(eligible, key=lambda r: (-r["metrics"]["correct"], r["metrics"]["truncation_rate"],
            r["learning_rate"]*r["steps"], [c["name"] for c in plan["candidates"]].index(r["name"]))) if eligible else None
        decision = {"baseline": base, "candidates": records, "selection_uses_comparison_only": True,
            "route": "real_sft" if selected else "public_posttrained_A",
            "selected_candidate": selected["name"] if selected else None,
            "checkpoint": "checkpoints/"+FAMILY+"-"+selected["name"] if selected else None,
            "confirmation_not_opened": True, "preregistration_sha256": sha256_file(EXP / "sft_preregistration.json")}
        json_write(EXP / "sft_decision.json", decision)
        state.update(status="completed_sft_diagnostics", finished_unix=time.time(), decision=decision["route"])
        json_write(status_path, state)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_unix=time.time())
        json_write(status_path, state)
        jsonl_append(EXP / "failures.jsonl", [{"phase": state.get("phase"), "error": repr(error), "time_unix": time.time()}])
        raise


if __name__ == "__main__":
    main()
