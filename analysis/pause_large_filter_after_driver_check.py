"""Pause between complete evaluations without modifying the frozen controller.

An ownership-distinct GPU lock fences the next launch. The current child and
controller finish normally, preserving their original audit and cost accounting.
Only after the current first-success record is durable is the controller stopped.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_append, sha256_file
from pause_large_filter_after_endpoint import open_handle, exit_info, K32
from large_filter_common import read, write_new, utc_now, host_snapshot
from run_large_filter import verify_frozen


def selfcheck(exp):
    directory = exp / ("evaluation_pause_guard_selfcheck_"+str(time.time_ns()))
    directory.mkdir(exist_ok=False)
    lock = directory / "synthetic_lock.json"
    json_write(lock, {"runner_pid": os.getpid(), "label": "pause-guard"})
    synthetic_controller = os.getpid()+1
    owner = read(lock)
    if owner.get("runner_pid") == synthetic_controller and owner.get("label") == "current-eval":
        lock.unlink()
    assert lock.exists()
    try:
        os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Guard did not fence the next model launch")
    result = {"status": "passed", "CPU_only": True, "current_owner_finally_preserves_other_owner_lock": True,
        "next_exclusive_model_launch_blocked": True, "frozen_runtime_unchanged": True}
    json_write(directory / "result.json", result)
    print(result, flush=True)


def main(family, parent_pid, expected_label):
    exp = ROOT / "experiments" / family
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    assert (exp / "all_training_endpoints_fixed.json").exists()
    assert (exp / "fresh_development_opened.json").exists()
    assert not (exp / "all_model_tasks_complete.json").exists()
    heartbeat = read(exp / "heartbeat.json")
    assert heartbeat["runner_pid"] == parent_pid and heartbeat["phase"] == expected_label
    assert time.time()-__import__("datetime").datetime.fromisoformat(heartbeat["heartbeat_utc"]).timestamp() < 60
    parent = psutil.Process(parent_pid)
    assert "resume_large_filter_after_gpu_recovery.py" in " ".join(parent.cmdline())
    child_pid = heartbeat["child_pid"]
    child = psutil.Process(child_pid)
    assert family+"-"+expected_label in " ".join(child.cmdline())
    evaluations_path = exp / "evaluation_units.json"
    before = read(evaluations_path)
    matches = [(k, u) for k, u in before.items() if u.get("status") == "running_attempt"
        and u["attempts"][-1]["label"] == expected_label]
    assert len(matches) == 1
    key, item = matches[0]
    selected_attempt = item["attempts"][-1]["attempt"]
    command = [r for r in jsonl_read(exp / "commands.jsonl") if r["label"] == expected_label]
    assert len(command) == 1
    command = command[0]
    directory = exp / ("user_driver_recovery_pause_"+time.strftime("%Y-%m-%d_%H%M%S_UTC", time.gmtime()))
    directory.mkdir(exist_ok=False)
    request = {"requested_utc": utc_now(), "family": family, "target_key": key,
        "target_label": expected_label, "controller_pid": parent_pid, "child_pid": child_pid,
        "observer_pid": os.getpid(), "user_instruction": "User requests checks only and no training; fence all future GPU launches while preserving current evaluation pending stop choice",
        "source_sha256": sha256_file(Path(__file__)), "frozen_source_models_data_verified": True,
        "original_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"],
        "child_never_suspended": True, "current_controller_keeps_original_cost_accounting": True}
    write_new(directory / "request.json", request)
    json_write(directory / "evaluation_units_before.json", before)
    lock_path = ROOT / "environment/large_filter_gpu.lock.json"
    handle = open_handle(child_pid)
    parent_suspended = False
    try:
        parent.suspend()
        parent_suspended = True
        current_lock = read(lock_path)
        assert current_lock["family"] == family and current_lock["runner_pid"] == parent_pid
        assert current_lock["label"] == expected_label
        lock_path.rename(directory / "current_model_lock_before_pause_guard.json")
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            __import__("json").dump({"family": family, "label": "user-evaluation-pause-guard",
                "runner_pid": os.getpid(), "started_unix": time.time(), "purpose": "Fence the next model launch while the current evaluation finishes",
                "current_controller_pid": parent_pid, "current_label": expected_label}, f)
        json_write(exp / "driver_recovery_evaluation_pause_armed.json", request)
        jsonl_append(exp / "process_events.jsonl", [{"event": "user_requested_pause_after_current_evaluation_armed",
            "label": expected_label, "runner_pid": parent_pid, "child_pid": child_pid,
            "observer_pid": os.getpid(), "time_unix": time.time(), "next_launch_fenced": True}])
        parent.resume()
        parent_suspended = False
        state = read(exp / "runner_state.json")
        state.update(status="pausing_after_current_evaluation", updated_utc=utc_now(), pause_observer_pid=os.getpid(),
            evaluation_pause_request=directory.relative_to(ROOT).as_posix())
        json_write(exp / "runner_state.json", state)
        last_heartbeat = 0
        while True:
            now = time.time()
            if now-last_heartbeat >= 15:
                json_write(directory / "observer_heartbeat.json", {"heartbeat_utc": utc_now(),
                    "controller_pid": parent_pid, "child_pid": child_pid, "observer_pid": os.getpid(),
                    "phase": expected_label, "next_model_launch_fenced": True, "host": host_snapshot()})
                last_heartbeat = now
            try:
                current = read(evaluations_path)
            except __import__("json").JSONDecodeError:
                time.sleep(.01)
                continue
            if current[key]["status"] == "first_success_fixed":
                break
            if current[key]["attempts"][-1].get("status") == "failed":
                raise RuntimeError("Current evaluation failed; next launch is fenced, inspect preserved evidence")
            if now-command["started_unix"] > command["timeout_seconds"]+90:
                raise TimeoutError("Current evaluation did not reach its original terminal boundary")
            time.sleep(.01)
        assert K32.WaitForSingleObject(handle, 0) == 0
        info = exit_info(handle)
        assert info["exit_code"] == 0
        if parent.is_running():
            parent.suspend()
            parent_suspended = True
        after = read(evaluations_path)
        completed = after[key]
        assert completed["selected_attempt"] == selected_attempt and completed["n"] == plan["development_n"]
        prediction_path = ROOT / "runs" / completed["run_name"] / "fresh_development/predictions.jsonl"
        assert sha256_file(prediction_path) == completed["predictions_sha256"]
        assert sum(bool(line.strip()) for line in prediction_path.open(encoding="utf-8")) == plan["development_n"]
        # A guard makes an unexpected scheduling race harmless: no next command,
        # log or model process can exist. Preserve any pre-launch metadata before
        # restoring only that unlaunched item's prior state.
        restored = []
        commands = jsonl_read(exp / "commands.jsonl")
        for other_key, old in before.items():
            if other_key == key or after[other_key] == old:
                continue
            new = after[other_key]
            extra = new["attempts"][len(old["attempts"]):]
            assert len(extra) == 1 and not any(r["label"] == extra[0]["label"] for r in commands)
            assert not (ROOT / "runs" / extra[0]["run_name"]).exists()
            assert not (ROOT / "logs" / (extra[0]["run_name"]+".log")).exists()
            restored.append({"key": other_key, "preserved_unlaunched_metadata": new,
                "restored_previous_state": old, "model_attempt_started": False})
            after[other_key] = old
        json_write(directory / "evaluation_units_after_current_saved_before_restoration.json", current)
        json_write(directory / "prelaunch_race_records.json", restored)
        if restored:
            json_write(evaluations_path, after)
        parent.terminate()
        parent.wait(timeout=10)
        parent_suspended = False
        guard = read(lock_path)
        assert guard["runner_pid"] == os.getpid() and guard["label"] == "user-evaluation-pause-guard"
        lock_path.rename(directory / "pause_guard_released_after_all_model_processes_finished.json")
        count = sum(u["status"] == "first_success_fixed" for u in after.values())
        paused = {"paused_utc": utc_now(), "family": family, "status": "paused_by_user_after_complete_evaluation",
            "completed_training": 35, "registered_training": 35, "completed_evaluation": count, "registered_evaluation": 37,
            "saved_evaluation_key": key, "saved_run_name": completed["run_name"],
            "saved_predictions_sha256": completed["predictions_sha256"], "n": completed["n"],
            "attempt_exit_info": info, "old_controller_terminated": parent_pid, "model_child_finished": child_pid,
            "no_next_model_started": True, "prelaunch_race_records": len(restored),
            "fresh_development_opened": True, "training_permanently_closed": True,
            "original_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"],
            "resume_policy": "Only after explicit user continue: skip all 35 training endpoints and completed evaluations; start next pending evaluation in a new process. No partial-evaluation resume."}
        write_new(directory / "completed.json", paused)
        write_new(exp / "pause_after_driver_recovery_evaluation_completed.json", paused)
        json_write(exp / "runner_state.json", {"updated_utc": utc_now(), "runner_pid": os.getpid(),
            "stage": "paused_after_evaluation", "status": "paused_by_user", "successful_training": 35,
            "successful_evaluation": count, "pause_receipt": "pause_after_driver_recovery_evaluation_completed.json"})
        print(paused, flush=True)
    except BaseException as error:
        json_write(directory / "observer_error.json", {"recorded_utc": utc_now(), "error": repr(error),
            "traceback": traceback.format_exc(), "controller_was_suspended": parent_suspended,
            "next_model_launch_remains_fenced_until_inspection": True})
        raise
    finally:
        K32.CloseHandle(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--controller", type=int)
    parser.add_argument("--label")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(ROOT / "experiments" / args.family)
    else:
        assert args.controller and args.label
        main(args.family, args.controller, args.label)

