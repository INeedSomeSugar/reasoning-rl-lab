"""User-requested boundary pause: let the current child save, prevent the next job.

Suspend only the controller, never the model child. An independent Windows handle
records the child's actual exit code/time and enforces the same existing timeout.
The frozen training/scoring/configuration code is not edited or resumed mid-run.
"""
import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_append, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, write_new, write_state, utc_now, host_snapshot
from run_large_filter import audit_training, verify_frozen


K32 = ctypes.WinDLL("kernel32", use_last_error=True)
K32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
K32.OpenProcess.restype = wintypes.HANDLE
K32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
K32.WaitForSingleObject.restype = wintypes.DWORD
K32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
K32.GetExitCodeProcess.restype = wintypes.BOOL
K32.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
K32.GetProcessTimes.restype = wintypes.BOOL
K32.CloseHandle.argtypes = [wintypes.HANDLE]


def open_handle(pid):
    handle = K32.OpenProcess(0x1000 | 0x00100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def exit_info(handle):
    code = wintypes.DWORD()
    created, ended, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not K32.GetExitCodeProcess(handle, ctypes.byref(code)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not K32.GetProcessTimes(handle, *[ctypes.byref(v) for v in (created, ended, kernel, user)]):
        raise ctypes.WinError(ctypes.get_last_error())
    def unix(value):
        return ((value.dwHighDateTime << 32)+value.dwLowDateTime)/10_000_000-11_644_473_600
    return {"exit_code": int(code.value), "child_created_unix": unix(created), "child_exit_unix": unix(ended),
        "timing_source": "Windows GetProcessTimes held process handle; not controller resume time"}


def selfcheck():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.4)"], cwd=ROOT,
        creationflags=subprocess.CREATE_NO_WINDOW)
    handle = open_handle(child.pid)
    try:
        owned = psutil.Process(child.pid)
        owned.suspend()
        time.sleep(.05)
        assert K32.WaitForSingleObject(handle, 0) == 258
        owned.resume()
        assert K32.WaitForSingleObject(handle, 5000) == 0
        info = exit_info(handle)
        assert info["exit_code"] == 0 and info["child_exit_unix"] >= info["child_created_unix"]
        assert child.wait(timeout=2) == 0
        return {"status": "passed", "CPU_only_synthetic_process": True,
            "suspend_resume_and_exact_exit_handle_verified": True, **info}
    finally:
        K32.CloseHandle(handle)


def main(family, parent_pid, child_pid):
    exp = ROOT / "experiments" / family
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    assert not (exp / "fresh_development_opened.json").exists()
    units_path = exp / "training_units.json"
    units = read(units_path)
    item = units["U-s47"]
    assert item["status"] == "running_attempt" and len(item["attempts"]) == 2
    attempt = item["attempts"][-1]
    assert attempt["label"] == "U-s47-a1" and attempt["attempt"] == 1
    parent, child = psutil.Process(parent_pid), psutil.Process(child_pid)
    assert "resume_large_filter_after_restart.py" in " ".join(parent.cmdline())
    assert family+"-U-s47-a1" in " ".join(child.cmdline())
    heartbeat = read(exp / "heartbeat.json")
    assert heartbeat["runner_pid"] == parent_pid and heartbeat["child_pid"] == child_pid
    assert heartbeat["phase"] == "U-s47-a1"
    command = [r for r in jsonl_read(exp / "commands.jsonl") if r["label"] == "U-s47-a1"]
    assert len(command) == 1
    command = command[0]
    pause_dir = exp / ("user_boundary_pause_"+time.strftime("%Y-%m-%d_%H%M%S_UTC", time.gmtime()))
    pause_dir.mkdir(exist_ok=False)
    write_new(pause_dir / "request.json", {"requested_utc": utc_now(), "family": family,
        "user_instruction": "Wait for current endpoint to save, then pause without starting another unit",
        "parent_pid": parent_pid, "child_pid": child_pid, "observer_pid": os.getpid(),
        "heartbeat_before_suspension": heartbeat, "child_not_suspended": True,
        "same_timeout_and_original_session_deadline": True, "frozen_source_verified": True,
        "source_sha256": sha256_file(Path(__file__))})
    handle = open_handle(child_pid)
    parent_suspended = False
    try:
        parent.suspend()
        parent_suspended = True
        jsonl_append(exp / "process_events.jsonl", [{"event": "user_requested_pause_after_current_endpoint_armed",
            "label": "U-s47-a1", "runner_pid": parent_pid, "child_pid": child_pid,
            "observer_pid": os.getpid(), "time_unix": time.time(), "controller_only_suspended": True}])
        write_state(family, stage="main_training", phase="U-s47-a1", status="pausing_after_current_endpoint",
            actual_controller_pid=parent_pid, pause_observer_pid=os.getpid(), child_pid=child_pid)
        last_heartbeat = 0
        timed_out = False
        while K32.WaitForSingleObject(handle, 1000) == 258:
            now = time.time()
            if now-last_heartbeat >= 15:
                state = {"family": family, "runner_pid": parent_pid, "child_pid": child_pid,
                    "phase": "U-s47-a1", "elapsed_seconds": now-command["started_unix"],
                    "timeout_seconds": command["timeout_seconds"], "heartbeat_utc": utc_now(), "host": host_snapshot(),
                    "controller_suspended_for_user_boundary_pause": True, "pause_observer_pid": os.getpid(),
                    "child_training_not_suspended": True}
                json_write(exp / "heartbeat.json", state)
                json_write(pause_dir / "observer_heartbeat.json", state)
                last_heartbeat = now
            if now-command["started_unix"] >= command["timeout_seconds"]:
                timed_out = True
                descendants = child.children(recursive=True)
                for process in reversed(descendants+[child]):
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(descendants+[child], timeout=5)
                for process in alive:
                    process.kill()
                assert K32.WaitForSingleObject(handle, 5000) == 0
                break
        info = exit_info(handle)
        assert info["exit_code"] != 259
        receipt = {"label": "U-s47-a1", "argv": command["argv"], "pid": child_pid, "runner_pid": parent_pid,
            "started_unix": command["started_unix"], "finished_unix": info["child_exit_unix"],
            "process_wall_seconds": info["child_exit_unix"]-command["started_unix"],
            "outer_budget_start_unix": command["outer_budget_start_unix"], "exit_code": info["exit_code"],
            "timeout": timed_out, "timeout_seconds": command["timeout_seconds"],
            "log": "logs/"+family+"-U-s47-a1.log", "host_before": command["host_before"], "host_after": host_snapshot(),
            "recorded_by_user_pause_observer": True, **info}
        assert not [r for r in jsonl_read(exp / "process_costs.jsonl") if r["label"] == receipt["label"]]
        jsonl_append(exp / "process_costs.jsonl", [receipt])
        attempt["receipt"] = receipt
        audit_start = time.perf_counter()
        try:
            if info["exit_code"] or timed_out:
                raise RuntimeError("The current model attempt failed before boundary pause")
            result = audit_training(family, item, attempt["run_name"], 0)
        except BaseException as error:
            attempt.update(status="failed", error=repr(error))
            item["status"] = "exhausted_failure"
            jsonl_append(exp / "attempt_validation_failures.jsonl", [{"key": "U-s47", "attempt": 1,
                "error": repr(error), "traceback": traceback.format_exc(), "user_pause_observer": True}])
        else:
            attempt["status"] = "success"
            parent_preparation = command["started_unix"]-command["outer_budget_start_unix"]
            item.update(status="first_success_fixed", selected_attempt=1, run_name=attempt["run_name"],
                checkpoint="checkpoints/"+attempt["run_name"], checkpoint_sha256=result["checkpoint_sha256"],
                training_process_seconds=receipt["process_wall_seconds"],
                parent_training_preparation_and_audit_seconds=parent_preparation+time.perf_counter()-audit_start,
                probe_algorithm_debit_seconds=0, R_experimental_matching_debit_seconds=0,
                total_probe_or_matching_debit_seconds=0, completed_optimizer_steps=result["completed_optimizer_steps"])
        json_write(units_path, units)
        # The controller cannot launch anything while suspended. Its current
        # endpoint is now terminal and audited, so terminate only that controller.
        parent.terminate()
        parent.wait(timeout=10)
        lock_path = ROOT / "environment/large_filter_gpu.lock.json"
        if lock_path.exists():
            lock = read(lock_path)
            assert lock["family"] == family and lock["runner_pid"] == parent_pid and lock["label"] == "U-s47-a1"
            target = pause_dir / "gpu_lock_after_saved_endpoint.json"
            assert ROOT.resolve() in target.resolve().parents and not target.exists()
            lock_path.rename(target)
        completed = sum(u["status"] == "first_success_fixed" for u in units.values())
        paused = {"paused_utc": utc_now(), "family": family, "status": "paused_by_user_after_terminal_checkpoint",
            "completed_training": completed, "registered_training": 35, "completed_evaluation": 0,
            "current_endpoint_status": item["status"], "checkpoint": item.get("checkpoint"),
            "checkpoint_sha256": item.get("checkpoint_sha256"), "attempt_exit_info": info,
            "old_controller_terminated_after_checkpoint": parent_pid, "model_child_finished": child_pid,
            "no_next_unit_started": True, "fresh_development_opened": False,
            "original_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"],
            "exact_sampler_optimizer_resume_claimed": False,
            "resume_policy": "Skip successful fixed endpoints; begin the next pending unit from the same frozen S2/config. No mid-unit resume."}
        write_new(exp / "pause_after_endpoint_completed.json", paused)
        json_write(pause_dir / "completed.json", paused)
        write_state(family, stage="paused_after_checkpoint", status="paused_by_user", successful_training=completed,
            successful_evaluation=0, pause_receipt="pause_after_endpoint_completed.json")
        print(paused, flush=True)
    except BaseException as error:
        json_write(pause_dir / "observer_error.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "recorded_utc": utc_now(), "controller_was_suspended": parent_suspended,
            "do_not_resume_controller_without_inspecting_child_and_checkpoint": True})
        raise
    finally:
        K32.CloseHandle(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--controller", type=int)
    parser.add_argument("--child", type=int)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        result = selfcheck()
        json_write(ROOT / "experiments" / args.family / "boundary_pause_CPU_selfcheck.json", result)
        print(result)
    else:
        assert args.controller and args.child
        main(args.family, args.controller, args.child)
