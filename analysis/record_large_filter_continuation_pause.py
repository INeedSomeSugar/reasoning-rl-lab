"""Verify a requested saved evaluation boundary and write the local handoff."""
import argparse
import datetime as dt
import os
import re
import statistics

import psutil
from reasoning_rl.runtime import ROOT, checkpoint_hash, jsonl_read, sha256_file
from resume_large_filter_after_gpu_recovery import host_gpu
from large_filter_common import read, utc_now, write_new
from run_large_filter import verify_frozen


def main(family, continuation_name):
    exp = ROOT / "experiments" / family
    directory = exp / continuation_name
    request = read(directory / "pause_requested.json")
    pause = read(directory / "paused.json")
    state = read(exp / "runner_state.json")
    assert state == {k: pause[k] for k in state}
    assert state["status"] == "paused_by_user" and state["runner_pid"] is None
    assert state["continuation"] == directory.relative_to(ROOT).as_posix()
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    assert not (exp / "all_model_tasks_complete.json").exists()
    runner_pid = request["runner_pid"]
    processes = []
    markers = ("reasoning_rl", "run_large_filter", "resume_large_filter", "continue_large_filter_evaluations",
               "large_filter_tasks", "pause_large_filter_after_driver_check", "check_large_filter_driver_recovery")
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        if proc.pid == os.getpid() or not (proc.info["name"] or "").casefold().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or [])
        local = str(ROOT).casefold() in command.casefold() or str(proc.info["cwd"]).casefold() == str(ROOT).casefold()
        if (local or proc.pid == runner_pid) and any(marker in command for marker in markers):
            processes.append({"pid": proc.pid, "command": command})
    assert not processes, processes
    gpu = host_gpu()
    assert gpu["driver"]["DriverVersion"] == read(directory / "resume_record.json")["GPU"]["driver"]["DriverVersion"]
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    locked = read(exp / "all_training_endpoints_fixed.json")
    assert locked["all35_successful"] and len(locked["objects"]) == 37
    for key, obj in locked["objects"].items():
        if obj["checkpoint"]:
            assert checkpoint_hash(ROOT / obj["checkpoint"]) == obj["checkpoint_sha256"], key
    units = read(exp / "evaluation_units.json")
    complete = [k for k in plan["evaluation_order"] if units[k]["status"] == "first_success_fixed"]
    remaining = [k for k in plan["evaluation_order"] if units[k]["status"] in ("pending", "pending_retry")]
    exhausted = [k for k in plan["evaluation_order"] if units[k]["status"] == "exhausted_failure"]
    assert len(complete)+len(exhausted)+len(remaining) == 37 and len(complete) == state["successful_evaluation"]
    assert len(complete) == request["complete_evaluations_before_request"]+1
    integrity = read(exp / (f"evaluation_integrity_through_{len(complete)}.json"))
    saved = {r["endpoint"]: r["predictions_sha256"] for r in integrity["records"]}
    assert set(saved) == set(complete)
    assert all(units[k]["predictions_sha256"] == v for k, v in saved.items())
    resume = read(directory / "resume_record.json")
    assert all(saved[k] == digest for k, digest in resume["complete_prediction_sha256"].items())
    assert exhausted == resume["exhausted_objects"]
    label = request["active_phase_at_request"]
    commands = jsonl_read(exp / "commands.jsonl")
    assert commands[-1]["label"] == label
    receipts = jsonl_read(exp / "process_costs.jsonl")
    last = receipts[-1]
    assert last["label"] == label and last["exit_code"] == 0 and not last["timeout"]
    match = [k for k in complete if units[k]["attempts"][units[k]["selected_attempt"]]["label"] == label]
    assert len(match) == 1
    key = match[0]
    current_path = ROOT / "runs" / units[key]["run_name"] / "fresh_development/predictions.jsonl"
    assert sha256_file(current_path) == saved[key]
    assert dt.datetime.fromisoformat(request["requested_utc"]).timestamp() < last["finished_unix"]
    assert last["finished_unix"] <= dt.datetime.fromisoformat(pause["updated_utc"]).timestamp()
    durations = [units[k]["evaluation_process_seconds"] for k in complete if k not in {"A", "S2"}]
    record = {"paused_utc": pause["updated_utc"], "verified_utc": utc_now(), "status": "paused_by_user_after_complete_evaluation",
        "family": family, "completed_training": 35, "completed_evaluation": len(complete), "saved_evaluation_key": key,
        "saved_prediction_n": plan["development_n"], "last_attempt": last, "no_next_model_started": True,
        "active_project_model_or_controller_processes": processes, "gpu_lock_present": False, "GPU": gpu,
        "exhausted": exhausted, "pending_order": remaining, "saved_complete_prediction_hashes": saved,
        "remaining_model_hours_at_observed_mean": len(remaining)*statistics.mean(durations)/3600,
        "original_deadline_unix": plan["session_limit"]["deadline_unix"],
        "files": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in (exp / "evaluation_units.json",
            exp / "runner_state.json", exp / "all_training_endpoints_fixed.json", exp / "protocol_frozen.json",
            directory / "pause_requested.json", directory / "paused.json")}}
    record_path = directory / "pause_handoff.json"
    write_new(record_path, record)
    local = dt.datetime.fromisoformat(record["paused_utc"]).astimezone(dt.timezone(dt.timedelta(hours=8)))
    rel = record_path.relative_to(ROOT).as_posix()
    report = ROOT / "reports/comparisons" / family / "evaluation_pause_20261118.md"
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"""# 11 月 18 日评估保存后暂停

北京时间 {local:%H:%M:%S}，`{label}` 完成全部 {plan['development_n']} 题，退出码 0。独立审计检查了现有 {len(complete)} 份完整预测的题目顺序、适配器与 SHA256。控制器在保存后退出，下一项尚未启动。主机核验没有本项目模型或控制器进程，GPU 锁已释放。

本家族训练 35/35 完成，评估 {len(complete)}/37 完整。{len(exhausted)} 项耗尽重试，仍保留训练权重及全部失败记录。冻结顺序中的待运行对象为 {', '.join(remaining)}。按当前实测均值，这 {len(remaining)} 项合计约需 {record['remaining_model_hours_at_observed_mean']:.1f} 小时评估，统计、盲审和报告另计。后续由用户明确恢复，训练阶段已锁定。

同一 RX 9070 XT 和重装后的驱动已在本次开机周期通过检查。前瞻结论等到剩余对象终止后按完整矩阵缺失范围报告。正式 GSM8K test 和 SVAMP 继续封存。

[本次暂停收据](../../../{rel}) · [当前状态](../../../experiments/{family}/runner_state.json) · [冻结协议](../../../experiments/{family}/protocol_frozen.json)
""")
    previous = exp / "AGENT_WORK_STATE.md"
    prefix = f"""## 2026-11-18 user pause verified after saved evaluation

The latest user instruction is to pause at this saved boundary. At {local.isoformat()}, {key} saved all {plan['development_n']} predictions and exited 0. Current 35 training, {len(complete)} complete evaluations, {len(exhausted)} exhausted, {len(remaining)} pending. Controller and GPU children exited and no GPU lock. Any GPU resumption requires an explicit user continue. Handoff {rel}. Continue the frozen remaining order ({', '.join(remaining)}) via the same continuation wrapper with that handoff as --prior-receipt. Keep same hard deadline and attempt limits.

During this turn the incomplete-matrix addendum and code were completed before any comparative fresh-outcome inspection. CPU regression suite 79 passed. After model completion, run offline analyze, prepare blind units, actually annotate each unique response before unblinding, render all eight figures, write Chinese findings and final local manifest. Read the plan and preserve the earlier operational visibility disclosure. Do not treat these preparations as completed analysis or review. No remote publication.

"""
    previous.write_text(prefix+previous.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 11 月 18 日保存后暂停

`{family}` 的 35 个训练终点均已锁定。按用户要求，在 `{key}` 保存全部 {plan['development_n']} 题后暂停。当前 {len(complete)}/37 份完整评估，{len(remaining)} 项待运行，{len(exhausted)} 项已耗尽重试。主机核验模型和控制器均已退出。用户恢复后从 `{remaining[0]}` 按冻结顺序继续，正式 GSM8K test 与 SVAMP 保持封存。

[暂停记录](reports/comparisons/{family}/evaluation_pause_20261118.md) · [收据]({rel}) · [冻结协议](experiments/{family}/protocol_frozen.json) · [复现](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)
<!-- LARGE_FILTER_CURRENT_END -->"""
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        content, n = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->",
            lambda _: block, path.read_text(encoding="utf-8"), flags=re.S)
        assert n == 1
        path.write_text(content, encoding="utf-8", newline="\n")
    reproduction = ROOT / "docs/REPRODUCING_FILTER_REPLICATION_20261110.md"
    previous_text = reproduction.read_text(encoding="utf-8")
    title, content = previous_text.split("\n", 1)
    note = f"\n> 11 月 18 日 {local:%H:%M:%S} 保存后暂停，当前 {len(complete)}/37 完整评估。实际状态见 [本次暂停收据](../{rel})。\n"
    reproduction.write_text(title+"\n"+note+content, encoding="utf-8", newline="\n")
    print({"paused": local.isoformat(), "complete": len(complete), "pending": remaining, "exhausted": len(exhausted),
        "remaining_model_hours": record["remaining_model_hours_at_observed_mean"], "receipt": str(record_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--continuation", required=True)
    args = parser.parse_args()
    main(args.family, args.continuation)
