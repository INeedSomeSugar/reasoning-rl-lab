"""CPU-only incident audit and handoff; never retries or reads response text."""
import argparse
import collections
import csv
import datetime as dt
import json
import re
import statistics
import sys
import time

from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, sha256_file

sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read
from run_large_filter import verify_frozen
from large_filter_cost_accounting import accounting


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    incident = exp / "gpu_hold_audit_2026-11-13"
    incident.mkdir(exist_ok=False)
    plan = read(exp / "protocol_frozen.json")
    units = read(exp / "evaluation_units.json")
    training = read(exp / "training_units.json")
    state = read(exp / "runner_state.json")
    assert state["status"] == "diagnostic_hold"
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    assert not (exp / "all_model_tasks_complete.json").exists()
    verify_frozen(family, plan)
    lock = read(exp / "all_training_endpoints_fixed.json")
    opened = read(exp / "fresh_development_opened.json")
    assert opened["endpoint_lock_sha256"] == sha256_file(exp / "all_training_endpoints_fixed.json")
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    verified_weights = []
    for key, obj in lock["objects"].items():
        if obj["checkpoint"]:
            assert checkpoint_hash(ROOT / obj["checkpoint"]) == obj["checkpoint_sha256"]
            verified_weights.append(key)
    assert len(training) == 35 and all(u["status"] == "first_success_fixed" for u in training.values())
    assert len(list((exp / "completed_training_audits").glob("*.json"))) == 35
    completed = [k for k in plan["evaluation_order"] if units[k]["status"] == "first_success_fixed"]
    exhausted = [k for k in plan["evaluation_order"] if units[k]["status"] == "exhausted_failure"]
    remaining = [k for k in plan["evaluation_order"] if units[k]["status"] in ("pending", "pending_retry")]
    assert len(completed) == 22 and len(exhausted) == 5 and len(remaining) == 10
    integrity = read(exp / "evaluation_integrity_through_22.json")
    assert {r["endpoint"] for r in integrity["records"]} == set(completed)
    for row in integrity["records"]:
        path = ROOT / "runs" / units[row["endpoint"]]["run_name"] / "fresh_development/predictions.jsonl"
        assert sha256_file(path) == row["predictions_sha256"]
    pnp = read(exp / "gpu_runtime_incident_pnp.json")
    rx = [d for d in pnp["pnp"] if d["FriendlyName"] == "AMD Radeon RX 9070 XT"]
    assert len(rx) == 1 and rx[0]["Present"] is False and rx[0]["Problem"] == 45
    host = read(exp / "gpu_runtime_incident_host_events.json")
    assert not host["active_project_models"]
    current_dumps = sorted({match for e in host["application_events"]
        for match in re.findall(r"[A-Z_]+-20261113-[0-9]+\.dmp", e["Message"])})
    durations = [units[k]["evaluation_process_seconds"] for k in completed if k in training]
    mean = statistics.mean(durations)
    physical, ledger, failures, _ = accounting(exp, training, units)
    with (incident / "all_attempt_physical_costs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    json_write(incident / "physical_cost_summary.json", physical)
    json_write(incident / "failed_endpoint_cost_bounds.json", failures)
    matrix = []
    for key in plan["evaluation_order"]:
        u = units[key]
        matrix.append({"endpoint": key, "status": u["status"], "attempts_used": len(u.get("attempts", [])),
            "attempt_labels": ";".join(a["label"] for a in u.get("attempts", [])),
            "prediction_sha256": u.get("predictions_sha256"),
            "remaining_attempts_under_frozen_rule": 0 if u["status"] in ("first_success_fixed", "exhausted_failure") else 2-len(u.get("attempts", []))})
    with (incident / "evaluation_status_all37.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = {"recorded_utc": now, "status": "blocked_on_absent_registered_GPU",
        "training_completed_and_CPU_audited": 35, "evaluation_completed_and_integrity_checked": 22,
        "evaluation_status_counts": dict(collections.Counter(u["status"] for u in units.values())),
        "exhausted_evaluation_objects": exhausted, "eligible_remaining_evaluation_order": remaining,
        "max_successful_evaluations_without_retry_policy_change": 32,
        "registered_adapter_hashes_reverified": verified_weights,
        "frozen_source_models_tokenizer_S2_data_and_protocol_verified": True,
        "registered_GPU": rx[0], "current_day_watchdog_dump_names": current_dumps,
        "host_receipt_had_no_active_project_model_processes": True,
        "GPU_allocation_smoke_skipped_because_registered_device_absent": True,
        "no_fallback_to_integrated_GPU_or_backend_change": True,
        "observed_complete_trained_object_mean_evaluation_seconds": mean,
        "ten_eligible_remaining_objects_process_hours_at_mean": len(remaining)*mean/3600,
        "estimate_excludes_GPU_recovery_failures_and_offline_report": True,
        "whole_session_deadline_unix_unchanged": plan["session_limit"]["deadline_unix"],
        "whole_session_elapsed_wall_seconds": time.time()-plan["session_limit"]["deadline_unix"]+plan["session_limit"]["total_session_hard_seconds"],
        "no_complete_primary_accuracy_inference_from_missing_cells": True,
        "no_training_or_model_evaluation_called_by_this_audit": True,
        "visibility_disclosure": "S2 aggregate and a subset of seed47 F-j20261111 progress IDs/correct flags were displayed during operational log inspection; no response text, seed29 blind identities, or sealed crosswalk inspected. Frozen protocol/scorer/gold unchanged.",
        "code45_reference": "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-phantom",
        "hip_error_reference": "https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/hip__runtime__api_8h.html",
        "source_sha256": sha256_file(ROOT / "analysis/record_large_filter_gpu_hold.py")}
    json_write(incident / "summary.json", summary)
    report = f"""# GPU设备缺失后的实验状态（2026-11-13）

本家族尚未完成。35/35个训练终点已经保存并通过独立CPU审计；22/37个对象完成同一1024题新开发集的完整独立评估，22份预测的SHA256、行数、ID顺序及适配器身份已复核。此次CPU审计另行复核全部36份登记适配器（35终点和S2）以及冻结源码、模型/tokenizer、数据与协议，均未改变。不重训，不覆盖成功结果，不打开GSM8K正式test或SVAMP。

北京时间约09:08—09:11，Windows记录当日AMD/WATCHDOG转储，评估子进程先原生退出0xC0000409，随后新进程在适配器dtype转换处报`hipErrorInvalidImage`。09:22:42的只读PnP核对显示RX9070 XT为`Present=false, Problem=45`，当前可见显示设备仅核显及虚拟显示适配器。微软将45解释为设备不在场；这些证据不能单独确定是驱动、供电、PCIe还是硬件根因。WER在同一分钟也重新登记历史日期转储，不能把所有WER条目当作本次新故障。

控制器已停止，最后失败子进程的真实Windows退出时间已留存，未用诊断等待时间冒充GPU进程运行时间；项目GPU锁已按所有权归档释放。未修改驱动、系统设置、缓存、冻结后端或评估阈值。因登记显卡缺失，没有转到核显执行模型，也没有再用模型任务试错。

## 完整矩阵中的缺口

5个对象已用完a0/a1两次尝试：{', '.join(exhausted)}。所有失败日志、部分输出、退出码和成本仍保留，部分预测不计为完整评估，也不拼接重试结果。冻结协议每单元最多两次；本家族不会静默授予第三次尝试。

仍可执行10个对象：{', '.join(remaining)}。其中`F-j20261112-s29`的a0已失败，唯一a1仍可用；其余9个尚未运行。显卡恢复后，在现有重试规则下最多能得到32/37份成功评估。因此原定完整5×3主比较和种子29盲化审阅存在缺口，不能通过删除缺失格、挑剩余seed或改审阅seed来冒充完整预登记结果。

全部37格的状态和已用attempt见[评估状态表](../../../{(incident / 'evaluation_status_all37.csv').relative_to(ROOT).as_posix()})；各模型权重仍在原始[评估前锁表](../../../{(exp / 'all_training_endpoints_fixed.json').relative_to(ROOT).as_posix()})中。

## 时间与成本

已完成训练模型评估的实测均值为{mean/60:.1f}分钟/对象。GPU恢复并通过同设备检查后，10个仍合规对象按均值约需{len(remaining)*mean/3600:.1f}小时纯评估进程时间；此数不含恢复、失败和后续统计/审阅/报告，不能作为当前可靠完成承诺。原完整37对象的预计完成时间已失效。原会话硬截止保持不变，用户暂停和诊断等待仍计入会话墙钟。

截至此次审计，全部模型attempt物理进程时间为{physical['all_attempt_model_process_seconds_lower_bound']/3600:.3f}—{physical['all_attempt_model_process_seconds_upper_bound']/3600:.3f}小时；其中失败进程为{physical['failed_model_process_seconds_lower_bound']/60:.2f}—{physical['failed_model_process_seconds_upper_bound']/60:.2f}分钟。区间来自旧U-s47-a0重启中断的未知退出界，不是置信区间。最后评估失败的父进程验证开销未单独计时，端点路径成本仍给保守界，未知值不记零。

[全部attempt物理账本](../../../{(incident / 'all_attempt_physical_costs.csv').relative_to(ROOT).as_posix()})单独保留真实进程成本；[逐端点失败路径成本界](../../../{(incident / 'failed_endpoint_cost_bounds.json').relative_to(ROOT).as_posix()})与[结构化状态](../../../{(incident / 'summary.json').relative_to(ROOT).as_posix()})可离线核验。F完整探测扣额、R实验配平项、共享三次探测的物理开销仍保持原口径；本次未计算缺失完整评估对象的端到端主信号比率或准确率区间。

## 恢复边界与审阅透明性

当前外部阻塞是登记显卡不在场。下一步由用户安排系统重启或显卡恢复；系统操作超出“仅修改本项目”的授权范围，不自动执行。恢复后先核对同一RX9070 XT、驱动/运行时及冻结SHA，并做不接触新开发题的轻量运行检查，再仅执行上述仍有额度的10个对象。任何补充5个耗尽对象的第三次尝试都必须明确视为事后协议例外，不能抹掉原失败或改称原预登记完整执行。

运维排障中已显示S2聚合正确数及F-j20261111-s47部分进度行的题目ID/判对标记，没有读取回答正文或seed29盲化身份映射。记录这一可见性限制；冻结评分器、金标、权重、阈值与审阅seed不变。本报告不对筛选收益作准确率结论，也不把22份结果当作完整37对象实验。
"""
    report_path = out / "gpu_runtime_blocked_status.md"
    assert not report_path.exists()
    report_path.write_text(report, encoding="utf-8", newline="\n")
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前阻塞：训练35/35，完整评估22/37

`{family}`全部35个训练终点和评估前权重锁已复核；22份完整1024题评估的预测SHA、ID顺序与适配器身份通过CPU核验。2026-11-13约09:11发生连续HIP运行错误，09:22核对RX9070 XT为设备不在场（PnP45）。控制器已停止并保留失败证据，未改冻结配置或重训。5个评估对象已用完两次尝试，另有10个对象仍有额度；恢复设备后最多32/37成功，不能省略缺失格声称完整预登记比较。

[阻塞状态与恢复边界](reports/comparisons/{family}/gpu_runtime_blocked_status.md)、[冻结协议](experiments/{family}/protocol_frozen.json)、[35终点权重锁](experiments/{family}/all_training_endpoints_fixed.json)、[22份预测完整性核验](experiments/{family}/evaluation_integrity_through_22.json)、[完整37格状态和成本审计](experiments/{family}/gpu_hold_audit_2026-11-13/summary.json)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。原会话硬截止及重试上限不变，GSM8K test/SVAMP继续封存，远端未发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        text, count = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->", block,
            path.read_text(encoding="utf-8"), flags=re.S)
        assert count == 1
        path.write_text(text, encoding="utf-8", newline="\n")
    work_path = exp / "AGENT_WORK_STATE.md"
    prior = work_path.read_text(encoding="utf-8")
    override = f"""## AUTHORITATIVE OVERRIDE: GPU DIAGNOSTIC HOLD, {now}

No current model/controller process. User latest authorization remains CONTINUE, not user pause. All35 training successful/fixed;22/37 complete evaluations CPU-verified. Five exhausted objects MUST NOT be reset or granted a third attempt;10 objects remain eligible (one pending_retry,9 pending). RX9070XT PnP Present=false/Problem45; host watchdog events and native abort/HIP invalid image logs retained. Do not launch model work or fall back to integrated GPU. Do not reboot/change drivers/system without user authorization; task scope only project modifications. Root cause not confirmed. Original full37 ETA invalid. Read gpu_runtime_blocked_status.md and gpu_hold_audit_2026-11-13/summary.json. Frozen source/model/tokenizer/data/S2 and36adapter hashes verified. Never edit frozen scripts/src/configs/tests. No more training; no completed evaluation reruns; no extending original deadline. CPU incident audit only, no primary outcomes/statistics or sealed blind identities read. Historical RUNNING/PAUSED directives below are superseded.

Remaining eligible order: {', '.join(remaining)}. Existing resume wrapper requires historical10complete/paused and is NOT usable as-is. Prepare a new audited evaluation-only wrapper only AFTER device recovery/readiness, importing unchanged controller, respecting exhausted statuses and original limits. Do not claim complete planned5x3 inference, full37 report or plannedseed29 blind review while required cells missing. GPU recovery or retry-policy exception needs explicit separate handling.

Operational visibility disclosure: S2 aggregate and a subset of seed47 F-j20261111 progress IDs/correct flags displayed, no response texts/seed29 sealed identity crosswalk inspected. Preserve this disclosure.

"""
    work_path.write_text(override+prior, encoding="utf-8", newline="\n")
    repro_path = ROOT / "docs/REPRODUCING_FILTER_REPLICATION_20261110.md"
    repro = repro_path.read_text(encoding="utf-8")
    note = f"\n> 2026-11-13实际状态：训练35/35、完整评估22/37；RX9070XT设备不在场，当前诊断停止。5格重试已耗尽，10格仍有额度；禁止直接按下面历史阶段命令重跑。先读[阻塞报告](../reports/comparisons/{family}/gpu_runtime_blocked_status.md)和`AGENT_WORK_STATE.md`最新覆盖说明。\n"
    repro_path.write_text(repro.replace("\n", "\n"+note, 1), encoding="utf-8", newline="\n")
    manifest_paths = [report_path, work_path, repro_path, ROOT / "README.md", ROOT / "PROJECT_CONTEXT.md", ROOT / "HANDOFF.md"]
    manifest_paths += list(incident.glob("*"))
    json_write(incident / "artifact_hashes.json", {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in manifest_paths})
    print(json.dumps({"training": 35, "complete_evaluations_verified": 22, "exhausted": exhausted,
        "eligible_remaining": len(remaining), "eligible_process_hours_at_mean": len(remaining)*mean/3600,
        "all_attempt_physical_hours_bounds": [physical['all_attempt_model_process_seconds_lower_bound']/3600,
            physical['all_attempt_model_process_seconds_upper_bound']/3600], "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
