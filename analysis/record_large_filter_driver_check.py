"""Document the recovered-device checks and the requested evaluation-boundary pause."""
import argparse
import datetime as dt
import re
import sys

from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, sha256_file

sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read
from run_large_filter import verify_frozen


def main(family, stage):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    lock = read(exp / "all_training_endpoints_fixed.json")
    assert len(lock["objects"]) == 37 and lock["all35_successful"]
    assert read(exp / "fresh_development_opened.json")["endpoint_lock_sha256"] == sha256_file(exp / "all_training_endpoints_fixed.json")
    for obj in lock["objects"].values():
        if obj["checkpoint"]:
            assert checkpoint_hash(ROOT / obj["checkpoint"]) == obj["checkpoint_sha256"]
    readiness_path = sorted(exp.glob("driver_recovery_readiness_*/completed.json"))[-1]
    readiness = read(readiness_path)
    assert readiness["status"] == "passed" and readiness["old64_parity"]["prediction_text_identical"] == 64
    assert readiness["old64_parity"]["original_predictions_sha256"] == readiness["old64_parity"]["recovery_predictions_sha256"]
    evaluations = read(exp / "evaluation_units.json")
    completed = [k for k in plan["evaluation_order"] if evaluations[k]["status"] == "first_success_fixed"]
    exhausted = [k for k in plan["evaluation_order"] if evaluations[k]["status"] == "exhausted_failure"]
    pending = [k for k in plan["evaluation_order"] if evaluations[k]["status"] in ("pending", "pending_retry")]
    assert len(exhausted) == 5
    state = read(exp / "runner_state.json")
    host_path = exp / "driver_recovery_paused_host_verification.json"
    pause_path = exp / "pause_after_driver_recovery_evaluation_completed.json"
    request_path = exp / "driver_recovery_evaluation_pause_armed.json"
    request = read(request_path)
    choice_path = exp / "driver_recovery_user_pause_choice_20261113.json"
    choice = read(choice_path)
    assert choice["selected_answer"] == "等当前评估保存后暂停（推荐）"
    assert not choice["training_authorized"] and not choice["further_evaluations_authorized"]
    if stage == "paused":
        pause, host = read(pause_path), read(host_path)
        assert pause["status"] == "paused_by_user_after_complete_evaluation" and pause["no_next_model_started"]
        assert pause["saved_evaluation_key"] == "F-j20261112-s29" and pause["completed_evaluation"] == len(completed) == 23
        assert host["active_project_model_or_controller_processes"] == [] and not host["GPU_lock_present"]
        assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
        assert state["status"] == "paused_by_user" and len(pending) == 9
        state.update(runner_pid=None, status="paused_by_user", successful_training=35,
            successful_evaluation=23, original_deadline_unix_unchanged=plan["session_limit"]["deadline_unix"],
            latest_pause_receipt=pause_path.relative_to(ROOT).as_posix(), updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        json_write(exp / "runner_state.json", state)
        phase_text = "已在F-j20261112-s29-a1的1024题完整评估保存、原控制器计费及核验后暂停。主机核验无项目模型、控制器或暂停观察器，GPU锁已释放。当前35/35训练、23/37完整评估、5格重试耗尽、9格待运行。"
        heading = "当前暂停：训练35/35，完整评估23/37"
        command_text = "用户明确继续前不启动任何GPU任务。训练在本家族永久关闭；未来仅可按冻结顺序运行9个有额度对象，跳过23份成功评估及5格耗尽对象。现有恢复包装器预检查固定为22+10，不能原样用于下一次恢复，需重新核验实际暂停状态。"
    else:
        assert len(completed) == 22 and state["status"] == "pausing_after_current_evaluation"
        assert read(ROOT / "environment/large_filter_gpu.lock.json")["runner_pid"] == request["observer_pid"]
        phase_text = "用户在恢复运行器启动后要求先不启动训练，并明确选择当前独立评估保存后暂停。当前只有F-j20261112-s29-a1评估在运行，无训练；下一GPU任务已由不同所有者锁拦住。当前35/35训练、22/37完整评估、5格重试耗尽，暂停尚待当前评估保存后核验。"
        heading = "正在保存后暂停：当前仅独立评估"
        command_text = "不得启动后续GPU任务；当前项保存后由观察器终止控制器。用户明确继续前，不启动任何训练或后续评估。"
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    directory = exp / ("driver_check_documentation_"+stage+"_"+stamp)
    directory.mkdir(exist_ok=False)
    report_path = out / "driver_recovery_check_20261113.md"
    doc_paths = [ROOT / "README.md", ROOT / "PROJECT_CONTEXT.md", ROOT / "HANDOFF.md", exp / "AGENT_WORK_STATE.md",
        ROOT / "docs/REPRODUCING_FILTER_REPLICATION_20261110.md"]
    for index, path in enumerate(doc_paths+[report_path]):
        if path.exists():
            (directory / (str(index)+"_before_"+path.name)).write_bytes(path.read_bytes())
    text = f"""# 驱动重装检查及保存后暂停（2026-11-13）

{phase_text}

## 实际恢复检查

- 用户清理并重新安装驱动；11:05重启后，同一RX9070XT设备实例恢复`Present=true, Problem=0, Status=OK`。
- 独显驱动从`32.0.32015.12012`改为`32.0.32015.16001`，对应[AMD Adrenalin26.11.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-11-1.html)。这是用户实施的环境恢复，后续评估按新驱动分段登记；不能写成所有评估处在完全相同驱动版本下。
- 项目Torch仍为`2.13.0+rocm10.0.0`、HIP`7.15.26333`，只暴露同一RX9070XT；原非量化BF16/SDPA、模型、S2、tokenizer、评分器和数据保持原SHA。
- 小张量BF16/SDPA前后向、有限非零梯度及BF16→FP32转换通过，用时{readiness['process_receipts'][0]['process_wall_seconds']:.3f}秒；没有优化器更新或保存模型。
- 原已使用开发64题以固定S2和原greedy配置完整重放，用时{readiness['process_receipts'][1]['process_wall_seconds']:.3f}秒。64/64回答原文及判对标记相同，预测文件SHA256完全一致：`{readiness['old64_parity']['original_predictions_sha256']}`。旧64题52/64；该重复检查不构成新的泛化证据，也不能证明整个新开发集在两个驱动下逐位一致。
- 全部35终点和S2适配器、基础模型/tokenizer、冻结源码/配置、数据与协议重新核验通过。恢复检查不使用fresh_development或GSM8K test/SVAMP。

重装后的只读系统事件快照仅见Display4107信息事件（显示模式枚举），截至该快照未见新的Display/WHEA错误。此前驱动故障前后存在看门狗、原生0xC0000409退出、HIP invalid image，以及显卡设备不在场；这些证据尚不能区分具体驱动缺陷、系统兼容、PCIe/供电或硬件原因。通过一次短检查不能证明长时间运行稳定。

## 运行与暂停的实际边界

11:32:50按此前仍生效的继续指令启动评估恢复运行器；11:33:05开始F-j20261112-s29-a1。用户随后要求检查完先不启动训练；未启动任何训练，并立即拦住后续GPU调度。用户明确选择“等当前评估保存后暂停”。11:34:32观察器设置下一任务启动锁，当前评估保留完整原进程成本及原评分，不截停、不拼接部分输出。

{command_text}

仍保留的5个耗尽对象：{', '.join(exhausted)}。冻结每对象最多两次，未增加第三次尝试；这些缺失格继续显式保留，不能声称完整37对象或完整预登记5×3比较。新的驱动分段、失败及准备成本也要计入最终限制说明。原B/E/O/C与硬截止不变。

当前阶段：`{stage}`；完整评估{len(completed)}/37。主统计和原定seed29盲化审阅尚未完成，不用部分结果选模型或换seed。GSM8K test/SVAMP继续封存，远端未发布。

## 证据与复现

- [恢复检查收据](../../../{readiness_path.relative_to(ROOT).as_posix()})及其中两项GPU检查的原始日志和物理成本。
- [暂停请求](../../../{request_path.relative_to(ROOT).as_posix()})；完成后使用`pause_after_driver_recovery_evaluation_completed.json`，历史Nov12暂停收据不覆盖。
- [用户明确选择保存后暂停](../../../{choice_path.relative_to(ROOT).as_posix()})；[仅CPU调度保护回归检查](../../../{(exp / 'gpu_recovery_failure_guard_regression_20261113.json').relative_to(ROOT).as_posix()})通过，未运行模型或训练。
- [全部评估状态](../../../{(exp / 'evaluation_units.json').relative_to(ROOT).as_posix()})、[评估前统一锁定权重](../../../{(exp / 'all_training_endpoints_fixed.json').relative_to(ROOT).as_posix()})、[先前故障审计](gpu_runtime_blocked_status.md)。
- `analysis/check_large_filter_driver_recovery.py`仅用于此次已执行检查；`analysis/resume_large_filter_after_gpu_recovery.py`导入未改的冻结控制器，增加运行前设备核验及新失败后停止的外围调度保护；`analysis/pause_large_filter_after_driver_check.py`仅用于当前保存边界暂停。所有源码快照和SHA保存在各收据目录，不将这些历史命令作为覆盖重跑入口。

运维可见性限制仍保留：此前显示S2聚合正确数及seed47部分ID/判对进度行；未读取seed29回答正文或盲化身份映射，冻结金标/评分器/阈值不改。记录时间{now}。
"""
    report_path.write_text(text, encoding="utf-8", newline="\n")
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## {heading}

`{family}`：{phase_text}

重装后同一RX9070XT驱动为32.0.32015.16001，BF16/SDPA轻量检查通过，旧64题S2输出与旧驱动逐字节一致。新驱动环境分段及5个耗尽评估缺口均保留；{command_text}

[驱动检查与暂停状态](reports/comparisons/{family}/driver_recovery_check_20261113.md)、[实时状态](experiments/{family}/runner_state.json)、[冻结协议](experiments/{family}/protocol_frozen.json)、[权重锁](experiments/{family}/all_training_endpoints_fixed.json)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。GSM8K test/SVAMP封存，原硬截止不顺延，远端未发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    for path in doc_paths[:3]:
        updated, count = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->", block,
            path.read_text(encoding="utf-8"), flags=re.S)
        assert count == 1
        path.write_text(updated, encoding="utf-8", newline="\n")
    work_path = doc_paths[3]
    override = f"""## AUTHORITATIVE OVERRIDE: USER REQUESTS PAUSE AFTER DRIVER CHECK, {now}

User explicitly selected wait for current complete evaluation then pause. Stage={stage}; completed training35, evaluations{len(completed)}, exhausted5, pending{len(pending)}. Current phase/status must be read live. No training permitted, no further GPU tasks until explicit continue. {phase_text} Read reports/comparisons/FAMILY/driver_recovery_check_20261113.md. RegisteredRX9070XT healthy after user reinstall, driver32.0.32015.16001 (previous.12012); sameTorch/HIP/BF16/SDPA, old64 prediction SHA bit-identical. Frozen source/models/data/S2/36adapters verified. New GPU failure circuit breaker exists in analysis recovery overlay; do not edit frozen scripts/src/configs/tests. Preserve5exhausted, do not grant third attempts. Original harddeadline unchanged. All37 report/statistics/blind review incomplete; no primary accuracy claims. Source snapshots/receipt directories preserved. Historical CONTINUE/RUNNING/diagnostic-GPU-absent directives below superseded by latest user pause.

""".replace("FAMILY", family)
    work_path.write_text(override+work_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    repro_path = doc_paths[4]
    repro = repro_path.read_text(encoding="utf-8")
    note = f"\n> 2026-11-13最新覆盖：用户要求当前评估保存后暂停，当前阶段`{stage}`。显卡已重装恢复；先读[驱动检查与暂停记录](../reports/comparisons/{family}/driver_recovery_check_20261113.md)。用户明确继续前不启动GPU任务；历史命令不得直接重跑。\n"
    repro_path.write_text(repro.replace("\n", "\n"+note, 1), encoding="utf-8", newline="\n")
    paths = doc_paths+[report_path, request_path, choice_path, readiness_path,
        exp / "gpu_recovery_failure_guard_regression_20261113.json", exp / "runner_state.json", exp / "evaluation_units.json"]
    paths += [ROOT / "analysis" / name for name in ("check_large_filter_driver_recovery.py",
        "resume_large_filter_after_gpu_recovery.py", "pause_large_filter_after_driver_check.py",
        "check_gpu_recovery_failure_guard.py", "large_filter_status.py", "record_large_filter_driver_check.py",
        "audit_completed_large_filter_evaluation_integrity.py")]
    if stage == "paused":
        paths += [pause_path, host_path, exp / "evaluation_integrity_through_23.json"]
    write_new = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    json_write(directory / "manifest.json", {"recorded_utc": now, "stage": stage, "training": 35,
        "complete_evaluations": len(completed), "exhausted": exhausted, "pending_order": pending,
        "saved_complete_prediction_hashes": {k: evaluations[k]["predictions_sha256"] for k in completed},
        "files": write_new, "source_sha256": sha256_file(ROOT / "analysis/record_large_filter_driver_check.py"),
        "no_complete_main_statistical_claim": True, "user_pause_authorization_active": True})
    print({"stage": stage, "complete_evaluations": len(completed), "exhausted": len(exhausted), "pending": len(pending), "report": str(report_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", choices=["armed", "paused"], required=True)
    args = parser.parse_args()
    main(args.family, args.stage)
