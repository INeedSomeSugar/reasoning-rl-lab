"""Document a verified user pause after a complete fresh-development evaluation."""
import argparse
import datetime as dt
import re
import sys

from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read
from run_large_filter import verify_frozen


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    paused = read(exp / "pause_after_evaluation_completed.json")
    host = read(exp / "evaluation_pause_host_process_verification.json")
    assert paused["status"] == "paused_by_user_after_complete_evaluation" and paused["no_next_model_started"]
    assert host["active_project_model_or_controller_processes"] == []
    assert not (ROOT / "environment/large_filter_gpu.lock.json").exists()
    assert read(exp / "runner_state.json")["status"] == "paused_by_user"
    plan = read(exp / "protocol_frozen.json")
    verify_frozen(family, plan)
    lock_path = exp / "all_training_endpoints_fixed.json"
    lock = read(lock_path)
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    units = read(exp / "training_units.json")
    assert len(units) == 35 and all(u["status"] == "first_success_fixed" for u in units.values())
    for item in units.values():
        assert checkpoint_hash(ROOT / item["checkpoint"]) == item["checkpoint_sha256"]
    evaluations = read(exp / "evaluation_units.json")
    done = [k for k in plan["evaluation_order"] if evaluations[k]["status"] == "first_success_fixed"]
    pending = [k for k in plan["evaluation_order"] if evaluations[k]["status"] == "pending"]
    assert len(done) == paused["completed_evaluation"] and len(done)+len(pending) == 37
    hashes = {}
    rows = []
    for key in done:
        item = evaluations[key]
        directory = ROOT / "runs" / item["run_name"]
        path = directory / "fresh_development/predictions.jsonl"
        digest = sha256_file(path)
        assert digest == item["predictions_sha256"] and item["n"] == plan["development_n"]
        assert sum(bool(line.strip()) for line in path.open(encoding="utf-8")) == plan["development_n"]
        protocol = read(directory / "evaluation_protocol.json")
        assert protocol["adapter_sha256"] == lock["objects"][key]["checkpoint_sha256"]
        for relative in ("fresh_development/predictions.jsonl", "fresh_development/metrics.json",
                "evaluation_protocol.json", "resolved_evaluation_ids.json"):
            target = directory / relative
            hashes[target.relative_to(ROOT).as_posix()] = sha256_file(target)
        rows.append(f"|{key}|{item['n']}|{item['evaluation_process_seconds']:.3f}|`{digest}`|")
    local = dt.datetime.fromisoformat(paused["paused_utc"]).astimezone(dt.timezone(dt.timedelta(hours=8)))
    deadline = dt.datetime.fromtimestamp(plan["session_limit"]["deadline_unix"], dt.timezone(dt.timedelta(hours=8)))
    text = f"""# 新开发评估保存后暂停

家族`{family}`于北京时间{local.isoformat()}按用户要求暂停。35/35个训练终点全部完成并通过独立CPU审计；37份评估权重在新开发开启前已固定。当前完整评估{len(done)}/37，每份均为同一1024题；剩余{len(pending)}对象尚未运行。主统计、盲化Agent审阅、8组图表与最终中文发现报告待全部评估完成，当前不作主准确率结论。

本次最后保存`{paused['saved_evaluation_key']}`，退出码0，完整预测SHA256为`{paused['saved_predictions_sha256']}`。观察器在当前任务正常运行期间设置归属不同的下一任务启动锁，当前控制器按原逻辑记录真实进程成本和完成审计；完整首次成功状态持久化后才终止控制器并释放锁。当前子进程未被暂停或中断，下一模型任务未启动，未消耗额外模型重试。退出后已作主机进程核验，无本项目模型或控制器运行。

## 已完成的完整评估证据

|对象|题数|模型进程秒|预测SHA256|
|---|---|---|---|
{chr(10).join(rows)}

以上只登记进度、完整性和成本，不以部分结果选择种子或改训练。原S2、全部35个终点、基础模型/tokenizer、源码、数据和协议哈希重新核验通过，普通train权限和正式holdout权限不变。GSM8K test/SVAMP继续封存。

## 继续与剩余工作

仅在用户明确继续后，从下一对象`{pending[0]}`启动独立新进程评估，跳过已完成的{len(done)}个评估及全部35个训练终点。训练在本家族永久关闭，不恢复半份评估、不重跑完成对象、不使用旧的训练阶段恢复入口。后续需新建只恢复评估的安全包装器，导入未变的冻结控制器，并保留新PID、来源哈希和日志。

剩余对象：{', '.join(pending)}。

B/E/O/C仍为3600/7500/300/11400秒；原全会话硬截止为北京时间{deadline.isoformat()}，本次暂停不顺延。暂停、重启和准备时间进入实际会话墙钟，不能作为GPU活跃时间重复收费。所有失败及U47唯一完整重试证据继续保留；此前中断成本界不变。完成全部37评估后再统一复算统计、逐响应实际Agent盲审、图表核验、主要发现和文档导航。远端未发布。
"""
    report = out / "evaluation_paused_status.md"
    report.write_text(text, encoding="utf-8", newline="\n")
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前暂停：35个训练完成，评估{len(done)}/37

`{family}`已按用户要求在`{paused['saved_evaluation_key']}`完整保存核验后暂停；35/35训练终点和37份权重锁保持固定，{len(done)}/37个独立评估各完成全部1024题。下一模型任务未启动，主机核验无活动模型或控制器。剩余{len(pending)}个评估，用户明确继续后从`{pending[0]}`开始；训练永久关闭，不重跑已完成对象。

256题共同池、fresh_development1024题、M=70/62/70及B/E/O/C=3600/7500/300/11400秒保持冻结，原硬截止不顺延。全部评估结束后再统一统计、盲化Agent审阅、图表与中文主要发现；当前不作主准确率结论。[暂停状态](reports/comparisons/{family}/evaluation_paused_status.md)、[暂停收据](experiments/{family}/pause_after_evaluation_completed.json)、[全部权重锁](experiments/{family}/all_training_endpoints_fixed.json)、[冻结协议](experiments/{family}/protocol_frozen.json)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。GSM8K test/SVAMP继续封存，远端未发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        updated, count = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->",
            block, path.read_text(encoding="utf-8"), flags=re.S)
        assert count == 1
        path.write_text(updated, encoding="utf-8", newline="\n")
        hashes[name] = sha256_file(path)
    for relative in ("protocol_frozen.json", "frozen_source_manifest.json", "training_units.json", "evaluation_units.json",
            "all_training_endpoints_fixed.json", "fresh_development_opened.json", "pause_after_evaluation_completed.json",
            "evaluation_pause_host_process_verification.json"):
        path = exp / relative
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    hashes[report.relative_to(ROOT).as_posix()] = sha256_file(report)
    json_write(exp / "evaluation_pause_handoff_manifest.json", {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "paused_by_user_not_complete", "completed_training": 35, "completed_evaluation": len(done),
        "next_evaluation": pending[0], "remaining_evaluations": pending, "files": hashes,
        "all35_checkpoint_hashes_reverified": True, "source_models_data_protocol_unchanged": True,
        "original_deadline_unchanged": True, "all37_analysis_not_yet_run": True,
        "visibility_note": "During process-metadata inspection an existing S2 aggregate correct count was displayed. No per-question response or method-identity crosswalk was inspected; no frozen design or scoring changed."})
    print({"status": "evaluation_pause_documented", "training": 35, "evaluation": len(done), "next": pending[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
