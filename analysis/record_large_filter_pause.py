"""Record an observed, audited checkpoint-boundary pause without claiming completion."""
import argparse
import datetime as dt
import re
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    paused = read(exp / "pause_after_endpoint_completed.json")
    assert paused["current_endpoint_status"] == "first_success_fixed"
    assert paused["no_next_unit_started"] is True and paused["completed_training"] == 21
    assert not (exp / "fresh_development_opened.json").exists()
    units = read(exp / "training_units.json")
    plan = read(exp / "protocol_frozen.json")
    completed = [u["key"] for u in plan["order"] if units[u["key"]]["status"] == "first_success_fixed"]
    assert len(completed) == 21
    pending = [u["key"] for u in plan["order"] if u["key"] not in completed]
    rows = []
    for key in completed:
        audit = read(exp / "completed_training_audits" / (key+".json"))
        assert audit["status"] == "independent_CPU_audit_passed"
        assert audit["checkpoint_sha256"] == units[key]["checkpoint_sha256"]
        d = audit["diagnostics"]
        rows.append(f"|{key}|{d['optimizer_steps']}|{d['effective_completions']}|{d['nonzero_gradient_steps']}|{d['adapter_delta_l2_independent']:.6f}|")
    interruption = read(exp / "interruption_cost_audit.json")["interrupted_processes"][0]
    deadline = dt.datetime.fromtimestamp(plan["session_limit"]["deadline_unix"], dt.timezone(dt.timedelta(hours=8))).isoformat()
    text = f"""# 用户请求的保存后暂停状态

家族`{family}`已在U-s47-a1完整保存并通过协议核验后暂停；实际完成21/35训练终点（训练seed13、29、47的完整区块），21份独立CPU审计通过。后续14单元尚未启动，新开发评估0/37，GSM8K test/SVAMP继续封存。这里没有新开发准确率结论，完整实验和最终中文发现报告仍待继续。

用户选择等待当前终点保存后暂停。外部观察器只暂停控制器，让当前GPU子进程按原B完成；Windows进程句柄记录真实退出时间/退出码，核验检查点后停止控制器，未启动下一单元。未暂停训练子进程、未改模型/数据/训练参数、未声称精确sampler/dataloader续训。已保存终点`{paused['checkpoint']}`，SHA256`{paused['checkpoint_sha256']}`。

## 已完成训练证据

|终点|优化步|有效completion|非零梯度步|相对S2适配器L2差|
|---|---|---|---|---|
{chr(10).join(rows)}

有效completion为实际优势非零且实际loss mask非空；这些是训练诊断，不是新开发准确率。35训练终点和A/S2全部固定后才能打开1024题fresh_development。

## 中断与成本

17:01附近系统重启中断U-s47-a0；该失败原始输出保留，不作为主终点。实际模型进程耗时只能界定在{interruption['process_seconds_lower_bound']:.3f}—{interruption['process_seconds_upper_bound']:.3f}秒，依据最后原始活动时间戳和首个随后OS启动事件；精确退出码未知。唯一允许的完整重试a1从相同S2/配置开始，保存后成为首次成功终点。恢复入口另有一次无label事件兼容错误，在启动GPU前发生；错误日志与源码保留，冻结训练源码未改。

算法扣额、R配平、共享物理探测及失败物理成本仍需在最终汇总中分列；被中断进程不能因为缺少完成收据而计为0。用户暂停和重启停机时间保留在实际会话墙钟中。原冻结B=3600、E=7500、O=300、C=11400秒和全会话硬截止保持不变：北京时间{deadline}，暂停不顺延。

## 继续入口

待用户明确继续后，从下一待执行单元`{pending[0]}`开始，同一S2和冻结配置；跳过21个已完成终点，保留a0/a1及所有成本证据。不能直接重启原main覆盖PID收据，不能恢复被终止的旧控制器；先读项目交接和本家族AGENT_WORK_STATE。剩余单元：{', '.join(pending)}。所有训练固定后再开启37对象评估，随后完成离线统计、盲化Agent审阅、8组实际图表、中文报告和最终工件清单。远端未发布。
"""
    (out / "paused_status.md").write_text(text, encoding="utf-8", newline="\n")
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前暂停：更大干净池固定预筛与随机子集复验

`{family}`已按用户要求在U-s47-a1保存核验后暂停。已完成21/35训练终点（seed13/29/47全区块），21份独立CPU证据审计通过；剩余14训练单元未启动，新开发评估0/37。256题共同池、1024题fresh_development、三次K4筛选M=70/62/70和B/E/O/C=3600/7500/300/11400秒均保持冻结。全部训练权重固定后才允许新开发评估，当前不能宣称新题收益。

U-s47-a0因系统重启中断，原始输出及成本上下界保留；同配置、同S2的唯一a1完整重试已成功。用户明确继续后从`{pending[0]}`开始，跳过21个成功终点。原全会话硬截止不顺延。当前[暂停记录](reports/comparisons/{family}/paused_status.md)、[暂停收据](experiments/{family}/pause_after_endpoint_completed.json)、[运行状态](experiments/{family}/runner_state.json)、[冻结协议](experiments/{family}/protocol_frozen.json)、[准备证据](reports/comparisons/{family}/preparation.md)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。GSM8K test/SVAMP继续封存，远端未发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    hashes = {}
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        original = path.read_text(encoding="utf-8")
        updated, n = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->", block, original, flags=re.S)
        assert n == 1
        path.write_text(updated, encoding="utf-8", newline="\n")
        hashes[name] = sha256_file(path)
    json_write(exp / "paused_documentation_updated.json", {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "paused_by_explicit_user_request_not_complete", "successful_training": 21, "fresh_evaluations": 0,
        "original_hard_deadline_unchanged": True, "files": hashes,
        "pause_report_sha256": sha256_file(out / "paused_status.md")})
    print({"status": "paused_state_documented", "training": 21, "evaluation": 0, "next": pending[0], "deadline_local": deadline})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
