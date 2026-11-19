"""Replace only this family's generated navigation block after complete verified results."""
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
    report = read(exp / "report_written.json")
    assert report["blinded_agent_review_completed"]
    assert report["sha256"] == sha256_file(out / "findings.md")
    s = read(out / "summary.json")
    assert s["successful_training"] == 35
    assert report["all35_training37_evaluation"] == (s["successful_evaluation"] == 37)
    missing = s["missing_evaluation_objects"]
    if missing:
        fu = s["available_pair_comparisons"]["F-U"]["available"]
        fr = s["available_pair_comparisons"]["F-R"]["available"]
        lo, hi = fu["conditional_percentile95_pp"]
        signal = s["signal_sensitivity"]["complete_seed_block_end_to_end"]
        signal_lift = signal["F"]["effective_completions_per_second"]/signal["U"]["effective_completions_per_second"]-1
        outcome = (f"F−U 在 {fu['pairs']}/15 个可观测配对中平均 {fu['mean_pp']:+.3f} pp，条件 95% 区间为 [{lo:+.3f}, {hi:+.3f}] pp。"
            f"F−R 在 {fr['pairs']}/15 个可观测配对中平均 {fr['mean_pp']:+.3f} pp。"
            f"端到端记录齐全的训练种子整块中，F 的有效 completion/秒相对 U 变化 {signal_lift:+.2%}。"
            f"{len(missing)} 个对象耗尽重试。原定完整矩阵准确率和端到端工程阈值无法判定。"
            "报告含 35 个训练终点的信号证据、完整缺失矩阵、可用配对区间和盲化 Agent 审阅。")
    else:
        ci = s["conditional_intervals"]["F-U"]["percentile95_pp"]
        signal_gain = s["F_vs_U_effective_signal_relative_improvement"]
        gain = f"{signal_gain:+.2%}" if signal_gain is not None else "未定义（U有效信号为0）"
        decision = "达到" if s["development_evidence_threshold_met"] else "未达到"
        outcome = (f"F−U 平均 {s['F_minus_U_mean_pp']:+.3f} pp，条件 95% 区间 [{ci[0]:+.3f}, {ci[1]:+.3f}] pp，"
            f"F−R 平均 {s['F_minus_R_mean_pp']:+.3f} pp。计费端到端有效信号率相对 U 变化 {gain}。{decision}前瞻开发阈值。")
    marker = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 固定预筛与随机子集复验已收尾

`{family}` 已复核 35/35 个训练终点及 {s['successful_evaluation']}/37 个对象在同一新开发 {s['n']} 题上的完整评估。共同池 256 题，三次固定 S2/K4 实际 F 集合大小 70/62/70。U 共 5 个独立终点，在三个探测块复用。全部训练权重在新开发集开启前锁定。

{outcome}

全部登记对象已到终态，待运行队列为空。新开发集来自官方 train 的后置隔离子集。项目记录未显示这些题被曝光，公开模型预训练曝光未知。GSM8K 正式 test 和 SVAMP 保持封存。工件、源码、原始预测和权重保存在本地待审阅。

[中文报告](reports/comparisons/{family}/findings.md) · [全部终点及成本](reports/comparisons/{family}/endpoints.csv) · [配对差与区间](reports/comparisons/{family}/summary.json) · [协议](experiments/{family}/protocol_frozen.json) · [完整工件 SHA256](experiments/{family}/artifact_manifest.json) · [复现](docs/REPRODUCING_FILTER_REPLICATION_20261110.md) · [文档导航](docs/INDEX.md)
<!-- LARGE_FILTER_CURRENT_END -->"""
    paths = []
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        current = path.read_text(encoding="utf-8")
        updated, n = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->", lambda _: marker, current, flags=re.S)
        assert n == 1
        path.write_text(updated, encoding="utf-8", newline="\n")
        paths.append(path)
    index = ROOT / "docs/INDEX.md"
    index_entry = f"- 固定预筛与随机子集复验的 {s['successful_evaluation']}/37 份完整评估已收尾。[中文报告](../reports/comparisons/{family}/findings.md) · [复现](REPRODUCING_FILTER_REPLICATION_20261110.md) · [冻结协议](../experiments/{family}/protocol_frozen.json) · [成本](../reports/comparisons/{family}/endpoints.csv)"
    if index.exists():
        current_index = index.read_text(encoding="utf-8")
        if index_entry not in current_index:
            header, tail = current_index.split("\n", 1)
            index.write_text(header+"\n\n"+index_entry+"\n"+tail, encoding="utf-8", newline="\n")
    else:
        index.write_text(f"""# 文档导航

{index_entry}
- [64 题固定预筛报告](../reports/comparisons/desktopfilter_20261110/findings.md) · [复现](REPRODUCING_FILTER_20261110.md)
- [SFT 排查和缓存动态采样报告](../reports/comparisons/desktoprepair_20261106_152712/findings.md) · [复现](REPRODUCING_REPAIR_20261106.md)
- [台式机 BF16 开发实验](../reports/comparisons/desktoprocm10pilot_20261105/findings.md) · [环境实测](DESKTOP_SETUP_20261105.md)
- [通用实验协议](PROTOCOL.md) · [数据与模型来源](SOURCES.md) · [相关工作](RELATED_WORK.md) · [学习指南](LEARNING_GUIDE.md) · [第三方许可](THIRD_PARTY.md)
- [交接](../HANDOFF.md) · [项目背景](../PROJECT_CONTEXT.md)

各家族以各自的冻结文件和原始工件为准。正式 GSM8K test 和 SVAMP 继续封存。本家族的工件保存在本地。
""", encoding="utf-8", newline="\n")
    paths.append(index)
    reproduction = ROOT / "docs/REPRODUCING_FILTER_REPLICATION_20261110.md"
    text = reproduction.read_text(encoding="utf-8")
    exp_link = f"../experiments/{family}"
    beijing = dt.timezone(dt.timedelta(hours=8))
    terminal_stamp = dt.datetime.fromisoformat(read(exp / "all_model_tasks_complete.json")["completed_utc"]).astimezone(beijing)
    latest_status = (f"> 截至北京时间 {terminal_stamp:%Y-%m-%d %H:%M:%S}，本家族 35 个训练终点已锁定，"
        f"{s['successful_evaluation']}/37 份完整评估已保存，{len(missing)} 个对象耗尽重试。"
        f"下面保留已执行命令以便核查。 [终态]({exp_link}/all_model_tasks_complete.json)"
        f" · [中文报告](../reports/comparisons/{family}/findings.md)")
    text, n = re.subn(r"\A(# [^\n]+\n\n)(?:> [^\n]*\n\n)+", lambda m: m[1]+latest_status+"\n\n", text, count=1)
    assert n == 1
    text = text.replace("报告、原始预测、逐终点CSV、盲化审阅、图表和离线复算将在全部模型任务结束后填入同一家族。", "本家族的实际结果和离线复算见下文。")
    text = text.replace("& $py -m pytest -q\n", "& $py -m pytest -q tests analysis/test_large_filter_missing_statistics.py\n")
    text = text.partition("\n## 本次实际完成与离线复算\n")[0].rstrip()
    text += f"""

## 本次实际完成与离线复算

原运行、写盘边界和总成本分别见 `all_model_tasks_complete.json` 与 `session_final.json`。在当前证据目录运行只读核验，校验最终清单登记的每个文件。

```powershell
& $py analysis/audit_large_filter_closeout.py --family $family --verify-only
```

在保留同一相对目录布局的证据副本内，以下命令使用实际运行预测及既有盲化注释重新生成派生统计、图表和中文报告。命令会写入副本中的派生工件。

```powershell
& $py analysis/analyze_large_filter.py --family $family
& $py analysis/large_filter_blind_units.py --family $family --mode materialize
& $py analysis/finalize_large_filter_reviews.py --family $family
& $py analysis/render_large_filter.py --family $family
& $py analysis/write_large_filter_report.py --family $family
& $py analysis/audit_large_filter_closeout.py --family $family
```

盲化 Agent 注释在身份解封前固定。`artifact_manifest.json` 为最终本地文件 SHA256 清单，覆盖模型和 tokenizer、共同 S2、35 个真实 GRPO 终点、Adam/RNG 诊断、配置源码、数据以及原始运行证据。正式保留集继续封存。
"""
    reproduction.write_text(text, encoding="utf-8", newline="\n")
    paths.append(reproduction)
    json_write(exp / "documentation_updated.json", {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}, "remote_published": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
