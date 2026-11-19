"""Human-readable frozen design and current-state navigation, with no model calls."""
import argparse
import csv
import datetime as dt
import re
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, START, START_HASH


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    plan = read(exp / "protocol_frozen.json")
    budget = plan["budget"]
    rows = []
    for index, unit in enumerate(plan["order"], 1):
        j = unit["probe_seed"]
        debit = budget["P_seconds"][str(j)] if j is not None else 0
        M = plan["subsets"][str(j)]["M"] if j is not None else 256
        rows.append({"run_order": index, **unit, "M": M, "B_seconds": budget["B_seconds"],
            "probe_or_matching_debit_seconds": debit, "own_training_allocation_seconds": budget["B_seconds"]-debit,
            "E_seconds": budget["E_seconds"], "O_seconds": budget["O_seconds"], "C_seconds": budget["C_seconds"],
            "config": plan["configs"][unit["key"]]["path"], "config_sha256": plan["configs"][unit["key"]]["sha256"],
            "initial_checkpoint": START, "initial_checkpoint_sha256": START_HASH,
            "development_n": plan["development_n"], "R_debit_is_artificial_matching": unit["arm"] == "R"})
    path = out / "registered_matrix.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Verify the identical model-generation files actually snapshotted by all three probes.
    shared = read(exp / "probe_inputs_frozen.json")["files"]
    probe_versions = {}
    for j, item in read(exp / "all_probes_complete.json")["probes"].items():
        snapshot = ROOT / "runs" / item["run"] / "source_snapshot"
        verified = {}
        for rel, digest in shared.items():
            assert sha256_file(snapshot / rel) == digest
            verified[rel] = digest
        probe_versions[j] = verified
    json_write(exp / "probe_shared_execution_inputs_verified.json", {"all_three_generation_inputs_identical": True,
        "verified": probe_versions, "main_pool_resolution_cost_metadata_may_be_extended_after_probes": True,
        "raw_success_filter_rule_unchanged": True})
    hours = plan["session_limit"]["likely_remaining_model_seconds_timing_estimate"]/3600
    hard = plan["session_limit"]["total_session_hard_seconds"]/3600
    M = "/".join(str(plan["subsets"][j]["M"]) for j in sorted(plan["subsets"]))
    marker = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前执行：更大干净池固定预筛与随机子集复验

`{family}` 已完成四份历史归档SHA256/全历史曝光/近重复审计、Agent质量审阅、新256题池与1024题fresh_development隔离、旧开发64题计时和前后向/重载。三次独立S2/K4探测实际M={M}，全部原始输出保留。主协议已在35个主训练终点之前冻结：B={budget['B_seconds']}、E={budget['E_seconds']}、O={budget['O_seconds']}、C={budget['C_seconds']}秒；F/R每终点扣对应完整P_j。U仅5个独立终点，F15/R15；全部终点固定后才开启37对象独立新进程开发评估。

实测推算剩余模型任务约{hours:.2f}小时；冻结全会话硬上限{hard:.2f}小时包含每失败单元一次完整重试及外围预留，不授权额外种子、训练或调参。当前[运行状态](experiments/{family}/runner_state.json)、[冻结协议](experiments/{family}/protocol_frozen.json)、[完整矩阵](reports/comparisons/{family}/registered_matrix.csv)、[准备证据](reports/comparisons/{family}/preparation.md)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。主训练／新开发结果尚未完成，不能据此宣称收益。GSM8K test/SVAMP继续封存；本轮未远端发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        p = ROOT / name
        text = p.read_text(encoding="utf-8")
        updated, count = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->", lambda _: marker,
            text, flags=re.S)
        assert count == 1
        p.write_text(updated, encoding="utf-8", newline="\n")
    frozen = f"""# 主实验冻结摘要

家族`{family}`；冻结时间UTC：{plan['frozen_utc']}。完整规范为本目录`protocol_frozen.json`，本文件是便于审阅的派生摘要。

共同候选池256题，新开发集1024题；真实S2为`{START}`，完整SHA256`{START_HASH}`。Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e，非量化BF16/SDPA，LoRA r8/alpha16/q_proj+v_proj。LR5e-6、G4、梯度累积4、grpo/group/beta0/epsilon0.2；token重要性比率、单轮更新、截断mask、Adam/scheduler及零梯度步行为沿用原实现。训练/探测temperature0.8/top_p0.95/top_k20，prompt512/newtoken512，评估greedy512，math_numeric_v2/numeric_final_v2/enable_thinking=False。

训练seed13/29/47/71/97；探测seed20261111/12/13；随机子集seed3101/3102/3103。U5、F15、R15，共35训练终点；A/S2锚点加35终点共37独立加载评估。每R_j从全共同池无放回抽真实M_j题，允许重叠；三方法各自池内均匀有放回训练。筛选仅按固定S2/K4原始成功次数1/2/3，实际M_j={M}，不凑数、不改seed/K。没有动态刷新或奖励塑形。

P_j完整进程耗时：{budget['P_seconds']}。B={budget['B_seconds']}秒（只按3maxP取整、下限3600），E={budget['E_seconds']}秒（前瞻修订为旧64题时间外推2倍加300取整），O={budget['O_seconds']}秒，C={budget['C_seconds']}秒。U可用B，F/R各用B−P_j。B从控制器全模型／源码缓存核验前计时，包括子进程启动、加载、rollout、优化和保存；自然原子边界及保存超额如实计入C。评估预留不回填训练。R的P_j是实验配平项，随机抽样实际物理成本另列，不能推断未运行的免扣额长训R效果。

全部35单元尝试结束并锁定成功权重之后才开新开发评估。失败每单元最多一次完整同配置同S2新attempt重试，采用首次成功，全部失败工件和成本保留。全会话硬上限{hard:.3f}小时，其中含全矩阵主要任务、各单元一次完整失败重试、20%外围余量及6小时分析，不按新开发成绩扩充。预期实际剩余模型任务约{hours:.3f}小时，硬上限不是预计用足的运行时间。

主要F−U；机制F−R；同时报告F−S2/F−A。联合近重复簇bootstrap20k次给固定训练/探测种子的条件95%区间；另有保持U复用的训练seed块/选择块/题目簇交叉bootstrap作探索性敏感性。总体有效信号+20%且三个探测块均正作为工程复现阈值；F−U≥+1pp且条件下界>0作为第二独立池复验的开发证据阈值，按未舍入指标判断，不保证达到。

新开发为官方train的后置隔离子集，限于项目记录未曝光；公开模型预训练污染未知。正式test/SVAMP继续封存，开发结果不作正式测试或广泛泛化结论。全部原始数据、疑义理由、集合ID、哈希、完整矩阵、失败、成本和负向结果均保留，远端发布等待后续明确指令。
"""
    (exp / "protocol_frozen.md").write_text(frozen, encoding="utf-8", newline="\n")
    print({"B": budget["B_seconds"], "E": budget["E_seconds"], "C": budget["C_seconds"], "M": M,
        "estimated_remaining_model_hours": hours, "session_hard_hours": hard})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
