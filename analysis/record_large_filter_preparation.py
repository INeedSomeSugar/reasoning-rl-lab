"""Write the completed preparation evidence and keep current-state navigation honest."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file


def main(family):
    exp = ROOT / "experiments" / family
    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    audit, data, cal, preflight = [read(exp / n) for n in ("history_audit.json", "data_frozen.json", "calibration_complete.json", "preflight.json")]
    out = ROOT / "reports/comparisons" / family
    out.mkdir(parents=True, exist_ok=True)
    text = f"""# 更大干净候选池复验：准备与定标已完成

家族：`{family}`。2026-11-10登记，跨北京时间2026-11-11继续执行。当前阶段为三次独立完整K4探测；尚无本轮新开发结果。本文件记录已完成证据，最终发现以同目录`findings.md`及模型任务完成收据为准。

## 数据资格已建立

核验四份项目内已发布源码／证据ZIP的完整SHA256，比对归档与当前checkout的原始运行工件。未发现缺失的归档运行证据；另存10份不同历史版本，没有覆盖现有工件。扫描不仅覆盖`runs`，还包含历史配置、实验登记、训练/探测/评估输出、报告、脚本和日志。曝光账本覆盖名义SFT128、实际监督记录、GRPO池与实际选择、定标、探测、开发评估及所有已保存预测；对仅有历史提及、不能证明模型接触的ID也保守排除。

- 历史被提及或配置的ID：{audit['historically_referenced_or_configured_ids']}；原validation512全部排除，不假设其剩余题目仍干净。
- 连同已知疑义题和相关簇，共排除{audit['blocked_ids_including_cluster_closure']}题；剩余{audit['eligible_rows']}题／{audit['eligible_clusters']}簇。
- 沿用原word(1,2) TF-IDF cosine≥0.90审计边，并增加不读取正式holdout题干的official-train单独重拟合检查：{audit['train_only_near_pairs']}对；归一化完全重复0对。最终簇取两套边及完全重复的并集，所有排除和开发抽样按完整簇。
- 按种子20261114的固定候选顺序，Agent读题干和原参考解并复核数量关系。前{data['reviewed_candidates']}题中排除{len(data['review_exclusions'])}题，保留共同256题；另看到的5题补位样例也留档。所有320题审阅／备用包均排除在新开发集之外。
- 按种子20261115，从其余合格原train簇冻结新开发{data['development_n']}题，具有独立manifest和`fresh_development`加载入口。原manifest与旧split未覆盖。普通train/select/develop加载器会剔除已登记新开发题；该新入口拒绝训练或普通holdout权限，评估还要求全部35个训练单元终止并锁定全部成功权重。

“未曝光”仅指可恢复项目记录中的模型训练／探测／评估及保守历史排除，不能证明公开Qwen预训练未见。题目审阅身份为**Agent审阅**，`human_reviewed=false`。没有改原文、金标或冻结评分器。

## 旧开发题定标与工程检查

共同真实S2完整checkpoint SHA256：`{preflight['fixed_s2_sha256']}`，11份基础模型/tokenizer文件逐项核对。单RX9070XT、非量化BF16/SDPA加载、有限非零梯度、冻结骨干梯度0、Adam零梯度后仍可能更新权重以及保存重载均已实测。诊断权重仅作检查，从未作为新训练起点；重载logits最大绝对差{preflight['reload_logits_max_absolute_difference']}。

已使用的开发64题独立S2进程耗时{cal['old64_evaluation_process_seconds']:.6f}秒。历史训练16题上的8步GRPO定标完成，未对新开发集运行模型。工程检查74项通过（后续实现如有改动，将重新核验并保留日志）。

三次256题完整K4探测的实测进程时间P_j完成后，以`B=max(3600,300*ceil(3*max(P_j)/300))`冻结共同额度。F每个终点扣完整P_j，R也扣P_j配平可训练时间，同时单列R该项属于实验配平，不将其写成随机方法本身的必需物理成本。E使用旧64题定标推算N并加35%及120秒后向上取整300秒；O固定300秒，C=B+E+O。全会话上限另包含全矩阵、每单元一次完整失败重试、20%外围余量及6小时分析预留，不沿用上一轮4小时上限。

## 已保留的准备异常

最初一次隐藏后台启动返回PID5484，但未观察到运行器入口收据或模型日志，随后已确认进程不存在；两份空stdout/stderr文件保留。未确认其根因，未记为成功模型任务。随后改用持久受控进程，逐子进程记录PID、命令、心跳、退出码和成本。一次交互查看候选时出现GBK输出异常，已用项目UTF-8环境继续读取；不影响原始UTF-8数据或金标。CIM只读进程查询不可用，改用项目Python/psutil与进程证据，不修改系统设置。

## 可审阅证据

- [前瞻登记](../../../experiments/{family}/preregistration.json)
- [历史审计](../../../experiments/{family}/history_audit.json)、[逐ID曝光账本](../../../experiments/{family}/history_exposure_ledger.jsonl)、[来源版本](../../../experiments/{family}/history_sources.jsonl)、[归档版本恢复](../../../experiments/{family}/history_restoration.jsonl)
- [候选顺序](../../../experiments/{family}/candidate_order.json)、[含原文的Agent审阅](../../../experiments/{family}/candidate_review_completed.jsonl)、[共同池](../../../experiments/{family}/candidate_pool.json)
- [新开发manifest](../../../{data['fresh_manifest_path']})、[数据冻结](../../../experiments/{family}/data_frozen.json)
- [前后向／重载](../../../experiments/{family}/preflight.json)、[定标](../../../experiments/{family}/calibration_complete.json)、[进程账本](../../../experiments/{family}/process_costs.jsonl)

GSM8K正式test与SVAMP继续封存。35训练终点、37评估对象和后续完整统计仍是本轮待完成工作；本准备记录不提供替代性的成功结论。
"""
    (out / "preparation.md").write_text(text, encoding="utf-8", newline="\n")
    prefix = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前执行：更大干净池固定预筛与随机子集复验

`{family}` 已完成历史归档SHA256/曝光/近重复审计、Agent质量审阅、新256题池与1024题fresh_development隔离、旧开发64题计时及轻量前后向/重载。三次独立K4探测正在执行；B/E/C和全会话硬上限将在完整探测后按计时冻结。尚无新开发准确率或主对照结果，不重跑历史家族、不重做SFT。

计划U5、F15、R15，共35训练终点；全部终点固定后再用37个独立进程评估同一新开发集。当前状态见[运行器收据](experiments/{family}/runner_state.json)与[准备报告](reports/comparisons/{family}/preparation.md)。GSM8K test/SVAMP封存，远端发布尚未授权。
<!-- LARGE_FILTER_CURRENT_END -->

"""
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        current = path.read_text(encoding="utf-8")
        if "<!-- LARGE_FILTER_CURRENT_BEGIN -->" in current:
            raise ValueError("Preserve existing current-state edits")
        title, remainder = current.split("\n", 1)
        path.write_text(title+"\n\n"+prefix+remainder.lstrip("\n"), encoding="utf-8", newline="\n")
    json_write(exp / "preparation_report_written.json", {"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report": (out / "preparation.md").relative_to(ROOT).as_posix(), "sha256": sha256_file(out / "preparation.md"),
        "fresh_model_predictions_exist": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
