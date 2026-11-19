# 更大干净候选池复验：准备与定标已完成

家族：`desktopfilterrep_2026-11-10_235416`。2026-11-10登记，跨北京时间2026-11-11继续执行。当前阶段为三次独立完整K4探测；尚无本轮新开发结果。本文件记录已完成证据，最终发现以同目录`findings.md`及模型任务完成收据为准。

## 数据资格已建立

核验四份项目内已发布源码／证据ZIP的完整SHA256，比对归档与当前checkout的原始运行工件。未发现缺失的归档运行证据；另存10份不同历史版本，没有覆盖现有工件。扫描不仅覆盖`runs`，还包含历史配置、实验登记、训练/探测/评估输出、报告、脚本和日志。曝光账本覆盖名义SFT128、实际监督记录、GRPO池与实际选择、定标、探测、开发评估及所有已保存预测；对仅有历史提及、不能证明模型接触的ID也保守排除。

- 历史被提及或配置的ID：676；原validation512全部排除，不假设其剩余题目仍干净。
- 连同已知疑义题和相关簇，共排除902题；剩余6571题／6565簇。
- 沿用原word(1,2) TF-IDF cosine≥0.90审计边，并增加不读取正式holdout题干的official-train单独重拟合检查：8对；归一化完全重复0对。最终簇取两套边及完全重复的并集，所有排除和开发抽样按完整簇。
- 按种子20261114的固定候选顺序，Agent读题干和原参考解并复核数量关系。前283题中排除27题，保留共同256题；另看到的5题补位样例也留档。所有320题审阅／备用包均排除在新开发集之外。
- 按种子20261115，从其余合格原train簇冻结新开发1024题，具有独立manifest和`fresh_development`加载入口。原manifest与旧split未覆盖。普通train/select/develop加载器会剔除已登记新开发题；该新入口拒绝训练或普通holdout权限，评估还要求全部35个训练单元终止并锁定全部成功权重。

“未曝光”仅指可恢复项目记录中的模型训练／探测／评估及保守历史排除，不能证明公开Qwen预训练未见。题目审阅身份为**Agent审阅**，`human_reviewed=false`。没有改原文、金标或冻结评分器。

## 旧开发题定标与工程检查

共同真实S2完整checkpoint SHA256：`e8da01dae86ea2d7a95e19a645440510ec7dce2e4ceb546b26b3ff2dc9ac0bff`，11份基础模型/tokenizer文件逐项核对。单RX9070XT、非量化BF16/SDPA加载、有限非零梯度、冻结骨干梯度0、Adam零梯度后仍可能更新权重以及保存重载均已实测。诊断权重仅作检查，从未作为新训练起点；重载logits最大绝对差0.0。

已使用的开发64题独立S2进程耗时218.602095秒。历史训练16题上的8步GRPO定标完成，未对新开发集运行模型。工程检查74项通过（后续实现如有改动，将重新核验并保留日志）。

三次256题完整K4探测的实测进程时间P_j完成后，以`B=max(3600,300*ceil(3*max(P_j)/300))`冻结共同额度。F每个终点扣完整P_j，R也扣P_j配平可训练时间，同时单列R该项属于实验配平，不将其写成随机方法本身的必需物理成本。E使用旧64题定标推算N并加35%及120秒后向上取整300秒；O固定300秒，C=B+E+O。全会话上限另包含全矩阵、每单元一次完整失败重试、20%外围余量及6小时分析预留，不沿用上一轮4小时上限。

## 已保留的准备异常

最初一次隐藏后台启动返回PID5484，但未观察到运行器入口收据或模型日志，随后已确认进程不存在；两份空stdout/stderr文件保留。未确认其根因，未记为成功模型任务。随后改用持久受控进程，逐子进程记录PID、命令、心跳、退出码和成本。一次交互查看候选时出现GBK输出异常，已用项目UTF-8环境继续读取；不影响原始UTF-8数据或金标。CIM只读进程查询不可用，改用项目Python/psutil与进程证据，不修改系统设置。

## 可审阅证据

- [前瞻登记](../../../experiments/desktopfilterrep_2026-11-10_235416/preregistration.json)
- [历史审计](../../../experiments/desktopfilterrep_2026-11-10_235416/history_audit.json)、[逐ID曝光账本](../../../experiments/desktopfilterrep_2026-11-10_235416/history_exposure_ledger.jsonl)、[来源版本](../../../experiments/desktopfilterrep_2026-11-10_235416/history_sources.jsonl)、[归档版本恢复](../../../experiments/desktopfilterrep_2026-11-10_235416/history_restoration.jsonl)
- [候选顺序](../../../experiments/desktopfilterrep_2026-11-10_235416/candidate_order.json)、[含原文的Agent审阅](../../../experiments/desktopfilterrep_2026-11-10_235416/candidate_review_completed.jsonl)、[共同池](../../../experiments/desktopfilterrep_2026-11-10_235416/candidate_pool.json)
- [新开发manifest](../../../data/fresh_development/desktopfilterrep_2026-11-10_235416/manifest.json)、[数据冻结](../../../experiments/desktopfilterrep_2026-11-10_235416/data_frozen.json)
- [前后向／重载](../../../experiments/desktopfilterrep_2026-11-10_235416/preflight.json)、[定标](../../../experiments/desktopfilterrep_2026-11-10_235416/calibration_complete.json)、[进程账本](../../../experiments/desktopfilterrep_2026-11-10_235416/process_costs.jsonl)

GSM8K正式test与SVAMP继续封存。35训练终点、37评估对象和后续完整统计仍是本轮待完成工作；本准备记录不提供替代性的成功结论。
