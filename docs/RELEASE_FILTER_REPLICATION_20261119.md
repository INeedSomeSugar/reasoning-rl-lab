# 256 题固定预筛复验与同规模随机子集消融

实验家族 `desktopfilterrep_2026-11-10_235416` 在同一 256 题池上比较完整池均匀采样 U、固定 S2/K4 预筛 F 和同规模随机子集 R。三次探测保留 70、62、70 题。35 个训练终点均已保存，32/37 个对象在同一批 1,024 道隔离开发题上完成独立评估。5 个对象耗尽重试，缺失行和失败输出均保留。

| 新开发准确率比较 | 可用配对 | 平均差（百分点） | 条件 95% 区间（百分点） |
| --- | ---: | ---: | ---: |
| F−U | 11/15 | −0.293 | [−0.923, +0.346] |
| F−R | 12/15 | −0.285 | [−0.741, +0.155] |
| F−S2 | 13/15 | −0.090 | [−1.036, +0.872] |
| F−A | 13/15 | +0.008 | [−1.021, +1.037] |

三个完整训练种子块中，有效 completion/计费端到端秒相对 U 提高 65.06%。准确率区间固定已评估的模型和种子，按新开发题近重复簇联合重采样 20,000 次。原定完整矩阵的两个决策阈值因缺失留空。GSM8K 正式 test 和 SVAMP 保持封存。

[中文报告与图表](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/fixed-prefilter-replication-dev-20261119/reports/comparisons/desktopfilterrep_2026-11-10_235416/findings.md) · [复现与离线复算](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/fixed-prefilter-replication-dev-20261119/docs/REPRODUCING_FILTER_REPLICATION_20261110.md)

## 下载与校验

- `desktopfilterrep_2026-11-10_235416-source-and-evidence.zip` 包含冻结源码、数据清单、三次探测、35 个训练终点的原始记录、32 份完整预测、失败输出、成本和盲化 Agent 审阅。内含 `release_file_manifest.json`。
- `desktopfilterrep_2026-11-10_235416-lora-checkpoints.zip` 包含 35 个 GRPO 终点、共同 S2 及 3 个旧数据诊断权重目录。适配器与配套 tokenizer 文件按原始相对路径保存。
- `desktopfilterrep_2026-11-10_235416-optimizer-diagnostics.zip` 包含 37 个 Adam/RNG 诊断状态。精确 sampler/dataloader 续训未验证。
- `archive_receipt.json` 和 `SHA256SUMS.txt` 登记提交、附件字节数和 SHA256。三个归档都使用 `reasoning-rl-lab/` 路径前缀。

基础模型按 `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` 获取。环境、缓存和含设备实例路径的原始 Windows PnP 快照保留在原工作区。发布包中的运行故障证据包括原始进程日志、失败退出状态、部分预测和计费记录。实验收尾 manifest 为历史记录，发布包的实有文件以发布清单为准。GitHub 传输在实验会话结束后执行。
