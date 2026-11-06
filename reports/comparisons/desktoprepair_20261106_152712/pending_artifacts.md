# 本轮待提交与备份清单

家族：`desktoprepair_20261106_152712`。本轮没有执行Git提交或推送，当前HEAD仍为 `4e938eb696b1cbb4290c14515d30069ff6e50113`。历史工件保留。

Git中已跟踪文件修改 11 项，新增可见文件 4683 项（逐项路径见 [完整JSON清单](pending_files.json)，不在正文展开每份run的源码快照）。

|工件组|位置 / 处理|
|---|---|
|核心代码|`src/reasoning_rl/{training,sampling,evaluation,cli,runtime}.py`：缓存采样、显式起点、验证ID与监督/成本证据|
|配置与工程检查|`configs/main.yaml`、`configs/desktoprepair_20261106_152712_*.yaml`、三个新增测试文件；47项测试通过|
|执行与复算|新增` scripts/*repair*`及`analysis/`脚本，复现文档与固定命令收据|
|实验登记|`experiments/desktoprepair_20261106_152712/`：ID、曝光、四候选、定标、冻结协议、确认开启清单及源码摘要|
|原始证据|49份`runs/desktoprepair_20261106_152712-*`、对应`logs/`与`environment/`；监督、预测、rollout、探测、缓存和成本|
|中文报告|`reports/comparisons/desktoprepair_20261106_152712/`：报告、CSV、逐题预测、图、归因与证据审计|
|项目交接|README、PROJECT_CONTEXT、HANDOFF、PROTOCOL、台式机记录与新复现说明|
|真实适配器（Git忽略）|17份；SFT4、GRPO12（3定标+9主终点）、未训练控制1，不能都称正式对照。逐文件摘要见`migration/desktoprepair_20261106_152712_CHECKPOINTS_MANIFEST.json`|

## 已生成的本地归档

以下只在本机生成，未上传。公开基础模型、环境、缓存与完整优化器状态不在包内。归档收据在源ZIP生成之后单独提供，避免摘要自引用。

|归档|字节数|SHA256|
|---|---:|---|
|[desktoprepair_20261106_152712-source-and-evidence.zip](../../../dist/desktoprepair_20261106_152712/desktoprepair_20261106_152712-source-and-evidence.zip)|17443536|`effe8fba0d18d7e6a86ecc1bc93deeadd89a2a152d266b9bc1f20a6d633e36d3`|
|[desktoprepair_20261106_152712-lora.zip](../../../dist/desktoprepair_20261106_152712/desktoprepair_20261106_152712-lora.zip)|96570221|`22231ff336b8da17262f3cd16d4a5ab63546aaf3b0174a6c23a87d4eb4dc2440`|

[归档收据](../../../environment/desktoprepair_20261106_152712-local-archives.json) · [SHA256SUMS](../../../dist/desktoprepair_20261106_152712/SHA256SUMS.txt)。源ZIP是生成时的完整源码/证据快照，不包含随后生成的收据或本清单最终版本。

本轮不上传模型/缓存，不把本地备份写成已发布Release。提交时应将小型源码、配置、文档、原始证据纳入Git；LoRA ZIP按单独的大工件备份处理。
