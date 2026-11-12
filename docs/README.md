# ReasoningRL Lab 文档索引

本目录存放实验协议、复现步骤与来源说明。**结果与证据不在这里**：对照报告在 [`reports/comparisons/`](../reports/comparisons/)，运行记录在 [`runs/`](../runs/) 与 [`experiments/`](../experiments/)。

更新：2026-11-10。

## 建议阅读顺序

1. [项目 README](../README.md)：研究问题、方法与主要结果。
2. [从可运行代码理解小模型后训练](LEARNING_GUIDE.md)：按调用链读源码。
3. [实验协议](PROTOCOL.md)：采样、预算与终点定义。
4. [固定预筛实验复盘与方案](PREFILTER_REVIEW_AND_PLAN_2026-11-10.md) 与 [复现 2026-11-06 实验](REPRODUCING_REPAIR_20261106.md)：完整结果、命令与工件核验。
5. [方法定位与阅读路线](RELATED_WORK.md)：与已有工作的关系。

## 全部文档

| 文档 | 内容 | 何时读 |
| --- | --- | --- |
| [PREFILTER_REVIEW_AND_PLAN_2026-11-10.md](PREFILTER_REVIEW_AND_PLAN_2026-11-10.md) | 固定预筛结果复算、关键问题与下一轮证据优先方案 | 解读 2026-11-10 实验、规划复验 |
| [NEXT_EXPERIMENT_PROMPT_2026-11-10.md](NEXT_EXPERIMENT_PROMPT_2026-11-10.md) | 下一轮实验的完整执行方案：目标、35 终点矩阵、指标、预算与交付 | 了解后续计划；尚未启动 |
| [LEARNING_GUIDE.md](LEARNING_GUIDE.md) | 从代码理解 SFT、GRPO 与采样 | 首次读源码 |
| [PROTOCOL.md](PROTOCOL.md) | 实验协议与终点、预算、冻结规则 | 核对实验口径 |
| [REPRODUCING_LOCAL.md](REPRODUCING_LOCAL.md) | 本机（RTX3060 Laptop）复现步骤 | 在笔记本环境复跑 |
| [REPRODUCING_REPAIR_20261106.md](REPRODUCING_REPAIR_20261106.md) | 台式机 `desktoprepair_20261106_152712` 复现 | 复核主实验 |
| [DESKTOP_SETUP_20261105.md](DESKTOP_SETUP_20261105.md) | RX 9070 XT 环境接续与实测记录 | 核对硬件与环境 |
| [HARDWARE_AND_BUDGET.md](HARDWARE_AND_BUDGET.md) | 硬件组合、兼容性与首轮预算 | 估算成本 |
| [RELATED_WORK.md](RELATED_WORK.md) | 方法与已有工作的定位 | 写论文或回答追问 |
| [SOURCES.md](SOURCES.md) | 数据、模型与来源的固定版本 | 核对版本与许可 |
| [THIRD_PARTY.md](THIRD_PARTY.md) | 第三方材料与许可声明 | 对外发布前 |

## 命名与边界

- 运行家族 ID（如 `desktoprepair_20261106_152712`）是实验身份，配置、报告与运行目录共用同一 ID，不拆分、不改名。
- 日期后缀文档（`*_20261106.md`）记录当时环境与命令，硬件或依赖变化后需重新核验。
