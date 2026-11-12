# ReasoningRL Lab

**小模型数学推理的可审计后训练实验：在固定计算预算下，按题目难度分配 GRPO 训练。**

ReasoningRL Lab 实现了一条完整的后训练实验链路：数据审计 → completion-only LoRA SFT → GRPO → 难度感知选题 → 独立重载评测 → 成本记账。所有实验在单张消费级显卡上运行，研究问题是：在共同的端到端预算上限下，用模型当前在每道题上的成功率来决定训练哪些题，能否提高有效训练信号，并进一步提升推理准确率。

![Python](https://img.shields.io/badge/python-3.12-3776AB)
![TRL](https://img.shields.io/badge/TRL-0.23.1-yellow)
![Model](https://img.shields.io/badge/model-Qwen3--1.7B-purple)
![Hardware](https://img.shields.io/badge/GPU-RX%209070%20XT%20(ROCm)-red)
![Status](https://img.shields.io/badge/status-development%20results-lightgrey)

## 目录

- [研究动机](#研究动机)
- [主要结论](#主要结论)
- [方法](#方法)
- [实验结果](#实验结果)
- [快速开始与复现](#快速开始与复现)
- [发布工件](#发布工件)
- [数据与评测协议](#数据与评测协议)
- [项目结构](#项目结构)
- [局限性](#局限性)
- [文档](#文档)

## 研究动机

GRPO 对同一道题采样一组回答（本项目每组 4 个），用组内奖励的相对高低计算优势。如果一组回答**全对或全错**，优势全部为 0，这一步就没有有效的策略梯度。

对历史实验的全部原始 rollout 复算后发现，979 个训练组中有 **735 组（75.1%）组内奖励完全相同**：对 Qwen3-1.7B 来说，GSM8K 训练题大多太容易，少数又太难。大部分训练计算因此没有产生学习信号。

本项目比较几种按难度选题的策略：

| 策略 | 做法 |
| --- | --- |
| 均匀采样 | 在训练池中均匀有放回采样，没有探测成本 |
| 固定预筛 | 先对每题探测 K=4 次，只保留成功 1–3 次（既非全对也非全错）的题，再均匀采样 |
| 静态加权 | 初始探测后按估计难度固定采样权重 |
| 缓存动态加权 | 初始探测后，复用训练中产生的 rollout 奖励持续更新每题的成功率估计 |

所有探测成本都计入同一个预算，对照组在同一硬件、同一起点权重、同一奖励和优化配置下运行。

## 主要结论

> [!IMPORTANT]
> 以下全部是 **GSM8K validation 划分上的开发集结果**，每组 3 个随机种子。GSM8K test 与 SVAMP 保持封存，至今没有用于任何训练或选择。这里的结论不能当作正式测试结果或泛化结论。

1. **固定预筛明显增加了有效训练信号。** 零优势组的比例从 81.1% 降到 51.9%；计入完整探测成本和独立评测后，每秒得到的有效训练样本多 72%。
2. **准确率提升很小，统计上尚未确立。** 3 个种子里预筛都比均匀采样高，平均高 1.56 个百分点（约多答对 2/128 题），但题目 bootstrap 区间为 [−0.78, +4.17] 个百分点，包含 0。
3. **缓存动态加权没有显示出稳定优势。** 相对均匀采样，它在一组开发题上高 0.78 个百分点，在另一组上低 1.04 个百分点。
4. **SFT 学习率是早期性能退化的主因。** 学习率从 `1e-4` 降到 `1e-5` 后，SFT 模型在 128 道开发题上的正确数从 64 恢复到 106，截断数从 40 降到 0。

## 方法

### 训练链路

| 阶段 | 配置 |
| --- | --- |
| 基础模型 | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)，固定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`（该公开模型本身已做过后训练） |
| SFT | completion-only LoRA：用 `labels=-100` 屏蔽 prompt 和 padding；过长的标准解整条丢弃并记录 ID，保留完整答案和 EOS |
| GRPO | TRL `0.23.1`，`loss_type=grpo`，`scale_rewards=group`，`beta=0`，单轮更新，每组 4 个回答；截断的回答不参与 loss |
| 奖励 | 最终数值答案正确记 1、错误记 0；格式合规和解析失败另外记录 |
| 评测 | greedy 解码，`enable_thinking=False`，最多生成 512 个新 token；每个终点在新进程中重新加载权重后评测 |
| 硬件 | 单张 AMD RX 9070 XT（Windows ROCm），非量化 BF16 / SDPA |

### 缓存动态加权（`rollout_cache_fifo_v1_dynamic`）

1. 每题最多保留最近 16 次二值观测，包括初始探测和训练 rollout 的奖励，用 Jeffreys 先验估计成功率：$p_i = (s_i + 0.5) / (n_i + 1)$。
2. 难度权重 $g_i = 4p_i(1-p_i)$，再混入 25% 的均匀概率：$q_i = 0.25/N + 0.75 \cdot g_i / \sum_j g_j$。
3. 如果某题连续 32 个 optimizer step 没有新观测，就清空它的缓存，成功率回到先验值 0.5。

初始探测之后完全复用训练奖励，不再做周期性的全池探测。

### 预算与成本记账

- 每个终点设有共同的端到端预算上限，包括训练与选题时间、独立重载评测预留和进程外围开销。探测、模型加载、训练和采样维护都计入训练时间，评测没用完的预留不会挪给训练。
- 每个 rollout、优化步和评测进程都单独记账：输入 / 生成 token、探测和维护开销、墙钟时间、PyTorch 峰值显存。中断或失败的运行也保留成本记录，不从总账中删掉。
- 实际改变权重的证据来自逐步记录的梯度、Adam 状态和权重差，而不是 loss 数值。

## 实验结果

### 实验 1：固定预筛 vs 全池均匀（2026-11-10）

运行 ID `desktopfilter_20261110`。训练池共 64 题。固定预筛（F）用共同的 SFT 起点 S2 对每题探测 4 次，按成功次数 0/1/2/3/4 分别有 8/5/3/8/40 题，严格保留成功 1–3 次的 16 题。每个终点的训练与选题时间上限为 1,200 秒，端到端上限为 1,830 秒；F 的每个终点都扣除完整的 292.8 秒探测时间。

**开发集准确率**（同一组 128 道题，SFT 起点 S2 为 101/128）

| 种子 | 均匀 U | 预筛 F | F − U |
| --- | ---: | ---: | ---: |
| 13 | 100/128 | 102/128 | +1.56 pp |
| 29 | 101/128 | 102/128 | +0.78 pp |
| 47 | 100/128 | 103/128 | +2.34 pp |
| **均值** | **78.39%** | **79.95%** | **+1.56 pp** |

**训练信号与成本**（3 个种子合计）

| 指标 | 均匀 U | 预筛 F |
| --- | ---: | ---: |
| 零优势组比例 | 81.1% | 51.9% |
| 非零梯度步 / 优化步 | 126 / 667 | 218 / 453 |
| 有效优势样本数 | 498 | 864 |
| 有效样本 / 计费秒（含探测与评测） | 0.102 | 0.176 |
| 平均端到端计费时间 | 1,622 s | 1,632 s |
| 实际训练覆盖的题数（每个种子） | 64 / 63 / 63 | 16 / 16 / 16 |

信号密度的提升和题目覆盖的收窄同时发生。逐题看，F 对而 U 错有 10 次，反过来有 4 次，涉及 10 道不同的题。

报告：[实验报告](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/reports/comparisons/desktopfilter_20261110/findings.md) · [复盘与下一轮方案](docs/PREFILTER_REVIEW_AND_PLAN_2026-11-10.md) · [逐终点成本](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/reports/comparisons/desktopfilter_20261110/standalone_endpoints.csv) · [证据审计](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/reports/comparisons/desktopfilter_20261110/evidence_audit.json) · [冻结协议](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/experiments/desktopfilter_20261110/protocol_frozen.json)

### 实验 2：缓存动态加权（2026-11-06）

运行 ID `desktoprepair_20261106_152712`。训练池 44 题，和 SFT 训练池不重叠。C/D/E 三组各跑 3 个种子，都从 S2 初始化。comparison 和 confirmation 是两组事先固定、互不重叠的开发题（各 128 题）；confirmation 在所有终点确定后才评测一次。

| 组别 | comparison（种子 13 / 29 / 47） | 均值 | confirmation（种子 13 / 29 / 47） | 均值 |
| --- | --- | ---: | --- | ---: |
| A：公开模型 | 107 | 83.59% | 102 | 79.69% |
| B = S2：SFT 起点 | 110 | 85.94% | 101 | 78.91% |
| C：均匀采样 | 108 / 110 / 109 | 85.16% | 102 / 104 / 103 | 80.47% |
| D：缓存动态加权 | 112 / 110 / 108 | 85.94% | 100 / 103 / 102 | 79.43% |
| E：初始探测后静态加权 | 111 / 109 / 107 | 85.16% | 103 / 102 / 102 | 79.95% |

D − C 在 comparison 上为 +0.78 pp，在 confirmation 上为 −1.04 pp，没有一致方向。9 组共 979 个优化步，其中 244 步梯度非零；所有梯度均为有限值，9 份适配器的权重都有实际更新。

报告：[findings.md](reports/comparisons/desktoprepair_20261106_152712/findings.md) · [逐终点成本](reports/comparisons/desktoprepair_20261106_152712/standalone_endpoints.csv)

### 实验 3：SFT 强度排查（2026-11-06）

历史实验中 SFT 后准确率下降。先用完全相同的适配器权重复现了这次退化；token 边界、loss mask、EOS 和保存重载检查都没有发现缺陷。在步数（32 步）、样本（128 条）和监督 token（15,160 个）都不变的情况下，把学习率从 `1e-4` 降到 `1e-5`，comparison 正确数从 64 变为 106，截断数从 40 变为 0。重复生成背后的具体机制仍未解释清楚。

按预先登记的规则，选定 S2（学习率 `1e-5`，8 步，实际监督 32 条样本、3,688 个 token）作为后续 GRPO 的共同起点。

## 快速开始与复现

### 环境

- Python 3.12（`>=3.12,<3.13`），每个项目使用独立的 `.venv`
- 依赖版本固定在 [requirements/common.txt](requirements/common.txt) 和 [pyproject.toml](pyproject.toml)
- 数据和模型来源分别由 [data.sources.lock.json](configs/data.sources.lock.json) 和 [models.lock.json](configs/models.lock.json) 锁定

| 硬件 | 安装与复现文档 |
| --- | --- |
| AMD RX 9070 XT / Windows ROCm | [环境安装记录](docs/DESKTOP_SETUP_20261105.md) |
| NVIDIA RTX 3060 Laptop / CUDA | [笔记本复现](docs/REPRODUCING_LOCAL.md)（含 NF4 量化和失败记录） |

```powershell
git clone https://github.com/INeedSomeSugar/reasoning-rl-lab.git
Set-Location reasoning-rl-lab
# 按上表文档创建 .venv 并安装依赖后：
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m reasoning_rl --help
```

### 命令行

```text
python -m reasoning_rl <command>

  prepare-data     下载、校验并划分数据（含近重复聚类）
  hardware         记录硬件与运行环境
  download-model   按锁定 revision 下载模型
  check-model      检查模型加载与运行
  sft              completion-only LoRA SFT
  grpo             GRPO 训练（--strategy uniform / static / dynamic）
  evaluate         独立重载评测
  freeze           冻结待正式评测的检查点
  compare          汇总多个终点
  report           生成报告
```

评测封存测试集时必须提供冻结计划（`--frozen-plan`），并且在生成开始前就写入评测登记，同一个终点不能重复评测。固定预筛实验的探测和编排脚本在 [`scripts/`](scripts/) 下。

### 复现实验

| 实验 | 复现文档 |
| --- | --- |
| 固定预筛 vs 均匀（2026-11-10） | [REPRODUCING_FILTER_20261110.md](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/docs/REPRODUCING_FILTER_20261110.md) · [复盘与下一轮方案](docs/PREFILTER_REVIEW_AND_PLAN_2026-11-10.md) |
| 缓存动态加权与 SFT 排查（2026-11-06） | [REPRODUCING_REPAIR_20261106.md](docs/REPRODUCING_REPAIR_20261106.md) |

已有运行 ID 的目录不会被覆盖，复现需要使用新的运行名。训练按时间预算停止，因此实际步数和终点权重会随设备吞吐量变化。

## 发布工件

LoRA 适配器通过 GitHub Releases 发布，附件大小和 SHA256 都有核验记录。基础模型、虚拟环境和缓存不包含在内。

| Release | 内容 | 核验记录 |
| --- | --- | --- |
| [fixed-prefilter-dev-20261110](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/fixed-prefilter-dev-20261110) | 9 份适配器（6 个 U/F 终点、S2、定标、重载诊断）、7 份 Adam/RNG 诊断状态、源码与证据 ZIP | [publication.json](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/environment/desktopfilter_20261110-github-publication.json) |
| [desktop-repair-cache-dev-20261106](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106) | 17 份适配器（4 份 SFT、9 份主 GRPO、3 份定标、1 份未训练对照）、源码与证据 ZIP | [publication.json](environment/desktoprepair_20261106_152712-github-publication.json) |

适配器可用于恢复权重和评测终点。目前还不支持从中断处精确续训（没有恢复 dataloader / sampler 位置）。

## 数据与评测协议

- **划分**：用种子 `20261105` 把 GSM8K 官方 train 划分为 6,961 道训练题和 512 道验证题。按题面 word (1,2)-gram TF-IDF 余弦相似度 ≥ 0.90 聚成近重复簇，以簇为单位划分，验证集和训练集之间没有近重复对。来源、许可、revision 和内容哈希见 [data/manifest.json](data/manifest.json)。
- **封存测试集**：GSM8K test（1,319 题）和 SVAMP（1,000 题）不参与训练、成功率估计、调参或检查点选择。正式测试协议目前还没有冻结（`formal_test_protocol_ready=false`）。
- **实验纪律**：协议和配置都在运行前冻结并记录哈希；所有种子的结果全部保留，不挑选最好的种子；每个终点都在新进程中重新加载后评测。规则见 [PROTOCOL.md](docs/PROTOCOL.md)。
- **测试**：本机 47 项工程测试通过；截至固定预筛发布提交 `def2a1b` 共 59 项回归测试通过（[记录](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/4ea43b03ce76a38fe5a525dd83a65e9751398d5a/environment/desktopfilter_20261110-pytest.txt)）。

## 项目结构

```text
reasoning-rl-lab/
├── src/reasoning_rl/
│   ├── data.py          # 数据锁定、近重复聚类与划分隔离
│   ├── scoring.py       # 数值答案解析与 0/1 奖励
│   ├── modeling.py      # 模型加载、LoRA、prompt 渲染与 SFT 编码
│   ├── training.py      # SFT / GRPO 训练与 loss mask 审计
│   ├── sampling.py      # 均匀、静态与缓存动态加权
│   ├── prefiltering.py  # 固定预筛
│   ├── evaluation.py    # 独立重载评测与逐题预测
│   ├── reporting.py     # 指标与报告
│   └── cli.py           # 命令行入口
├── configs/             # 实验配置与数据 / 模型锁定文件
├── scripts/             # 实验编排、探测、冻结与分析脚本
├── experiments/         # 预登记与冻结协议、成本账本
├── runs/                # 原始运行记录（rollout、采样权重、缓存状态）
├── reports/comparisons/ # 各实验的报告、图表与逐终点 CSV
├── environment/         # 硬件、依赖与发布核验记录
├── analysis/            # 结果复算与审计脚本
├── docs/                # 协议、复现步骤、学习指南与来源说明
└── tests/
```

## 局限性

- **单一模型和数据集**：只测试了 Qwen3-1.7B 和 GSM8K。这个公开模型本身已经后训练过，GSM8K 对它偏容易，这正是零优势组比例高的原因，结论不一定适用于更弱的模型或更难的数据。
- **样本量小**：每组 3 个种子、128 道开发题；bootstrap 区间只覆盖题目层面的不确定性，没有覆盖训练种子、探测种子和训练池选择带来的变化。
- **开发集复用**：confirmation 开发题在实验 2 中已经用过。
- **训练时长短**：每个终点的训练时间预算只有 10–20 分钟，没有画出完整的学习曲线。
- **数据质量**：冻结协议之后，发现 1 道训练题（`gsm8k-train-06831`）题面措辞有歧义。为保证可比性，本轮没有修改它，其影响已单独统计。
- **答案正确不等于推理正确**：评分只看最终数值，抽样审阅中发现过推理过程错误、答案却碰巧正确的情况。

## 文档

完整索引见 [docs/README.md](docs/README.md)。

- [学习指南](docs/LEARNING_GUIDE.md)：按调用链阅读源码，附公式和设计取舍
- [实验协议](docs/PROTOCOL.md)：采样、预算、终点定义和冻结规则
- [相关工作](docs/RELATED_WORK.md)：与课程学习、自适应采样等工作的关系
- [硬件与预算](docs/HARDWARE_AND_BUDGET.md)
- [数据与模型来源](docs/SOURCES.md) · [第三方许可](docs/THIRD_PARTY.md)
- [项目上下文与历史记录](PROJECT_CONTEXT.md)：各轮实验的完整状态、历史硬件记录和失败记录
