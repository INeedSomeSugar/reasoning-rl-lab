# ReasoningRL Lab

<!-- FILTER_20261110_CURRENT_BEGIN -->
## 最新完成：固定S2预筛与均匀对照（2026-11-10）

`desktopfilter_20261110` 已完成：64题共同池，固定S2/K4严格保留原始成功次数1/2/3的16题，U/F各seed13/29/47；全部六个训练终点固定后，由独立进程重载评测同一开发128题。U正确数100/101/100，F为102/102/103；均值78.385%/79.948%，F−U平均+1.5625pp。固定预筛明显提高了有效训练信号，三个种子的开发正确数均高于全池均匀；稳定的推理泛化收益仍未建立。

零方差组U/F为81.109%/51.876%，计入完整筛选和独立重载的有效completion/秒为0.10233/0.17646。每F扣完整292.832841秒探测，每臂1200秒训练/筛选额度、共同1830秒端到端上限。原U/13交互中断的136步失败工件和成本区间保留；主结果来自同配置新前缀r1，不混入失败权重。

冻结后发现训练题06831自指措辞疑义，已单列其信号贡献，本轮未改金标/删题。confirmation128明确是已使用的开发数据；GSM8K test/SVAMP继续封存。下一步：保留固定筛选作为简单基线，优先在更大共同候选池、预先排除新疑义题后，用一次新冻结开发子集做同预算三配对复验。额外计算首先用于题目覆盖和复验，不直接延长同16题的训练；动态过滤留作后续独立消融，当前未见信号系统性枯竭。模型、学习率、奖励、G4和温度继续共同锁定，GSM8K test/SVAMP继续封存。

[完整报告](reports/comparisons/desktopfilter_20261110/findings.md) · [复现命令](docs/REPRODUCING_FILTER_20261110.md) · [证据审计](reports/comparisons/desktopfilter_20261110/evidence_audit.json) · [冻结协议](experiments/desktopfilter_20261110/protocol_frozen.json)。全部模型任务已结束，成果保存在本项目，尚未提交/推送；不要重跑已完成家族或启封正式测试。
<!-- FILTER_20261110_CURRENT_END -->

**小模型数学推理后训练与难度感知数据选择。** 实现数据审计、completion-only LoRA SFT、GRPO、动态采样和成本评测的完整实验链路，研究在同一硬件、共同端到端预算上限下，如何利用模型当前的解题成功率分配训练计算。

[GitHub 仓库](https://github.com/INeedSomeSugar/reasoning-rl-lab)现已公开（2026-11-10 核验）。源码、配置、报告与原始文本运行证据随 Git 保存；LoRA 权重通过 [开发预发布 Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106) 提供。

[最新实验报告](reports/comparisons/desktoprepair_20261106_152712/findings.md) · [复现命令](docs/REPRODUCING_REPAIR_20261106.md) · [逐终点指标与成本](reports/comparisons/desktoprepair_20261106_152712/standalone_endpoints.csv) · [当前交接](HANDOFF.md)

## 核心实现

- **复用训练 rollout 的难度感知采样**：初始全池探测后，直接用已生成的训练奖励维护每题成功率缓存，减少重复全池探测；保留均匀采样和静态采样作为对照。
- **可审计的 SFT / GRPO 训练**：检查 prompt/padding loss mask、EOS、实际监督 token、有限梯度、适配器权重改变量及保存重载一致性。
- **带成本约束的受控实验**：三个配对随机种子共享起点权重哈希、训练池、奖励和优化配置，记录探测、训练、采样维护与独立重载评测的实际开销。
- **可回放的证据链**：保存逐题预测、训练 rollout、采样权重、缓存状态、源码快照及成本账本。最新家族的 49 份运行通过证据审计，工程测试 47 项通过。

## 上一轮开发实验：SFT 排查与缓存采样

最新已完成家族为 `desktoprepair_20261106_152712`（2026-11-06），使用单张 **RX 9070 XT、非量化 BF16/SDPA**，模型为固定 revision 的 **Qwen3-1.7B**。该公开模型本身已后训练。

### SFT 强度对照

四个候选按预登记规则运行与选择。同为 32 步、128 条完整样本、15,160 个监督 token，将学习率从 `1e-4` 降至 `1e-5` 后，comparison128 正确数从 **64 增至 106**，截断数从 **40 降至 0**。历史退化终点的适配器 SHA256 和旧 32 题预测均得到精确复现；token 边界、mask、EOS 与重载检查未发现缺陷，重复生成的内部机制仍待解释。

选中的共同起点 **S2** 为真实 LoRA SFT：学习率 `1e-5`、8 步，名义池 128 题，**实际监督 32 题 / 3,688 token**。[完整 SFT 对照](reports/comparisons/desktoprepair_20261106_152712/findings.md)保留全部四个候选和未训练 LoRA 控制。

### 三种子结果

C/D/E 各运行 seed **13、29、47**，共 9 个 GRPO 终点，均从 S2 初始化并在新进程独立重载评测。comparison128 和 confirmation128 是预先固定、互斥的两组 **validation 开发题**；confirmation 在全部主终点固定后开启一次。

| 组别 | comparison 正确数 /128（seed 13 /29 /47） | comparison 准确率 | confirmation 正确数 /128（seed 13 /29 /47） | confirmation 准确率 |
| --- | --- | ---: | --- | ---: |
| A：公开模型 | 107 | 83.59% | 102 | 79.69% |
| B=S2：共同 SFT 起点 | 110 | 85.94% | 101 | 78.91% |
| C：均匀采样 | 108 /110 /109 | 85.16% | 102 /104 /103 | 80.47% |
| D：缓存动态采样 | 112 /110 /108 | 85.94% | 100 /103 /102 | 79.43% |
| E：初始探测后静态采样 | 111 /109 /107 | 85.16% | 103 /102 /102 | 79.95% |

C/D/E 的准确率为三种子均值。D−C 在 comparison 上为 **+0.781 个百分点**，在 confirmation 上为 **−1.042 个百分点**：当前开发结果没有建立动态采样的稳定优势。两组结果均完整保留，GSM8K test / SVAMP 尚未运行。

九组共完成 **979 个优化步骤**，其中 **244 步非零梯度**，所有梯度检查有限，9 份适配器均有实际权重更新。采样机制执行、有效学习信号与终点准确率分别记录，详见[完整报告](reports/comparisons/desktoprepair_20261106_152712/findings.md)和[证据审计](reports/comparisons/desktoprepair_20261106_152712/evidence_audit.json)。

## 实验链路与采样机制

| 组别 | 初始化与训练 | 采样方式 |
| --- | --- | --- |
| A | 固定 revision 的公开 Qwen3-1.7B | 公开模型基线 |
| B | A + completion-only LoRA SFT | 最新家族选用真实 S2 |
| C | B + GRPO | 均匀有放回采样，无探测费用 |
| D | B + GRPO | K=4 初始探测 + rollout 缓存动态更新 |
| E | B + GRPO | K=4 初始探测后固定权重，关闭后续更新与过期 |

最新主对照使用 **44 题 GRPO 训练池**，与 SFT 池互斥。D/E 分别执行并计费初始全池探测，成对种子的初始预测和权重已核对一致。

动态机制 `rollout_cache_fifo_v1_dynamic` 的具体规则为：

1. 每题最多保留最近 **16 次二值观测**，包括初始探测和后续真实训练 rollout 奖励；用 Jeffreys 先验估计 `p_i=(成功数+0.5)/(观测数+1)`。
2. 以 `g_i=4p_i(1-p_i)` 分配难度权重，混合 **25% 均匀概率质量**：`q_i=0.25/N+0.75*g_i/Σg_j`。
3. 每次训练 rollout 组结束后更新对应题目的 FIFO。某题距最近观测达到 **32 个 optimizer step**，在选题/更新边界清空缓存并回到 `p=0.5`。

该机制在初始探测后复用训练奖励，**不再生成周期性全池探测**。历史版本的周期全池刷新仍保留在旧配置中；最新家族没有补跑同协议的周期刷新对照，因此没有与该历史机制比较效果优劣。动态采样沿用课程学习与自适应采样思路，相关工作见 [RELATED_WORK.md](docs/RELATED_WORK.md)。

## 训练与成本口径

- **SFT**：`labels=-100` 屏蔽 prompt/padding；过长标准解整条丢弃并记录 ID，保留完整答案与 EOS。
- **GRPO**：固定 TRL `0.23.1`，`loss_type=grpo`、`scale_rewards=group`、`beta=0`、token 重要性比率、单轮更新，截断 completion 不参与 loss；这是本项目明确配置的 GRPO 实现。
- **评分与生成**：最终数值答案正确性为 0/1 奖励，格式合规和解析失败另记；统一 `math_numeric_v2`、`enable_thinking=False`，评测为 greedy，最新开发评测上限为 512 个新 token。
- **共同预算上限**：主对照每组预留 600 秒训练/采样、540 秒独立重载 comparison 评测、30 秒进程外围费用，共 **1,170 秒**。初始探测、模型加载、训练和采样维护均在训练/采样钟内，未花完的评测预留不回填训练。
- **实际成本**：九组训练加 comparison 重载评测实际为 **932.32–1,033.49 秒**，自然停止边界造成的训练钟超额单列；阶段 confirmation 成本另计。共同预算上限不表示实际耗时或优化步数完全相同。

`costs.jsonl` 与[逐终点账本](reports/comparisons/desktoprepair_20261106_152712/standalone_endpoints.csv)保留输入/生成 token、探测和维护开销、wall time 与进程 PyTorch 峰值显存。device elapsed 含设备等待，不等于活跃 kernel 时间或 FLOPs。

## 获取代码与复现

```powershell
git clone https://github.com/INeedSomeSugar/reasoning-rl-lab.git
Set-Location reasoning-rl-lab
```

使用 **Python 3.12** 和项目独立 `.venv`；上层依赖固定在 [requirements/common.txt](requirements/common.txt) 与 [pyproject.toml](pyproject.toml)，数据与模型分别由 [data.sources.lock.json](configs/data.sources.lock.json)、[models.lock.json](configs/models.lock.json) 锁定。模型 revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，依赖、缓存、下载和运行数据均保存在项目内。

| 环境 | 安装与运行入口 |
| --- | --- |
| RX 9070 XT / Windows ROCm | [环境实测与安装记录](docs/DESKTOP_SETUP_20261105.md)、[最新实验复现](docs/REPRODUCING_REPAIR_20261106.md)；使用非量化 BF16/SDPA 与 `--hardware rocm` |
| RTX 3060 Laptop / CUDA | [笔记本历史复现](docs/REPRODUCING_LOCAL.md)；保留 NF4、不同 attention 后端及失败记录 |

环境和锁定模型准备完成后，先运行工程检查：

```powershell
$py='.venv\Scripts\python.exe'
& $py -m pytest -q
& $py -m reasoning_rl --help
```

以下示例用于已按文档准备完成的 RX 9070 XT 环境，用新运行名复现 S2 和 comparison 评测：

```powershell
. .\scripts\desktop_env.ps1
$py='.venv\Scripts\python.exe'
$ReplayPrefix='replay_'+(Get-Date -Format 'yyyyMMdd_HHmmss')
$Original='desktoprepair_20261106_152712'
& $py -m reasoning_rl sft --hardware rocm --name "$ReplayPrefix-S2" --config "configs/${Original}_S2.yaml"
& $py -m reasoning_rl evaluate --hardware rocm --name "$ReplayPrefix-S2-comparison" --config "configs/${Original}_eval.yaml" --adapter "checkpoints/$ReplayPrefix-S2"
```

三种子 C/D/E 命令、原 S2 权重哈希、阶段确认和证据复算见[完整复现说明](docs/REPRODUCING_REPAIR_20261106.md)。已有运行名不可覆盖；时间预算停止下的步数与终点权重可能随设备吞吐变化。LoRA 归档用于权重恢复和终点评测，不含可精确续训的完整优化器状态。

## 公开工件

当前 [开发预发布](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106) 提供 4 个附件，大小和 SHA256 已有[发布核验收据](environment/desktoprepair_20261106_152712-github-publication.json)：

| 附件 | 内容 |
| --- | --- |
| [source-and-evidence.zip](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/download/desktop-repair-cache-dev-20261106/desktoprepair_20261106_152712-source-and-evidence.zip) | 本轮源码与证据归档快照 |
| [lora.zip](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/download/desktop-repair-cache-dev-20261106/desktoprepair_20261106_152712-lora.zip) | 17 份适配器：4 份 SFT、9 份主 GRPO、3 份定标 GRPO、1 份未训练控制 |
| [SHA256SUMS.txt](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/download/desktop-repair-cache-dev-20261106/SHA256SUMS.txt) | 两个 ZIP 的摘要 |
| [archive_receipt.json](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/download/desktop-repair-cache-dev-20261106/archive_receipt.json) | 本地归档收据 |

源码 ZIP 早于最后收尾与发布元数据，继续工作以 `main` 为准；基础模型、虚拟环境和完整优化器状态不在附件中。发布收据中的私有可见性字段记录的是 **2026-11-06 发布时状态**，当前仓库已公开。第三方材料与许可见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。

## 数据与保留评测

GSM8K 官方 train 按种子 **20261105** 划分为 **6,961 题 train /512 题 validation**。按题目 word(1,2) TF-IDF cosine ≥0.90 构建近重复簇并整体划分，发现 208 个近重复对、0 个归一化完全重复，当前阈值下无跨 train/validation 的近重复对。[数据 manifest](data/manifest.json)保留来源、许可、revision 与内容哈希；两个基准的预训练污染情况未知。

GSM8K 官方 **test 1,319 题**和 **SVAMP 1,000 题**保持封存，不参与训练、成功率估计、调参或选检查点。当前 `configs/main.yaml` 的 stage 为 `main_capacity_calibrated_development_only`，`formal_test_protocol_ready=false`；最终协议、具体端点与独立评测预算尚未冻结。规则见 [PROTOCOL.md](docs/PROTOCOL.md)。

## 历史记录与代码导航

| 入口 | 内容 |
| --- | --- |
| [2026-11-05 台式机开发报告](reports/comparisons/desktoprocm10pilot_20261105/findings.md) | 历史 28 题池、C/D 三种子与 E 单种子；SFT 退化及周期刷新证据 |
| [笔记本结果](reports/LOCAL_RESULTS.md)、[运行事件](reports/runtime_incidents.md) | RTX 3060 短链路、CUDA 异常与恢复记录 |
| [迁移归档](migration/README_ZH.md) | 初次迁移与历史 16 份 LoRA 下载核验 |
| [学习指南](docs/LEARNING_GUIDE.md)、[相关工作](docs/RELATED_WORK.md) | 公式、设计取舍、代码入口与参考来源 |
| [data.py](src/reasoning_rl/data.py)、[scoring.py](src/reasoning_rl/scoring.py) | 数据锁定、划分隔离与可验证数值评分 |
| [modeling.py](src/reasoning_rl/modeling.py)、[training.py](src/reasoning_rl/training.py)、[sampling.py](src/reasoning_rl/sampling.py) | 模型加载、loss mask、LoRA/TRL 训练与缓存采样 |
| [evaluation.py](src/reasoning_rl/evaluation.py)、[reporting.py](src/reasoning_rl/reporting.py) | 原始预测、评测凭据、指标与历史通用报告 |
| [analysis/](analysis/)、[experiments/](experiments/)、[runs/](runs/) | 最新家族分析、预登记/冻结协议与原始运行证据 |

历史轮次按硬件、精度、训练池和评测集分别保留；最新结论以本页链接的 `desktoprepair_20261106_152712` 报告为准。
