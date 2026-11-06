# ReasoningRL Lab

<!-- REPAIR_20261106_PUBLICATION_BEGIN -->
## GitHub 发布已完成（2026-11-06）

本轮代码、配置、报告及49份运行证据已推送私有仓库main，源码提交为 [d4eec11](https://github.com/INeedSomeSugar/reasoning-rl-lab/commit/d4eec11d675e079e7462f601a7ff98487801c18a)；[开发预发布](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106) 已发布。4个附件的远端大小与SHA256全部匹配：[源码/证据ZIP](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106)、17份适配器ZIP、SHA256SUMS.txt、archive_receipt.json。仓库可见性保持私有。

17份适配器包括4份SFT、9份主GRPO、3份定标GRPO和1份未训练控制，不能都标成正式对照。源ZIP为本轮较早的收尾快照，不等同于当前main；main另包含最后收尾与发布收据。继续工作优先克隆main，按 [发布核验收据](environment/desktoprepair_20261106_152712-github-publication.json) 和检查点manifest核对下载；不包含基础模型、环境或完整优化器状态。

原待提交清单、session_final和本地归档收据保留的是发布前状态，本次发布不改写实验耗时、原始预测、冻结配置或test/SVAMP封存边界。
<!-- REPAIR_20261106_PUBLICATION_END -->

<!-- REPAIR_20261106_CURRENT_BEGIN -->
## 最新完成状态：2026-11-06

家族 `desktoprepair_20261106_152712` 已完整完成：实际 tokenizer/标签/EOS/重载审计、未训练 LoRA 控制、四个预登记 SFT 候选、成本定标、C/D/E 各 seed13/29/47，以及所有终点独立重载和互斥128题的一次阶段确认。模型任务结束于北京时间 2026-11-06 20:12:10；本轮完整墙钟见 `experiments/desktoprepair_20261106_152712/session_final.json`。

S1 以完全相同的适配器权重复现历史 SFT 退化；同32步降低学习率后比较集从64/128变为106/128、截断40变0。按预登记规则选真实 S2（1e-5、8步、实际32条样本/3688监督token），没有把未训练适配器标为 SFT。所查边界、mask、EOS与重载未发现缺陷；重复输出的内部机制仍未确认。

固定 comparison128：A107、S2=110；C=108/110/109，D=112/110/108，E=111/109/107。D−C 均值 +0.781pp。固定 confirmation128：A=102、S2=101；C=102/104/103，D=100/103/102，E=103/102/102，D−C 均值 -1.042pp。分母均为128；两组都是 validation 开发数据，不能写成正式独立测试或稳定泛化收益。

新主线为 `rollout_cache_fifo_v1_dynamic`，44题训练池，D/E各自初始K4探测；D复用真实训练rollout的最近16次奖励，32步无新观测回归先验，E固定初始权重。每组600秒训练/采样钟、540秒重载评测预留及30秒外围预留。979个主优化步骤均检查有限、244步非零，九个主适配器确实更新。没有补跑同协议周期全池动态基线，因此不声称优于该历史方法。

工程47项测试、69个依赖兼容检查通过；本轮 49 份运行完成证据审计，含保存权重改变量和精确选题/缓存重放。当前进程隔离为单 RX9070XT，非量化BF16/SDPA；公开模型11个文件已复核，版本与数据均沿用锁定来源。

`configs/main.yaml` 已改为实测容量配置 `main_capacity_calibrated_development_only`，不再沿用未测量2048题/NF4草案。最终正式方案、端点选择及独立测试预算仍未冻结；GSM8K test/SVAMP继续封存。具备真实开发对照的工程条件，尚无足够证据宣称动态策略优势或直接启封正式测试。

[完整中文报告](reports/comparisons/desktoprepair_20261106_152712/findings.md) · [复现命令](docs/REPRODUCING_REPAIR_20261106.md) · [逐终点成本](reports/comparisons/desktoprepair_20261106_152712/standalone_endpoints.csv) · [证据审计](reports/comparisons/desktoprepair_20261106_152712/evidence_audit.json)。本轮源码、配置、报告及原始工件已推送main，适配器与源码备份已上传 [开发预发布](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106)，远端摘要核验通过。此前历史结果与发布前快照保持原样。
<!-- REPAIR_20261106_CURRENT_END -->

**小模型推理后训练与难度感知数据选择。** 面向大模型算法实习的独立研究工程项目。

研究问题：在**同一硬件、相同端到端计算预算**下，用模型当前解题成功率指导训练题采样，能否比均匀随机采样更有效地改善数学推理？主要交付是训练代码、检查点、原始运行证据和受控实验，先看[当前交接与证据边界](HANDOFF.md)、[RX 9070 XT 开发结果](reports/comparisons/desktoprocm10pilot_20261105/findings.md)；[笔记本历史结果](reports/LOCAL_RESULTS.md)和[历史自动报告](reports/experiment_report.md)单列保留。

## 2026-11-05 历史真实结果

RX 9070 XT 台式机已自动完成 main 克隆、Release SHA256 核验、16 份历史 LoRA 恢复、项目独立 ROCm 环境与公开模型下载。非量化 BF16/SDPA 的前后向、LoRA、保存重载和短链路实测通过，40 项工程测试通过。

按实测成本扩大的开发轮次也已完成：128 题/32 步 SFT，独立 28 题 GRPO 池；C/D 各 seed 13/29/47，E 仅 seed 13，全部终点独立重载。在共同 32 道开发验证题上，A=28/32、B=16/32，C 均值58.33%、D均值56.25%、E=17/32。D−C 配对差均值 −2.08 个百分点，没有动态优势证据，所有训练终点仍低于原公开模型。SFT 退化与输出重复/截断证据保留，根因未确认。D seed29 实际全池刷新并改变采样权重；其他两个种子的后续探测不完整，耗时计入但未更新分布。66 份运行证据审计通过。

这是三配对种子开发预实验，正式 main 与最终独立测试尚未完成；test/SVAMP 继续封存。训练池仅28题、验证仅32题、静态仅一次，不能写成泛化提升。见[当前交接](HANDOFF.md)、[台式机记录](docs/DESKTOP_SETUP_20261105.md)、[扩大轮次报告](reports/comparisons/desktoprocm10pilot_20261105/findings.md)、[短链路报告](reports/comparisons/desktoprocm10bf16_20261105/findings.md)。本轮已结束，无活动训练；代码、报告及原始日志已推送 main，[台式机预发布 Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-rocm10-pilot-20261105) 提供源码快照、25份真实适配器/恢复工件及 SHA256SUMS，远端摘要已核验。

下面的 NVIDIA 结果与命令为笔记本历史，台式机结果按硬件和精度独立记录。

RTX 3060 Laptop 6GB已实际完成Qwen3-1.7B的前后向、LoRA更新、保存重载、128题/32步SFT，以及均匀/动态/静态各8步的seed13预实验；适配器权重保存在本项目`checkpoints/`。工程测试36项通过，源码快照、原始预测、梯度、token和成本账本保留在`runs/`。

首轮统一重载8道开发题：A5/8、B6/8、C5/8、D6/8、E5/8；各组耗时不同，不构成同预算或泛化提升证据。[首轮报告](reports/local_pilot_findings.md)明确保留零方差无更新、答案区域失败和截断等负面结果。

扩大开发验证中，原NF4/SDPA配置的A为22/32、B为19/32；其预算轮次C完成21步，D在8步后的验证中发生CUDA非法内存访问。NF4/eager完成4步G4 GRPO冒烟后另开轮次：A11/16，128题/32步B已完成，但B独立评测在8/16题后再次发生非法内存访问，没有完整B正确率，也未启动该轮C/D/E。当前无活动训练进程，建议把剩余三种子同预算对照、扩大训练池和独立评测迁移后完成；目标机仍需实查验证。详见[运行事件](reports/runtime_incidents.md)、[复现命令](docs/REPRODUCING_LOCAL.md)。没有同预算优势或正式测试提升结论。

已推送到[GitHub私有仓库](https://github.com/INeedSomeSugar/reasoning-rl-lab)；[预实验Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/local-pilot-20261105)提供源码迁移包、16份本机LoRA备份及SHA256，远端摘要已核对。台式机可克隆main获取最新交接说明，按[migration/README_ZH.md](migration/README_ZH.md)重建AMD环境。Release源码包保留首次推送前的本机快照，发布状态以main为准。

自动迁移入口为[从空目录拉取与核验的指令](migration/BOOTSTRAP_CODEX_ZH.md)，本轮已在 `D:\User\AI\reasoning-rl-lab` 执行并核验完成；历史源码包保留归档，没有覆盖 main。

## 实验链路

| 组别 | 起点与训练 | 主要用途 |
| --- | --- | --- |
| A | 固定 revision 的公开 Qwen3-1.7B | 原始公开模型基线；**已后训练，不是 Base** |
| B | A + completion-only LoRA SFT | 共同的 GRPO 起点 |
| C | B + GRPO，均匀有放回采样 | 主要对照 |
| D | B + GRPO，动态难度感知采样 | 主要方法 |
| E | B + GRPO，只在初始时探测难度 | 关闭动态更新的静态消融 |

C/D/E 共享训练池、SFT 权重哈希、损失、奖励、生成和优化配置，仅采样策略及其必要探测开销不同。动态采样属于已有课程学习/自适应采样思路的实现与检验，不宣称新算法。

## 当前使用方式

在本项目目录运行。所有依赖、缓存和运行数据留在本目录，其他现有项目保持原状。

台式机环境已装好，运行时先 dot-source `scripts/desktop_env.ps1`，仅向项目进程暴露 RX 9070 XT，避免 Trainer 自动使用集显。使用 `.venv\Scripts\python.exe` 与 `configs/desktop_rocm10_bf16_smoke.yaml`/`configs/desktop_rocm10_bf16_pilot.yaml`、`--hardware rocm`。下述 CUDA/NF4 安装命令仅适用于历史笔记本；不用于本机。已完成家族不可直接覆盖重跑；新尝试使用唯一的新运行名和独立配置，运行中的冻结预实验不要修改 src/configs。

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements\common.txt bitsandbytes==0.48.1
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
```

首次安装使用官方包源；如本地 PyPI TLS 不可用，可对普通 Python 依赖使用 `--index-url https://mirrors.aliyun.com/pypi/simple`。CUDA torch 仍来自官方 PyTorch。大文件断线可用 `python scripts/download_public.py torch` 下载并核对官方 SHA256，再通过 uv 安装该本地 wheel。

```powershell
$py=".venv\Scripts\python.exe"
& $py -m reasoning_rl prepare-data
& $py -m pytest -q
& $py -m reasoning_rl hardware
& $py -m reasoning_rl download-model --model qwen3_17b
# Hugging Face 大文件也有可续传的免费公开下载入口：
python scripts/download_public.py model qwen3_17b
& $py -m reasoning_rl check-model --name capability-17b --config configs/smoke.yaml
& $py -m reasoning_rl evaluate --name A-smoke --config configs/smoke.yaml
& $py -m reasoning_rl sft --name B-smoke --config configs/smoke.yaml
& $py -m reasoning_rl evaluate --name B-smoke-val --config configs/smoke.yaml --adapter checkpoints/B-smoke
& $py -m reasoning_rl grpo --name C-smoke-s13 --config configs/smoke.yaml --sft-checkpoint checkpoints/B-smoke --strategy uniform
& $py -m reasoning_rl grpo --name D-smoke-s13 --config configs/smoke.yaml --sft-checkpoint checkpoints/B-smoke --strategy dynamic
& $py -m reasoning_rl grpo --name E-smoke-s13 --config configs/smoke.yaml --sft-checkpoint checkpoints/B-smoke --strategy static
& $py -m reasoning_rl compare --runs C-smoke-s13 D-smoke-s13 E-smoke-s13 --name smoke-s13
& $py -m reasoning_rl report
```

运行名不可覆盖；重跑使用新名字。`check-model` 保存的适配器仅用于能力验证，不充当 B 的 SFT 检查点。`--model qwen3_06b` 是小模型备选，必须记录新的模型选择理由和独立结果，不能与 1.7B 混算。

## 数据与独立评测

GSM8K 来自官方 `openai/grade-school-math`，官方训练集按种子 **20261105** 分为训练 **6,961** 题和验证 **512** 题。按题目 word(1,2) TF-IDF cosine ≥ 0.90 构建近重复簇并整体划分，检查到 208 个近重复对、无归一化完全重复；当前阈值下没有跨 train/validation 的近重复对。这个检查不保证没有语义复述或预训练污染。

官方测试 **1,319** 题和官方 SVAMP **1,000** 题保存在 `data/holdout/`，不用于训练、难度估计、调参或选检查点。两个来源均保留 MIT 许可与精确 Git revision；详见 `data/manifest.json`。重复审计只读取题目进行一次卫生检查，不输出保留评测题内容，不使用评测模型表现。两套基准的预训练污染情况均未知。

开发阶段只允许验证集评测；最终阶段先冻结具体配置与检查点，再单次评测：

```powershell
& $py -m reasoning_rl freeze --config configs/main.yaml --checkpoints checkpoints/B-main checkpoints/C-main-s13 checkpoints/D-main-s13
# 将上一步返回的 JSON 路径填入；实际正式方案应含全部预注册种子和消融。
& $py -m reasoning_rl evaluate --name A-final-gsm --config configs/main.yaml --split test --frozen-plan experiments/frozen/<id>.json
& $py -m reasoning_rl evaluate --name D-final-svamp --config configs/main.yaml --adapter checkpoints/D-main-s13 --split external --frozen-plan experiments/frozen/<id>.json
```

`main.yaml` 已按2026-11-06本机实测改为开发容量配置，尚非最终正式协议；本轮不得使用上面的历史示意命令启封test/SVAMP。测试题及其派生结果不得回流训练。最终评测会在生成前写入单次使用凭据，意外中断也保留记录，不能重复抽取有利结果。

## 训练与成本约定

- SFT：`labels=-100` 屏蔽 prompt 和 padding；过长的标准解整条丢弃，记录 ID，不把截断答案当标签。
- GRPO：固定 TRL 0.23.1，显式 `loss_type=grpo`、`scale_rewards=group`、`beta=0`、token 重要性比率、单轮更新、截断 completion 不参与 loss。这是明确配置的简化 GRPO，不能当成对原论文配置的逐项复现。
- 奖励：仅最终数值答案正确性 0/1；分数、小数等按精确有理数比较。格式合规、提取失败和正确率分别统计。格式不产生额外奖励。
- 提示统一 `math_numeric_v2`，`enable_thinking=False`，评测固定 greedy；训练及难度探测使用同一采样参数。显式禁用评测时的模型默认值合并，避免 Transformers 将 greedy 覆盖成模型默认采样。
- 探测：每题 K 次，Jeffreys 先验平滑 `p=(s+0.5)/(K+1)`；难度分数 `4p(1-p)`，混合 25% 均匀质量。固定 optimizer step 间隔刷新完整训练池；静态组只做初始探测。
- 成本：`costs.jsonl` 细分训练输入、生成、探测和验证；端到端 wall time 是同机比较的主要预算轴，CUDA event 的 device elapsed 另列，显存为本进程 PyTorch 峰值。device elapsed 包含设备空闲区间，不等于活跃 kernel 时间或 FLOPs。
- smoke 是运行验证，不能凭相同步数或少量验证题声称成本公平与泛化提升。pilot/main 用共同端到端时间预算，探测和验证都计入，报告停止边界的超额。

## 文件入口

| 入口 | 内容 |
| --- | --- |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)、[HANDOFF.md](HANDOFF.md) | 研究边界、当前证据与下一步 |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | 实验、预算、公平比较与独立测试规则 |
| [docs/HARDWARE_AND_BUDGET.md](docs/HARDWARE_AND_BUDGET.md) | 本机实查、AMD 官方兼容与待确认云方案 |
| [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | 公式、代码入口、取舍和面试问题 |
| [docs/REPRODUCING_LOCAL.md](docs/REPRODUCING_LOCAL.md) | 本机实际轮次、从头复现与接续命令 |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | 课程学习、DAPO与本项目方法的关系 |
| `src/reasoning_rl/data.py`、`scoring.py` | 数据锁定、隔离与可验证评分 |
| `modeling.py`、`training.py`、`sampling.py` | loss mask、LoRA、TRL 接入与动态采样 |
| `evaluation.py`、`reporting.py` | 原始预测、指标、误差队列和对照图 |
| `models/`、`checkpoints/`、`runs/` | 公开模型、实际 LoRA 权重、原始运行记录 |
| [migration/README_ZH.md](migration/README_ZH.md) | 本机出现明确瓶颈后的台式机接力方式 |

不使用付费模型 API；云 GPU 方案等待用户确认。本项目暂不编写简历条目。
