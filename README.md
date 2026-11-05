# ReasoningRL Lab

**小模型推理后训练与难度感知数据选择。** 面向大模型算法实习的独立研究工程项目。

研究问题：在**同一硬件、相同端到端计算预算**下，用模型当前解题成功率指导训练题采样，能否比均匀随机采样更有效地改善数学推理？主要交付是训练代码、检查点、原始运行证据和受控实验，先看[本机结果与证据边界](reports/LOCAL_RESULTS.md)，逐运行指标见[自动报告](reports/experiment_report.md)。

## 已有真实结果与当前实验

RTX 3060 Laptop 6GB已实际完成Qwen3-1.7B的前后向、LoRA更新、保存重载、128题/32步SFT，以及均匀/动态/静态各8步的seed13预实验；适配器权重保存在本项目`checkpoints/`。工程测试36项通过，源码快照、原始预测、梯度、token和成本账本保留在`runs/`。

首轮统一重载8道开发题：A5/8、B6/8、C5/8、D6/8、E5/8；各组耗时不同，不构成同预算或泛化提升证据。[首轮报告](reports/local_pilot_findings.md)明确保留零方差无更新、答案区域失败和截断等负面结果。

扩大开发验证中，原NF4/SDPA配置的A为22/32、B为19/32；其预算轮次C完成21步，D在8步后的验证中发生CUDA非法内存访问。NF4/eager完成4步G4 GRPO冒烟后另开轮次：A11/16，128题/32步B已完成，但B独立评测在8/16题后再次发生非法内存访问，没有完整B正确率，也未启动该轮C/D/E。当前无活动训练进程，建议把剩余三种子同预算对照、扩大训练池和独立评测迁移后完成；目标机仍需实查验证。详见[运行事件](reports/runtime_incidents.md)、[复现命令](docs/REPRODUCING_LOCAL.md)。没有同预算优势或正式测试提升结论。

GitHub同步已获用户授权，默认私有仓库。本机CLI登录已失效，用户选择稍后登录，当前仅准备本地Git版本；不把“已初始化Git”写成“已推送GitHub”。后续可通过GitHub下载到台式机，迁移时重建AMD环境。

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

`main.yaml` 目前是容量上限草案，正式冻结前需要 pilot 定标。测试题及其派生结果不得回流训练。最终评测会在生成前写入单次使用凭据，意外中断也保留记录，不能重复抽取有利结果。

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
