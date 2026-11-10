# 项目上下文

<!-- REPAIR_20261106_PUBLICATION_BEGIN -->
## GitHub 发布与当前可见性（2026-11-10 更新）

本轮代码、配置、报告及49份运行证据已推送main，源码提交为 [d4eec11](https://github.com/INeedSomeSugar/reasoning-rl-lab/commit/d4eec11d675e079e7462f601a7ff98487801c18a)；[开发预发布](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106) 已发布。4个附件的远端大小与SHA256全部匹配：[源码/证据ZIP](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-repair-cache-dev-20261106)、17份适配器ZIP、SHA256SUMS.txt、archive_receipt.json。2026-11-10 已核验仓库为公开；2026-11-06 发布收据中的私有字段保留当时状态。

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

## 历史上下文（2026-11-05 轮次与迁移原文保留）

更新日期：2026-11-06。工作区仅 `D:\User\AI\reasoning-rl-lab`；不改动其他项目。方向为**大模型算法相关实习**，优先展示训练、算法、受控消融、误差分析和可复现结果。

问题：同一计算预算下，按当前模型训练题成功率进行动态采样是否优于均匀采样？不声称算法新颖性。第一版仅做 GSM8K 数学推理后训练，外部迁移集为 SVAMP。

历史笔记本：Windows NT build 26300、RTX 3060 Laptop 6GB、独立 Python 3.12；其运行与失败证据原样保留，不结束其他用户进程。用户现已授权在 RX 9070 XT 台式机自动下载、配置并接续，仍只操作本项目。

当前台式机：Windows 11 build 26300.9818/26H2、Ryzen 7 9700X、24 GiB 物理 RAM、RX 9070 XT 15.92 GiB，设备状态正常，驱动 32.0.32015.12012。main `c011b71` 已克隆，Release 三件通过 SHA256、16 份 LoRA/48 文件已恢复并核验。项目内 Python 3.12.14/.venv、AMD 官方 torch 2.13.0+rocm10.0.0/gfx1201 与锁定上层依赖已安装，锁定公共模型已核验。本机 ROCm 初始化、40 项工程测试与模型前后向/LoRA/重载通过。FP16 的多设备原生异常和单卡非有限梯度均保留；当前使用进程级单独暴露 9070 XT 的 BF16/非量化/SDPA。官方矩阵与实际 OS/驱动并非逐项相同，不能称为完整官方认证；详见 [台式机记录](docs/DESKTOP_SETUP_20261105.md)。

本机短链路已完成；随后 `desktoprocm10pilot_20261105` 已完成 128 题/32 步 SFT、28 题训练池的 C/D 各 seed 13/29/47、E 仅 seed 13，以及全部独立重载终点评测。共同 32 道开发题、512-token 上限：A 28/32、B 16/32；C 19/19/18、D 19/16/19、E 17（均以32为分母）。C/D 均值 58.33%/56.25%，D−C 配对差均值 −2.08 个百分点，无动态优势证据，全部仍低于 A。D seed 29 完成真实全池刷新且 28 个采样权重变化；另两个 D 种子的后续探测不完整，未用于更新。194 次 GRPO 梯度检查均有限、冻结骨干无梯度，七个终点有 LoRA 更新；66 个历史及新增运行证据审计通过。

本轮已经结束，没有活动训练；当前为已完成三配对种子开发预实验，非正式 main/独立测试。SFT 退化与输出重复/截断需要在新家族排查；具体根因未确认。当前训练池和验证题都很小，K=2 含噪声，静态仅一次，不能声称泛化提升。测试与 SVAMP 继续封存；最终方案、检查点与评测预算尚未冻结。本轮 wall time 74分42秒，训练峰值 allocated/reserved 4.62/7.07 GiB。完整结果、成本、失败和下一步见 [HANDOFF.md](HANDOFF.md) 与 [台式机开发报告](reports/comparisons/desktoprocm10pilot_20261105/findings.md)。台式机代码、报告和原始日志已推送main；[台式机预发布Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/desktop-rocm10-pilot-20261105)含源码快照、25份适配器/恢复工件及SHA256SUMS，远端大小和摘要均已核验；发布收据为environment/desktop_publication_20261106.json。

采用 Qwen3-1.7B（公开已后训练版本，非 Base），锁定官方完整 revision；备选 Qwen3-0.6B。笔记本历史使用 NVIDIA NF4 + BF16 + rank-8 LoRA。台式机首轮 FP16 失败后，独立使用官方 ROCm PyTorch + BF16 + 非量化 + SDPA，LoRA rank 8；两种硬件/精度的结果分开记录。

数据：GSM8K 官方 train→6961/512，seed=20261105，question-only TF-IDF 近重复阈值0.90、组件整体划分；正式 test=1319 与 SVAMP=1000 保留。两个基准的预训练污染未知。当前审计发现208个近重复对、0个完全重复、0个跨开发划分残留。数据及模型的源 revision、内容哈希随项目保留。

状态边界请以 [HANDOFF.md](HANDOFF.md) 和实际 `runs/` 为准：实现不等于冒烟；真实少步训练不等于正式对照；验证集表现不等于独立测试结果。云 GPU 未租用，300元为待确认首轮上限。预算方案、源码和学习说明已经纳入本项目；不生成应用界面，不修改简历。

笔记本历史：已实跑首轮A→4步B→8步C/D/E及共同重载验证，36项工程测试通过，未见同预算收益证据。扩大128题/32步SFT已完成，同32题验证为A22/32、B19/32。原NF4/SDPA时间预算轮次的C完成21步，D完成8步后再次CUDA非法内存访问，中断证据保留，E及29/47未启动。已增加验证前保存恢复权重；完整BF16/eager通过能力检查和4步SFT，但GRPO只记录3步，后续长生成停滞并中断。NF4/eager完成能力检查、4步SFT及4步G4 GRPO；新开发轮次localnf4eagerpilot_20261105的A为11/16，B完成128题/32步，但B独立验证8/16后再次CUDA非法内存访问，C/D/E未启动、预算未校准冻结。单题同步诊断成功不等于长链路稳定。当前无活动训练进程，建议剩余扩大训练、三种子同预算对照及最终独立测试转台式机重新验证后执行；该段保留笔记本迁移前状态，台式机当前证据见上文与 HANDOFF。训练题疑义审阅、原始预测、曝光和探测成本均保存，详见reports/LOCAL_RESULTS.md与HANDOFF.md。已按授权推送到私有仓库https://github.com/INeedSomeSugar/reasoning-rl-lab，源码与16份LoRA备份已上传预发布Release：https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/local-pilot-20261105，远端SHA256已核对。环境与公开基础模型不上传；台式机克隆main后按migration说明接续。
