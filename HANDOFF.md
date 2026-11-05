# 当前交接状态

更新：2026-11-05。只修改本项目，求职方向为大模型算法实习。笔记本阶段完整与失败证据保留；当前已在 RX 9070 XT 台式机执行接续。

台式机已从 main `c011b71` 克隆，三个 Release 附件通过 SHA256，16 份笔记本 LoRA/48 文件已恢复。项目内 Python 3.12.14、官方 torch 2.13.0+rocm10.0.0、固定上层依赖及锁定 Qwen3-1.7B 均准备完成。pip check 和本机 40 项工程测试通过；独显可寻址显存约 15.92 GiB。系统 OS/驱动高于官方验证组合，实际证据与矩阵边界分开记录，未修改系统驱动或使用付费服务。

本机 `desktoprocm10bf16_20261105` 已完成单卡、非量化 BF16/SDPA 的能力检查、4 步 SFT、各 4 步 C/D/E 和共同独立重载。能力重载 logits 最大差 0；SFT 更新生效；C 全部组内零方差、delta=0；D/E 有非零优势与有限非零梯度、delta≈0.014659。D 两次探测的实际权重相同（变化事件 0），E 只探测一次；不能据此声称动态更新产生优势。同 8 道验证题 A=8、B=C=D=E=7；这是负面/工程预实验，非正式对照。GRPO 阶段实际 C/D/E=36.59/67.60/51.85 秒，独立重载评测另计。完整报告：[台式机短链路](reports/comparisons/desktoprocm10bf16_20261105/findings.md)。

`desktoprocm10pilot_20261105` 扩大开发轮次已全部完成，状态为 `completed_development_pilot`，结束于 2026-11-05 23:50（UTC+8），没有活动训练。128 题/32 步 SFT，独立 GRPO 候选 32 题经 agent 标签与条件审阅后共同排除 4 题、保留 28 题；不是人工审阅或全训练集质量保证。共同 32 道验证题、512-token 上限，A/B 均在本机重新评测。C/D/E 前冻结共同 B 哈希、池、源码和 300 秒策略+探测额度，以及 180 秒诊断验证预留；480 秒是共同上限，实际成本并不完全相等。

|本轮组别|实际完成|独立重载的开发正确数|
|---|---|---|
|A 原公开模型|共同 32 题验证|28/32（87.5%）|
|B completion-only SFT|128 题、32 步、20.23 秒实测训练|16/32（50.0%）|
|C 均匀 GRPO|seed 13/29/47；51/35/44 步|19、19、18 /32；均值 58.33%|
|D 动态 GRPO|seed 13/29/47；8/16/8 步|19、16、19 /32；均值 56.25%|
|E 静态消融|仅 seed 13；32 步|17/32（53.125%）|

D−C 配对差均值为 −2.08 个百分点，无动态采样收益证据，所有训练终点仍低于 A。B 的 9/32 输出截断、10/32 解析失败，重复输出及可解析算错的审阅在 `reports/comparisons/desktoprocm10pilot_20261105/sft_degradation_review.json`；SFT 退化的具体根因未确认，不能以 loss 下降代替准确率。D seed 29 完成一次后续全池探测，28 个实际权重均变化；seed 13/47 的后续探测不完整，耗时照计但不更新分布。K=2 有采样噪声，权重变化不等于能力改善。

194 次 GRPO 优化前梯度检查均有限且冻结骨干无梯度，其中 134 次非零；七个 GRPO 终点均有真实 LoRA 更新，八份本轮 B/C/D/E 权重及验证前恢复工件均保留。GRPO 阶段合计 3108.52 秒，独立重载评测另计 1045.00 秒，七次训练加载另计 21.42 秒；整轮墙钟 74 分 42 秒，包含 A/B 与进程准备等。训练峰值 allocated 4.62 GiB、reserved 7.07 GiB，均为 PyTorch 进程统计，不是整个系统显存占用。探测和训练内验证已含在 GRPO 总时间，不重复相加。66 个历史及新增运行证据审计通过，无缺失检查点、哈希或原始奖励/token/梯度对账错误。

报告：[扩大轮次结果](reports/comparisons/desktoprocm10pilot_20261105/findings.md)、[逐终点与成本 CSV](reports/comparisons/desktoprocm10pilot_20261105/standalone_endpoints.csv)、[准确率与成本图](reports/comparisons/desktoprocm10pilot_20261105/standalone_comparison.png)。共同 B：`checkpoints/desktoprocm10pilot_20261105-B`，SHA256 `b187c090389df7d4fbd23c4e2e6e29301ea511da2455b1f9601c2e2a696872e2`；C/D/E 的完整哈希在各 `runs/desktoprocm10pilot_20261105-*/training_result.json`。冻结开发源码哈希 `b11b646a22ec91b54b261527078bc2cfb1a2f431b303cfde27a9211b52ad8810`。所有加载和终点评测均记录真实 BF16、非量化、SDPA，不与笔记本 NF4 结果合并。

证据等级：实现与冒烟已验证；本机 SFT、GRPO 与三配对种子开发对照已真实完成；**正式 main/最终独立测试尚未完成**。当前训练池仅 28 题，开发验证仅 32 题且被重复使用，静态仅一次；不能写成正式泛化提升。下一轮优先在新的运行家族中排查 SFT 的标签/EOS/优化强度与输出重复，再按实测探测成本确定各 D 种子能完成多次全池刷新的 main 配置；不改动本轮冻结配置或覆盖负面结果。最终协议、具体检查点与评测预算冻结后才运行 GSM8K test/SVAMP；本次两个保留集均未评测。云 GPU 和付费 API 仍未使用，本机电费未计量。

保留两次 FP16 失败：首次 B 在误启用独显+集显 DataParallel 后原生异常；单卡 B 在首次优化前检测到非有限 LoRA 梯度。两者均没有成功 SFT 终点。`scripts/desktop_env.ps1` 仅向项目进程暴露独显；另建明确 BF16 配置后短链路及扩大开发链路通过，未将这些异常解释为已确认驱动故障。下载、实查、官方支持与失败日志入口见 [台式机接续记录](docs/DESKTOP_SETUP_20261105.md)。历史源码 ZIP 未覆盖 main，历史 NF4 权重未作为本机 B。本轮所有改动和新工件仅在本地，未推送。

## 已完成的笔记本历史证据

|阶段|证据等级|实际结果|
|---|---|---|
|固定数据、隔离、评分器|已实现并验证|GSM8K train6961/val512/test1319；SVAMP1000；seed20261105；208近重复对、0跨train/val残留|
|独立CUDA环境与模型|已真实验证|Python3.12.14、torch2.8.0+cu128、PEFT0.17.1、TRL0.23.1；Qwen3-1.7B官方revision及两块权重SHA256核对|
|前后向、LoRA、保存重载|已冒烟验证|1,605,632个可训练参数；非零有限梯度；冻结层无梯度；重载logits最大差0|
|首轮SFT B|已完成真实训练|4步，754处理token/380监督token；池32题；LoRA delta约0.306684|
|首轮C/D/E|已完成真实训练及单种子预实验|各8步、G4、16题池、seed13；均有真实LoRA更新；D探测2次、后续权重变化1次；E仅初始探测|
|共同重载终点评测|已完成开发验证|同8题：A5/8、B6/8、C5/8、D6/8、E5/8；各组耗时不同，不能声称同预算优势|
|扩大SFT|已完成真实训练|localpilot2_20261105_B：128题、32步、1 epoch；28381处理token/15160监督token，delta约2.54994|
|扩大A/B验证|已完成开发验证|同32题、256-token上限：A22/32，B19/32；B截断8/32，不能声称SFT带来收益|
|时间预算C/D/E|已中断，证据保留|C完成21步、1148.61秒并保存权重；D完成8步后验证第8题CUDA非法内存访问，未保存终点；E尚未启动。见experiments/local_budget_interruption.json|
|完整BF16/eager候选|能力与SFT通过，GRPO未完成|4步SFT真实完成；GRPO记录3步后长生成停滞，显存接近6GB，主动中断；未观察到OOM，原因未定位|
|NF4/eager候选|已冒烟验证及真实短训练|前后向、LoRA、重载通过；4步SFT与4步G4 GRPO完成；验证前恢复权重与最终适配器字节一致|
|NF4/eager新开发轮次|B训练完成，评测中断|A为11/16；128题/32步B已保存，B评测完成8/16后CUDA非法内存访问；无完整B正确率，C/D/E未启动|
|正式三种子、最终独立测试|未完成|GSM8K test和SVAMP尚无模型评测；不编造迁移或泛化指标|
|GitHub|已推送私有仓库并发布迁移附件|[仓库](https://github.com/INeedSomeSugar/reasoning-rl-lab)；[预发布Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/local-pilot-20261105)含源码包、16份LoRA备份、SHA256，已核对远端摘要|

完整结果以training_result.json和完成的metrics.json为准。总报告：[LOCAL_RESULTS.md](reports/LOCAL_RESULTS.md)；首轮报告：[local_pilot_findings.md](reports/local_pilot_findings.md)；成本图：[local_pilot_endpoints.png](reports/local_pilot_endpoints.png)。最近一次完整工程测试36 passed，见environment/verification.json；包含恢复检查点与训练标签排除的回归检查。工程通过不等于长GPU链路稳定。

## 检查点与日志位置

- 共同首轮SFT：checkpoints/local17v2_20261105_smoke-B，hash `08bac92bda6e0dec47cd4de6c53caa9c225171a00d18db3ae05e9940abbcfd73`。
- 首轮GRPO：checkpoints/local17signalv2_20261105_C_s13、_D_s13、_E_s13；对应runs保留源码快照、rollouts、selections、probes、costs、真实训练参数。
- 扩大SFT：checkpoints/localpilot2_20261105_B，hash `068fdf285e8f01b40012a5660260ca762868de4d4b1a0711fb80e320fe9bf8f8`。
- 新NF4/eager扩大SFT：checkpoints/localnf4eagerpilot_20261105-B，hash `cd2560ad9a4fbda89ef0ed8db5c076333c0abc81254113d1625c0c4961b27152`。训练完整，验证未完整。
- 模型：models/qwen3_17b；已后训练，官方revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。0.6B尚未下载或评估。
- 日志：logs保存终端输出；runs/*/costs.jsonl为结构化成本与梯度日志。只有adapter权重与trainer state JSON，未保存可精确恢复优化器的完整状态。

## 保留的失败及解释

1. v1提示占位符被照抄、Transformers默认生成配置覆盖greedy：该A轮次已明确invalidated，不作基线。
2. v1评分器不接受答案另起一行：只在验证集校准为numeric_final_v2，原始预测与旧指标保留，无保留集调参。
3. 首次GRPO Float/BF16生成报错：显式autocast修复。首次G2的4步GRPO零方差100%、delta0，作为无学习负面诊断保留。
4. TRL会再次把embedding/head从FP32变为BF16，已用真实模型测量。训练内验证标为诊断，终点均用共同独立重载；见reports/trl_precision_audit.json。
5. 首轮D的探测占总GRPO时间约70.6%；K2刷新前梯度为0，权重变化可能来自采样噪声。首轮负奖励主要是最终答案区域失败和截断，不能证明推理提升。
6. 扩大SFT第一次在新增精度元数据采集处失败，尚未训练：PEFT包装层与bitsandbytes层同名但属性不同；已修复属性检测，新名字重跑，不覆盖旧记录。
7. 一次独立C验证仅写出1题后进程结束，原因未知；保留残留并用新run在验证集重跑，不归因为OOM。
8. 扩大A验证完成18题后CUDA非法内存访问；新进程单题同步诊断成功，完整重跑localpilot3_20261105_A_val得到22/32。原始异常根因未定位。Ledger清理不再遮盖原始CUDA异常，失败成本仍保留，新增工程回归验证通过。

## 笔记本继续顺序与历史约束

原NF4/SDPA开发计划固定在experiments/local_budget_frozen_development.json，源码哈希d90de43622ff1792fc630b9e4b16670343143f69ae99dc1e5e0193a7ebb43b6b；该轮在D验证阶段再次CUDA非法内存访问，父runner已失败退出，不能继续混入新源码。D在step8的后续探测只完成7/8题，156.77秒已计入但未用于抽题。完整BF16/eager尝试也保留为未完成运行。上述异常不能证明SDPA是根因，详见[运行事件记录](reports/runtime_incidents.md)。

当前改用已经完成短链路验证的NF4/eager，另建configs/local_nf4_eager_pilot.yaml与localnf4eagerpilot_20261105前缀。计划见experiments/local_nf4_eager_development_plan.json：A/B共同16题、256-token上限；128题SFT不变。训练题gsm8k-train-05134存在百分比条件与参考解不符，原始数据和旧结果保留，新轮次预先排除此题，候选8题实际保留7题。C/D/E共享600秒优化及探测预算；验证预留按B实测耗时向上取整30秒后再加30秒，在任何C/D/E开始前冻结。先13，再按计划补29/47；不得混入旧后端结果。正式测试前仍需冻结具体方案和检查点；开发冻结不授权保留集评测。

该尝试在B的第9题再次失败，按预计划停止后续，**预算未校准、冻结文件未生成**。同步单题诊断成功也不能证明长链路修复。新中断凭据为experiments/local_nf4_eager_interruption.json。当前无活动训练进程；建议现在迁移扩大训练、三种子同预算对照与最终独立测试，在目标机重新建立A/B及共同SFT。没有将失败B的部分预测拼成完整准确率。不同后端均失败，根因仍未知，不能承诺换卡必然解决。

以上“本机”异常和“当前无活动训练进程”是笔记本交接时的历史状态。现在已在台式机本地访问 RX 9070 XT 并完成独立 ROCm 短链路和扩大三配对种子开发轮次，详见本文顶部；不能把 NVIDIA 异常归因于 AMD，也不能把更换显卡当作正确率提升。源码迁移包与单独的16份笔记本LoRA备份在dist；清单和SHA256在migration及environment。历史成果已推送到私有仓库 https://github.com/INeedSomeSugar/reasoning-rl-lab；16份LoRA二进制通过Release附件提供，没有放入Git历史。本轮台式机改动和新工件尚未推送。台式机接力提示词见migration/CODEX_PROMPT_ZH.md。云GPU预算300元仍待确认，没有租卡或付费API调用。
