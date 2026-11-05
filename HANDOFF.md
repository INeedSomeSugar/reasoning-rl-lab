# 当前交接状态

更新：2026-11-05。只修改本项目，求职方向为大模型算法实习。当前继续本机运行，台式机尚未执行任何项目命令。

## 已完成的证据

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
|时间预算C/D/E|正在继续本机运行|configs/local_pilot.yaml，先seed13；共同上限1200秒，预留验证540秒，GRPO池8题，与SFT池不相交|
|正式三种子、最终独立测试|未完成|GSM8K test和SVAMP尚无模型评测；不编造迁移或泛化指标|
|GitHub|已授权，尚未推送|本地独立Git仓库已初始化；CLI登录失效，用户选择稍后登录，先继续实验|

完整结果以training_result.json和完成的metrics.json为准。首轮报告：[local_pilot_findings.md](reports/local_pilot_findings.md)；成本图：[local_pilot_endpoints.png](reports/local_pilot_endpoints.png)。最近一次完整工程测试32 passed，见environment/verification.json；证据审计17个run通过，后续新增运行完成后重审。

## 检查点与日志位置

- 共同首轮SFT：checkpoints/local17v2_20261105_smoke-B，hash `08bac92bda6e0dec47cd4de6c53caa9c225171a00d18db3ae05e9940abbcfd73`。
- 首轮GRPO：checkpoints/local17signalv2_20261105_C_s13、_D_s13、_E_s13；对应runs保留源码快照、rollouts、selections、probes、costs、真实训练参数。
- 扩大SFT：checkpoints/localpilot2_20261105_B，hash `068fdf285e8f01b40012a5660260ca762868de4d4b1a0711fb80e320fe9bf8f8`。
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

## 继续顺序与约束

扩大A/B验证已完成；依据B的511秒耗时，将验证预留校准为540秒、GRPO池缩为8题。开发计划固定在experiments/local_budget_frozen_development.json，源码哈希d90de43622ff1792fc630b9e4b16670343143f69ae99dc1e5e0193a7ebb43b6b。当前顺序运行seed13 C/D/E并统一重载，状态见environment/localbudget_20261105-status.json；此期间不要修改src或configs。观察稳定性、奖励、零方差、输出、选择分布和成本后，按同一配置推进C/D的seed29、47。正式测试前还需冻结具体方案和检查点；当前开发冻结不授权保留集评测。

9070XT位于另一台台式机，无当前访问通道。其16GB显存适合尝试更充分的1.7B实验，实际吞吐、PEFT/TRL与FP16链路需在那里另验。更换显卡不会自动提高正确率。当前用户要求仍是先本机运行；后续通过GitHub接力，必要时再用迁移脚本。云GPU预算300元待确认，没有租卡或付费API调用。
