# 实验报告：实际运行快照

本报告由保存的原始指标自动生成。smoke和少量验证题仅用于检查链路；尚未完成正式多种子独立测试的部分不作提升结论。

本机首轮研究发现与统一重载终点见 [local_pilot_findings.md](local_pilot_findings.md)，[终点和探测开销图](local_pilot_endpoints.png)。实际库精度变化见 [trl_precision_audit.json](trl_precision_audit.json)。

## 实际评测

|运行|划分/阶段|评测上下文|题数|正确|正确率|平均生成token|截断率|格式错误率|
|---|---|---|---:|---:|---:|---:|---:|---:|
|local17signalv2_20261105_C_reloadval_r2|validation|standalone_reload|8|5|62.5%|112.0|12.5%|87.5%|
|local17signalv2_20261105_C_s13|validation|in_training_diagnostic|8|6|75.0%|110.2|0.0%|75.0%|
|local17signalv2_20261105_D_reloadval|validation|standalone_reload|8|6|75.0%|110.8|0.0%|75.0%|
|local17signalv2_20261105_D_s13|validation|in_training_diagnostic|8|6|75.0%|110.4|0.0%|75.0%|
|local17signalv2_20261105_E_reloadval|validation|standalone_reload|8|5|62.5%|112.4|12.5%|87.5%|
|local17signalv2_20261105_E_s13|validation|in_training_diagnostic|8|6|75.0%|110.2|0.0%|75.0%|
|local17v2_20261105_smoke-A|validation|standalone_reload|8|5|62.5%|108.0|12.5%|75.0%|
|local17v2_20261105_smoke-B-val|validation|standalone_reload|8|6|75.0%|113.0|0.0%|75.0%|
|local17v4_20261105_C_reloadval|validation|standalone_reload|8|6|75.0%|113.0|0.0%|75.0%|
|local17v4_20261105_C_s13|validation|in_training_diagnostic|8|6|75.0%|119.9|12.5%|75.0%|
|localbudget_20261105-C-s13|validation|in_training_diagnostic|32|16|50.0%|150.6|28.1%|50.0%|
|localnf4eager_20261105_C_smoke|validation|in_training_diagnostic|8|5|62.5%|114.4|12.5%|87.5%|
|localnf4eagerpilot_20261105-A-val|validation|standalone_reload|16|11|68.8%|125.9|6.2%|68.8%|
|localpilot2_20261105_B_val|validation|standalone_reload|32|19|59.4%|140.3|25.0%|43.8%|
|localpilot3_20261105_A_val|validation|standalone_reload|32|22|68.8%|130.1|9.4%|75.0%|

## 实际训练

|运行|阶段/步数|LoRA delta L2|更新生效|检查点|
|---|---|---:|---|---|
|local17signalv2_20261105_C_s13|8|0.00578409|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17signalv2_20261105_C_s13`|
|local17signalv2_20261105_D_s13|8|0.00588737|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17signalv2_20261105_D_s13`|
|local17signalv2_20261105_E_s13|8|0.00758704|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17signalv2_20261105_E_s13`|
|local17v2_20261105_smoke-B|SFT|0.306684|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17v2_20261105_smoke-B`|
|local17v4_20261105_C_s13|4|0|False|`D:\User\AI\reasoning-rl-lab\checkpoints\local17v4_20261105_C_s13`|
|localbudget_20261105-C-s13|21|0.0317474|True|`D:\User\AI\reasoning-rl-lab\checkpoints\localbudget_20261105-C-s13`|
|localeager_20261105_B_smoke|SFT|0.307407|True|`D:\User\AI\reasoning-rl-lab\checkpoints\localeager_20261105_B_smoke`|
|localnf4eager_20261105_B_smoke|SFT|0.306825|True|`D:\User\AI\reasoning-rl-lab\checkpoints\localnf4eager_20261105_B_smoke`|
|localnf4eager_20261105_C_smoke|4|0.0111697|True|`D:\User\AI\reasoning-rl-lab\checkpoints\localnf4eager_20261105_C_smoke`|
|localnf4eagerpilot_20261105-B|SFT|2.54851|True|`D:\User\AI\reasoning-rl-lab\checkpoints\localnf4eagerpilot_20261105-B`|
|localpilot2_20261105_B|SFT|2.54994|True|`D:\User\AI\reasoning-rl-lab\checkpoints\localpilot2_20261105_B`|

## 未形成完成结果的运行

|运行|失败阶段|恢复检查点记录数|说明|
|---|---|---:|---|
|local17signal_20261105_C_s13|未完成|0|See saved logs; no completed result|
|local17signalv2_20261105_C_reloadval|未完成|0|See saved logs; no completed result|
|local17v3_20261105_C_s13|grpo_total|0|See saved logs; no completed result|
|localbudget_20261105-D-s13|validation,grpo_total|0|CUDA error: an illegal memory access was encountered|
|localeager_20261105_C_smoke|未完成|0|Long generation at step 4 showed no optimizer progress for over 300 seconds while total GPU memory was near capacity; interrupted owned session, not an observed OOM|
|localnf4eagerpilot_20261105-B-val|validation|0|CUDA error: an illegal memory access was encountered|
|localpilot2_20261105_A_val|未完成|0|See saved logs; no completed result|
|localpilot_20261105_B|未完成|0|See saved logs; no completed result|

## 解释边界与错误分析

当前数据划分、评分与训练配置见 README、PROTOCOL.md。逐题原始输出、自动诊断与待人工审阅队列位于每个评测目录。只有实际审阅的题才可归因为算术错误或条件遗漏。

训练loss下降、少量训练奖励变动及不同计算成本的smoke结果，都不等同于泛化提升。动态探测的全部开销保留在costs.jsonl和comparison图中；grpo_total包含子阶段，不能重复相加。

官方GSM8K测试与SVAMP保留集是否已启用，以experiments/final_claims和对应原始预测为准；缺少正式评测时不填猜测指标。预训练污染未知。

人工错误分析及失败归因补充在 [error_analysis.md](error_analysis.md)，当前完成边界见 [HANDOFF.md](../HANDOFF.md)。
