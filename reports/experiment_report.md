# 实验报告：实际运行快照

本报告由保存的原始指标自动生成。smoke和少量验证题仅用于检查链路；尚未完成正式多种子独立测试的部分不作提升结论。

## 实际评测

|运行|划分/阶段|题数|正确|正确率|平均生成token|截断率|格式错误率|
|---|---|---:|---:|---:|---:|---:|---:|
|local17signalv2_20261105_C_reloadval_r2|validation|8|5|62.5%|112.0|12.5%|87.5%|
|local17signalv2_20261105_C_s13|validation|8|6|75.0%|110.2|0.0%|75.0%|
|local17signalv2_20261105_D_reloadval|validation|8|6|75.0%|110.8|0.0%|75.0%|
|local17signalv2_20261105_D_s13|validation|8|6|75.0%|110.4|0.0%|75.0%|
|local17signalv2_20261105_E_reloadval|validation|8|5|62.5%|112.4|12.5%|87.5%|
|local17signalv2_20261105_E_s13|validation|8|6|75.0%|110.2|0.0%|75.0%|
|local17v2_20261105_smoke-A|validation|8|5|62.5%|108.0|12.5%|75.0%|
|local17v2_20261105_smoke-B-val|validation|8|6|75.0%|113.0|0.0%|75.0%|
|local17v4_20261105_C_reloadval|validation|8|6|75.0%|113.0|0.0%|75.0%|
|local17v4_20261105_C_s13|validation|8|6|75.0%|119.9|12.5%|75.0%|

## 实际训练

|运行|阶段/步数|LoRA delta L2|更新生效|检查点|
|---|---|---:|---|---|
|local17signalv2_20261105_C_s13|8|0.00578409|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17signalv2_20261105_C_s13`|
|local17signalv2_20261105_D_s13|8|0.00588737|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17signalv2_20261105_D_s13`|
|local17signalv2_20261105_E_s13|8|0.00758704|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17signalv2_20261105_E_s13`|
|local17v2_20261105_smoke-B|SFT|0.306684|True|`D:\User\AI\reasoning-rl-lab\checkpoints\local17v2_20261105_smoke-B`|
|local17v4_20261105_C_s13|4|0|False|`D:\User\AI\reasoning-rl-lab\checkpoints\local17v4_20261105_C_s13`|

## 解释边界与错误分析

当前数据划分、评分与训练配置见 README、PROTOCOL.md。逐题原始输出、自动诊断与待人工审阅队列位于每个评测目录。只有实际审阅的题才可归因为算术错误或条件遗漏。

训练loss下降、少量训练奖励变动及不同计算成本的smoke结果，都不等同于泛化提升。动态探测的全部开销保留在costs.jsonl和comparison图中；grpo_total包含子阶段，不能重复相加。

官方GSM8K测试与SVAMP保留集是否已启用，以experiments/final_claims和对应原始预测为准；缺少正式评测时不填猜测指标。预训练污染未知。

人工错误分析及失败归因补充在 [error_analysis.md](error_analysis.md)，当前完成边界见 [HANDOFF.md](../HANDOFF.md)。
