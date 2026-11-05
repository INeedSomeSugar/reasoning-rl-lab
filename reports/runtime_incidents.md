# 本机运行中断与后端验证

这些记录区分真实异常、资源观测和仍未定位的原因。原始运行从不覆盖，失败没有计入已完成对照。

## NF4 / SDPA 的间歇性 CUDA 异常

`localpilot2_20261105_A_val`在完成18题后发生CUDA非法内存访问；单题同步诊断和新名字的完整32题重跑成功。之后`localbudget_20261105-D-s13`完成8步GRPO，在末次验证完成7题后再次发生非法内存访问。后一次堆栈显示在PyTorch的`scaled_dot_product_attention`调用，但CUDA异步执行可能影响错误报告位置，因此**尚未证明具体根因，也没有OOM证据**。

旧预算轮次C完成21步、GRPO阶段1148.61秒，检查点已保存；D没有形成终点检查点，E及其余种子未启动。D的初始探测完成8题，step8刷新只完成7题，156.77秒部分探测开销保留但未被用于抽题。失败成本账本的device elapsed和显存读数受损时记null并保留wall time，未伪造设备计时。

原冻结计划为`experiments/local_budget_frozen_development.json`；中断说明为`experiments/local_budget_interruption.json`。不能用新源码在旧前缀中补入部分组，然后继续声称源码协议一致。

## 完整 BF16 / eager 的显存与长生成问题

`localeager_20261105_capability`真实通过前向、反向、LoRA更新、保存重载，logits最大差为0，加载峰值allocated约3.79GiB。`localeager_20261105_B_smoke`完成4步SFT。

`localeager_20261105_C_smoke`完成3步，出现非零且有限梯度，随后第4步长生成超过300秒没有优化步进展；当时显卡总占用5863MiB、空闲132MiB。只中断了本项目该会话，未结束其他用户进程。未观察到OOM异常；共享显存分页、系统驱动与具体算子行为未被进一步测定，所以不把显存接近满载直接写成已经证实的根因。这一轮无完成的GRPO结果或终点权重，不能按4步训练成功计数。

## NF4 / eager 的独立候选

`localnf4eager_20261105_capability`通过实际前后向和保存重载，logits最大差0。随后4步SFT及`localnf4eager_20261105_C_smoke`的4步G4 GRPO完成：LoRA delta约0.011170，组内零方差50%，非零优势completion为8个；8题训练内验证完成。GRPO阶段240.60秒，峰值allocated约1.697GiB、reserved约3.434GiB。以上是功能与少步训练证据，不是长期稳定性或泛化提升证据。

新代码在诊断验证前保存带optimizer step的恢复权重。该真实smoke的恢复adapter与最终adapter字节SHA256相同，见`diagnostics/recovery_checkpoint_real_check.json`；注入验证异常的工程回归也检查了权重和收据仍保留。恢复目录没有完整优化器状态，不支持精确续训。

## 新轮次与归因边界

新候选使用同一公开Qwen3-1.7B revision，重建本后端的A/B；训练池按单独记录的训练题标签疑义审阅排除1项，原始数据不变。新前缀`localnf4eagerpilot_20261105`与旧轮次分别统计。新模型输出若与SDPA轮次不同，也不能直接当成训练增益。

该新轮次A完成16题、11题正确；B真实完成128题/32步SFT并保存权重，但B独立验证完成8题后，在第9题再次CUDA非法内存访问。失败阶段182.95秒已保留，失败调用内部token未知，已记录token仅为完成调用的下界；GPU计时/峰值因上下文失效不可用。新进程开启CUDA_LAUNCH_BLOCKING=1后，第9题单题诊断成功，见`diagnostics/localnf4eagerpilot_20261105-B-sync-q9.json`。这没有复现根因，也不能据此确认长链路稳定。

按预计划停止该轮后续C/D/E；B评测未完成，因此验证预留尚未实测校准，最终开发冻结文件未生成。新轮次中断凭据为`experiments/local_nf4_eager_interruption.json`，B适配器hash为`cd2560ad9a4fbda89ef0ed8db5c076333c0abc81254113d1625c0c4961b27152`。后续扩大对照与最终独立评测建议转到台式机重新验证环境，当前没有活动GPU进程。

目前仅说明NF4/eager的短链路完成，不能由此断言SDPA就是故障源。不同attention实现和浮点计算可产生数值差异；参见[PyTorch2.8 SDPA说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)及[Transformers4.57.1 attention接口](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)。项目没有更新系统驱动或混装另一套全局依赖。

若NF4/eager仍复现非法内存访问，先保留恢复检查点、原始输出和成本，重新评估本机环境；完整精度、长生成和更大训练池优先考虑另一台16GB的9070XT，并在目标机重新验证AMD训练链路。云GPU仍需用户确认付费。
