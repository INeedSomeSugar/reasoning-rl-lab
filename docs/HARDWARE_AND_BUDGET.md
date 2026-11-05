# 硬件、兼容组合与首轮预算

核查日期：2026-11-05。当前未租卡、未付费、未调用付费模型 API。

本文件下方“本机”与 NVIDIA 运行数字指迁移前笔记本。RX 9070 XT 台式机已完成迁移、BF16 短链路及三配对种子开发预实验；最新实查与官方组合边界单列在 [DESKTOP_SETUP_20261105.md](DESKTOP_SETUP_20261105.md)，不与历史成本混算。本轮 300 秒训练+探测及180秒验证预留在 C/D/E 前冻结，实际成本见 [开发报告](../reports/comparisons/desktoprocm10pilot_20261105/findings.md)。七组 GRPO 合计3108.52秒，独立重载评测另计1045.00秒；整轮含A/B共74分42秒。峰值 PyTorch allocated/reserved 为4.62/7.07 GiB，未发生本轮OOM；不代表官方认证或长期稳定性保证。

## 本机实查

- 当前运行主机是 Windows NT 10.0 build 26300；注册表产品名保留 Windows 10 Home China，DisplayVersion=26H2，因此保留原始字段，不凭旧产品名推断当前 Windows 品牌版本。
- `nvidia-smi`：RTX 3060 Laptop GPU，6144 MiB，初查空闲3652 MiB，驱动621.01，compute capability8.6。其他用户进程占用的显存不作释放。
- 独立 CPython3.12.14；uv0.12.10。初始 Python 中未安装 torch、transformers、PEFT、TRL。
- D盘初查约153.5GB可用；C盘约68GB可用。所有本项目下载和环境留在本目录。
- CIM 查询失败，WSL 查询返回 E_ACCESSDENIED。当前不把 WSL 或另一台设备算作可运行环境。
- 用户确认 RX 9070 XT 位于另一台台式机，当前没有该主机的远程执行通道。本机可完成的阶段优先在3060上实跑；出现明确瓶颈后再交付迁移包。

实际训练时的 GPU/RAM/版本快照由 `rrl hardware` 与每个 run 的 `environment.json` 保存；上面的空闲显存是初查快照，不能当作后续保证。

## NVIDIA 本机组合

候选并固定安装：Python3.12，PyTorch2.8.0+cu128，Transformers4.57.1，PEFT0.17.1，TRL0.23.1，Accelerate1.10.1，bitsandbytes0.48.1。torch来自[官方cu128索引](https://download.pytorch.org/whl/cu128/torch/)，普通Python依赖在PyPI TLS失败后使用阿里云镜像。完整实际解析版本写入 `environment/requirements-local.lock.txt`。

使用 NF4 双重量化、BF16计算、rank8 LoRA、SDPA、AdamW、gradient checkpointing，不依赖vLLM、FlashAttention或系统级CUDA Toolkit安装。量化只在已验证的NVIDIA路径启用。已真实完成1.7B的SFT与GRPO；扩大SFT的128题/32步用时63.68秒，峰值allocated约3.353GiB、reserved约4.149GiB。32道验证题的B耗时511.07秒，生成4491token；生成明显比SFT优化更耗时。完整正式对照的主要资源约束是生成时间与多次重复，不仅是模型权重能否放进显存。

NF4/SDPA原模型验证与后续动态组诊断验证分别出现CUDA非法内存访问；未观察到OOM，原因未定位。完整BF16/eager实际通过前后向、LoRA、重载和4步SFT，但G4 GRPO长生成在显存接近6GB时停滞，记录3步后中断。NF4/eager完成4步G4 GRPO及8题诊断验证，GRPO峰值allocated约1.697GiB、reserved约3.434GiB；另开扩大轮次又完成128题/32步SFT，但B评测8/16后再次非法内存访问。短链路通过不能证明长训练稳定，SDPA也不是已确认根因。当前建议把扩大对照迁移后重新验证；没有活动训练进程，未修改系统驱动或结束用户进程。详见[运行事件记录](../reports/runtime_incidents.md)。

## RX 9070 XT 的官方支持与边界

9070XT官方规格为16GB显存（[AMD产品页](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html)），相较本机6GB有更多模型、KV cache、激活和生成组余量；它不直接保证正确率或固定提速倍数。本项目建议先用1.7B把SFT、三种子C/D、静态消融和独立评测做充分，再把3–4B作为另一个需测显存/吞吐的规模实验。

本次读取并保存了AMD固定版本的[Radeon Linux矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)、[Windows矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)和[Linux PyTorch安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/native_linux/install-pytorch.html)。它们提供一个明确受支持的稳定组合，不代表它仍是最新发布版本。

| 平台 | AMD文档支持 | 本项目候选上层库 | 本项目实跑状态 |
| --- | --- | --- | --- |
| 原生Linux Radeon | RX9070XT/gfx1201；ROCm7.2.1；PyTorch2.9.1；按文档选择受支持Ubuntu/RHEL与内核 | Python3.12、Transformers4.57.1、PEFT0.17.1、TRL0.23.1 | 未在台式机验证 |
| 原生Windows11 | RX9070XT/gfx1201；PyTorch2.9系列；Python3.12；包含ROCm7.2.1组件 | 同上，必须使用AMD原生Windows wheel | 未在台式机验证 |
| WSL | 单独的GPU/驱动/OS矩阵；不能用原生Windows支持代替WSL支持 | 另行核实 | 当前访问被拒绝 |

AMD明确说明Windows PyTorch含ROCm组件，但完整ROCm栈尚未全部支持；不能由“PyTorch可用”推出所有加速库可用。其Radeon文档列出FP16等数据类型，台式机首轮选择FP16非量化，以实跑检查为准。

[PEFT0.17.1依赖源](https://github.com/huggingface/peft/blob/v0.17.1/setup.py)要求torch≥1.13、accelerate≥0.21；[TRL0.23.1](https://github.com/huggingface/trl/tree/v0.23.1)与已固定Transformers/Accelerate版本构成候选Python依赖组合。它们没有在所读AMD矩阵中获得逐项认证，依赖满足不等于完整后训练链路已验证。台式机必须依次验证forward、backward、LoRA更新、save/reload、SFT、GRPO。

本项目固定的[bitsandbytes0.48.1安装文档](https://huggingface.co/docs/bitsandbytes/v0.48.1/installation)与最新版支持范围可能不同；不会把当前CUDA NF4方案直接迁移到AMD。`--hardware rocm` 会切换为FP16、非量化，沿用原生PyTorch SDPA/AdamW。若将来采用新量化组件，另建环境并保存单独验证证据。

## 300元首轮云方案：待用户确认

按单卡、完整环境可控制的GPU实例计费，避免把Serverless推理费当训练卡价。报价仅作2026-11-05方案，租用前重新核对控制台库存、地区与附加费用。

| 选项 | 公开单价 | 规划时长（非实测承诺） | GPU费用估计 | 用途 |
| --- | --- | --- | --- | --- |
| AutoDL RTX4090/24GB | 官网索引公开￥1.88/时 | 12–40小时 | ￥22.56–75.20 | 国内优先询价；存储规则需在控制台核对 |
| Runpod RTX3090/24GB | $0.50/时 | 24–40小时 | ￥87.60–146.00 | 替代卡型，先用1小时pilot测吞吐 |
| Runpod RTX4090/24GB | $0.89/时 | 16–32小时 | ￥103.95–207.90 | 更快完成生成密集的GRPO与难度探测 |

来源：[AutoDL官网](https://www.autodl.com/)、[Runpod2026-10-28更新的Pods价格表](https://www.runpod.io/pricing)。Runpod其他产品页存在更低“from”价格，与当前列表不同，预算使用更保守的列表价，不预支低价库存。美元换算**仅采用7.30元/美元的预算假设，不是实时汇率或含税结算报价**。

存储：Runpod[官方计费文档](https://docs.runpod.io/pods/pricing)显示volume运行时$0.10/GB/月、停止时$0.20/GB/月；network volume低于1TB为$0.07/GB/月。以40GB volume、7天保留、32小时运行/136小时停止估算：`40*(0.10*32+0.20*136)/720*$7.30≈￥12.33`。统一预留**￥20存储上限**。AutoDL存储当前未得到稳定公开费率，不写成已核实报价，同样保留20元上限并在确认租用前给出控制台明细。

保守首轮总额分配：**GPU最多￥220 + 存储最多￥20 + 预留￥60 = ￥300**。先审批不超过1小时pilot，再按实际tokens/s、生成长度、探测开销及显存确定main，所有安装、下载与最终评测时间都算入租卡窗口。主配置是上限草案，不承诺跑完整数据集。

建议阶段：本机smoke → 租卡1小时pilot → 同机C/D各3种子、每次最多2小时 → 静态消融 → 冻结后的完整保留评测。预计时长还包含SFT、验证和文件回收，能否完成需由实测定标。若本机已能完成某阶段，就不为该阶段租卡。

停止条件：单次到达预设wall/token上限；预算预计将超过GPU220元/总额300元；连续非有限loss/梯度；OOM无法通过降低microbatch或序列预算恢复；首轮探测预计占单次预算超过25%；超过90%生成持续截断；3个检查点均为零方差且无LoRA更新。训练脚本只管停止训练，**不会自动停止云实例计费**，云控制台必须另设自动关机/到时提醒，由用户确认具体平台后落实。停机前导出权重、日志和哈希，下载验证后终止计费实例并清理收费存储。

这些是方案，不是购买动作。当前尚未触发租卡或费用。
