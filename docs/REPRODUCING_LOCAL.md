# 复查和复现本机实验

本文对应 Windows + RTX3060 Laptop 6GB。精确版本见 `environment/requirements-local.lock.txt`；先按 README 建立独立 `.venv`。命令均在项目根目录执行，运行名必须唯一。GPU 命令顺序执行，等待前一个进程退出。

## 复查已保存证据

```powershell
$py='.venv\Scripts\python.exe'
& $py -m reasoning_rl prepare-data
& $py scripts/audit_evidence.py
& $py scripts/analyze_sampling.py --prefix local17signalv2_20261105 --runs local17signalv2_20261105_C_s13 local17signalv2_20261105_D_s13 local17signalv2_20261105_E_s13
```

已有数据时，`prepare-data` 核对内容哈希和协议，保持划分，不再次解读保留集题目答案。审计核对源码快照、训练池、token、奖励、零方差、梯度和检查点。GitHub 源码仓库不包含模型及 LoRA 二进制；缺少权重时会标为 `pass_with_unavailable_checkpoints`，不能称为完成权重核验。笔记本真实权重在 `checkpoints/`。

## 当前本机路径：NF4/eager

原 `local_pilot.yaml` 使用 NF4/SDPA，发生过两次 CUDA 非法内存访问，原预算轮次中断。完整 BF16/eager 能完成能力检查和 SFT，但较长 G4 生成停滞。随后 NF4/eager 完成真实短 SFT、G4 GRPO 和诊断验证，另开 `localnf4eagerpilot_20261105` 开发轮次。事件和证据边界见 [runtime_incidents.md](../reports/runtime_incidents.md)。

下列命令从公开模型重做相同设计，使用新前缀 `recheck`。首先复制 YAML 与预计划，在计划中将 `prefix` 改为 `recheck`；保持问题 ID 选择规则、排除题、模型、生成和优化配置不变。改配置会改变 `code_sha256`，必须保留新的冻结与快照。

```powershell
$py='.venv\Scripts\python.exe'
$cfg='configs/recheck_nf4_eager.yaml'
# 先复制 configs/local_nf4_eager_pilot.yaml 到 $cfg。
# 复制 experiments/local_nf4_eager_development_plan.json 为 experiments/recheck_plan.json，改 prefix/config 路径。
& $py -m reasoning_rl download-model --model qwen3_17b
& $py -m reasoning_rl check-model --name recheck-capability --config configs/local_nf4_eager_smoke.yaml
& $py -m reasoning_rl evaluate --name recheck-A-val --config $cfg
& $py -m reasoning_rl sft --name recheck-B --config $cfg
& $py -m reasoning_rl evaluate --name recheck-B-val --config $cfg --adapter checkpoints/recheck-B
& $py scripts/freeze_local_development.py --prefix recheck --config $cfg --development-plan experiments/recheck_plan.json --baseline recheck-A-val --sft-evaluation recheck-B-val --sft-checkpoint checkpoints/recheck-B --output experiments/recheck_frozen.json
& .\scripts\run_local_budget_pilot.ps1 -Prefix recheck -Config $cfg -FrozenPlan experiments/recheck_frozen.json -SftCheckpoint checkpoints/recheck-B -OneSeed
# seed13完成且稳定、至少一组有真实参数更新，再补齐预定29、47。
& .\scripts\run_local_budget_pilot.ps1 -Prefix recheck -Config $cfg -FrozenPlan experiments/recheck_frozen.json -SftCheckpoint checkpoints/recheck-B
& $py scripts/summarize_budget_pilot.py --prefix recheck --baseline recheck-A-val --sft-evaluation recheck-B-val
& $py scripts/analyze_sampling.py --prefix recheck
& $py -m reasoning_rl report
```

预计划规定：相同前16道开发题、greedy、thinking=False、生成上限256；128题/32步 SFT；候选8题排除已审计的疑义标签 `gsm8k-train-05134`，实际池7题，原始数据不改。C/D/E共享600秒优化及探测预算；验证预留为 `30*ceil(B验证秒数/30)+30`，总上限为600加预留。冻结脚本只在 A/B 完成、C/D/E 尚未启动时校准；没有读取测试/迁移集表现。每组自然停止边界的实际耗时仍需报告，不能称为精确等计算量。

原始 NF4/eager 轮次的 B 评测已中断（8/16），没有生成 `local_nf4_eager_frozen_development.json`，C/D/E 未启动。它不能用 runner 原地继续；保留失败，以新前缀重新验证环境和完整 A/B。该 YAML 的验证预留仍是初始候选值，不能写成已经按 B 实测冻结。

runner 的旧默认参数指向已经中断的 NF4/SDPA 轮次，当前必须显式传参。它核对源码、配置、数据与共同 SFT 权重，跳过有真实完成证据的阶段。失败运行不能覆盖；改源码或后端需另开前缀。支持按阶段接续，不支持任意训练步的优化器恢复。验证前保存的恢复适配器可保留训练成果，但 `trainer_state.json` 与适配器不等于完整优化器恢复包。

## 历史轮次与可移植性

每个 `runs/<name>/source_snapshot/` 包含当时的源码、配置与依赖记录。历史 NF4/SDPA 与新 NF4/eager 的模型、训练池或评测规模不同，分别报告，不能拼成一次受控实验。训练题疑义审计见 `reports/training_label_audit.json`；旧奖励和参考答案保持原样。

台式机从 [CODEX_PROMPT_ZH.md](../migration/CODEX_PROMPT_ZH.md) 开始实查 AMD 环境，重新建立共同 SFT 起点。不能在只有历史指标、没有权重的 Git 克隆中直接使用本机接续命令，并声称重新训练。

## 结果入口

- `training_result.json`：完成步数、参数 delta、奖励、零方差、token 和检查点哈希。
- `costs.jsonl`：加载、训练/探测/验证、device elapsed 和显存；嵌套阶段不可重复相加。
- `recovery_checkpoints/`：验证前保存的适配器凭据，是否完成验证单列记录。
- `runs/<name>-reloadval/validation/predictions.jsonl`：共同独立重载后的原始预测。
- `reports/comparisons/<prefix>/standalone_comparison.json`：核对协议后的逐种子终点。
- `sampling_diagnostics.json`、`item_exposure.csv`、`probe_uncertainty.csv`：真实采样曝光、探测不确定性和开销。

训练内验证使用 TRL 准备后的模型，仅作诊断。比较终点共同独立重载。当前开发冻结不授权 GSM8K 官方测试或 SVAMP 评测；最终测试另冻具体方案和检查点。所有负面结果保留。
