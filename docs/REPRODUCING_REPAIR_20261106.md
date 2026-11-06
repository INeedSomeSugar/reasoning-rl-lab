# 复现 2026-11-06 SFT 排查与缓存采样开发实验

范围：`desktoprepair_20261106_152712`；仅开发集，不授权 GSM8K test 或 SVAMP。先阅读 [完整报告](../reports/comparisons/desktoprepair_20261106_152712/findings.md) 和 [冻结协议](../experiments/desktoprepair_20261106_152712/sampling_frozen.json)。历史运行名不可覆盖。

## 环境与来源

工作目录为 `D:\User\AI\reasoning-rl-lab`。复用本目录 `.venv`、`.cache`、`models/qwen3_17b` 及已准备数据。原台式机安装方法见 [环境记录](DESKTOP_SETUP_20261105.md)，不要在本机套用历史 NVIDIA/NF4 命令。

```powershell
Set-Location -LiteralPath 'D:\User\AI\reasoning-rl-lab'
. scripts/desktop_env.ps1
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$py='.venv/Scripts/python.exe'
& $py -c 'import torch; assert torch.version.hip and torch.cuda.device_count()==1 and "9070 XT" in torch.cuda.get_device_name(0); assert torch.cuda.is_bf16_supported(); print(torch.__version__,torch.cuda.get_device_name(0))'
& .cache/bootstrap/uv-bin/uv.exe pip check --python $py
& $py -m pytest -q
```

公开模型锁定 Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，已后训练、非 Base。全部模型/tokenizer 文件的本轮 SHA256 复核见 `experiments/desktoprepair_20261106_152712/public_model_verification.json`，对照此前官方 LFS/Git 核验收据。没有把本地重新计算摘要误称为本轮重新向厂商下载认证。

数据 manifest、train/validation 文件内容、固定两组128题 ID 和源码摘要均由运行保存；`load_split` 每次核对实际文件。模型和数据的源 revision 在 `configs/*.lock.json`。检查点 ZIP 只打包适配器的三份必要文件，tokenizer 继续使用已核验的公开基础模型目录。

## 使用新名字重跑所选真实 SFT

下面的 `$ReplayPrefix` 必须使用未存在的名字；原家族的配置、ID 和 raw artifacts 留作比较。S2 的名义池128题、实际8步×4个微批次=32题，实际3688个监督token，不能写成已经监督了128题。

```powershell
$ReplayPrefix='desktoprepair_replay_20261107_01'
$Original='desktoprepair_20261106_152712'
& $py -m reasoning_rl sft --hardware rocm --name "$ReplayPrefix-S2" --config "configs/${Original}_S2.yaml"
& $py -m reasoning_rl evaluate --hardware rocm --name "$ReplayPrefix-S2-comparison" --config "configs/${Original}_eval.yaml" --adapter "checkpoints/$ReplayPrefix-S2"
& $py -m reasoning_rl evaluate --hardware rocm --name "$ReplayPrefix-A-comparison" --config "configs/${Original}_eval.yaml"
```

要逐一复现四候选，分别使用 `${Original}_S1.yaml` 至 `_S4.yaml` 和新的运行名。原始候选门槛与选择规则在 `sft_preregistration.json`，不得重新解释规则来选择更有利候选。完整实际命令收据为 `experiments/desktoprepair_20261106_152712/commands.jsonl`。

S1 与历史 B 的真实 adapter 权重 SHA256 完全相同，证明了本机这一条训练链路的精确复现；不据此保证所有跨设备/驱动/时间预算运行都逐字节相同。

## 复现冻结的三种子对照

如果需要完全相同的真实起点，先从本轮适配器归档恢复原 S2，并核对 checkpoint SHA256 `e8da01dae86ea2d7a95e19a645440510ec7dce2e4ceb546b26b3ff2dc9ac0bff`。也可使用刚重跑的 S2，但要先核对其实际哈希并作为新复现家族的共同起点记录。

```powershell
$CommonStart="checkpoints/$Original-S2"
$Design=@(
  @{seed=13;arm='C';strategy='uniform'},
  @{seed=13;arm='D';strategy='dynamic'},
  @{seed=13;arm='E';strategy='static'},
  @{seed=29;arm='E';strategy='static'},
  @{seed=29;arm='D';strategy='dynamic'},
  @{seed=29;arm='C';strategy='uniform'},
  @{seed=47;arm='D';strategy='dynamic'},
  @{seed=47;arm='C';strategy='uniform'},
  @{seed=47;arm='E';strategy='static'}
)
foreach ($Item in $Design) {
  $RunName="$ReplayPrefix-$($Item.arm)-s$($Item.seed)"
  $Cfg="configs/${Original}_grpo_s$($Item.seed).yaml"
  & $py -m reasoning_rl grpo --hardware rocm --name $RunName --config $Cfg --sft-checkpoint $CommonStart --strategy $Item.strategy
  if ($LASTEXITCODE -ne 0) { throw "Preserve failed run $RunName" }
  & $py -m reasoning_rl evaluate --hardware rocm --name "$RunName-comparison" --config $Cfg --adapter "checkpoints/$RunName"
  if ($LASTEXITCODE -ne 0) { throw "Preserve failed endpoint $RunName" }
}
```

`strategy=dynamic/static` 在这些配置里分别表示 `rollout_cache_fifo_v1_dynamic` 与初始探测后固定的静态消融；不代表历史周期全池刷新。C 不生成探测；D/E 各自真实生成全池K4并支付费用。不要跨臂复用一份探测文件却给两臂记虚构费用。逐步采样、缓存与初始权重可由审计脚本重放。

每组算法钟600秒，独立评测预留540秒、进程外围预留30秒，总上限1170秒。依赖导入在外部进程成本中，算法钟从源码中的 `arm_started` 开始，包含初始化元数据核验、run快照、模型加载、Trainer准备、探测及训练；自然边界保存可能略超600秒。所有实际耗时必须以新 run 的账本重新计算。不同输出长度、设备吞吐与自然停止边界会改变完成步数，不能保证墙钟早停下完全相同的终点权重。

原家族的 `run_repair_after_sft.ps1`、SFT/定标/受控 runner 是本次带固定六小时截止时间的原始执行收据，会拒绝已有状态或已结束的截止时间。它们不是可直接覆盖重启的通用启动器。新名字的上述命令用于复现训练链路；开展新研究阶段须复制固定输入并重新登记自己的总预算与协议，保留原记录。

## 阶段确认和证据复算

原确认在九个checkpoint全部固定后只开启一次，开启前清单为 `confirmation_opened.json`。它是已开放 validation 的互斥128题，不是最终独立测试。后来复现同一确定性终点评测只检查工程复现，不能重新称作未见数据或用于再次挑选参数。

```powershell
# 对一个已经固定的新终点做同口径工程复现；原家族不会覆盖重跑。
& $py -m reasoning_rl evaluate --hardware rocm --name "$ReplayPrefix-D-s13-confirmation" --config "configs/${Original}_confirm.yaml" --adapter "checkpoints/$ReplayPrefix-D-s13"

# 以下只复算原家族保存的证据，不再运行模型、不读取保留题内容。
& $py analysis/repair_sft_forensics.py
& $py analysis/audit_repair_study.py
& $py analysis/summarize_repair_study.py
```

审计包括源码快照、模型设备、监督 token、保存权重改变量、奖励重评分与精确选题/缓存重放。汇总保留逐种子结果、实际耗时、未用足与超额、完整/不完整探测、真实分布有效池和覆盖；图形只使用实际观测点。配对题目 bootstrap 以这三个固定种子为条件，不是种子总体或独立泛化区间。

`configs/main.yaml` 已从未测量的2048题/NF4草案改为本机实测容量配置，stage 为 `main_capacity_calibrated_development_only`，`formal_test_protocol_ready=false`。它不是最终保留集授权。不要执行旧 README 中示意性的正式 freeze/test 命令来启封本轮保留集。

所有负面结果、未训练控制、历史失败、原始日志和权重分别保留。适配器加 trainer state JSON 不能精确恢复优化器；从头复现和独立终点评测与精确续训是不同能力。
