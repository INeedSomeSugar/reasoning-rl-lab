# 固定 S2 预筛后均匀采样：复现与工件入口

本轮是 `desktopfilter_20261110` 的三个配对种子开发实验，GSM8K test/SVAMP 不得启封。沿用的 confirmation128 已被旧家族使用，本轮明确记作开发数据。禁止将新池与旧44题训练结果直接比较以证明筛选收益。

## 先验证固定输入

项目位于 `D:\User\AI\reasoning-rl-lab`。复用 `.venv`、锁定依赖、`models/qwen3_17b` 与真实 `checkpoints/desktoprepair_20261106_152712-S2`。环境为 Python3.12.14、torch2.13.0+rocm10.0.0、Transformers4.57.1、PEFT0.17.1、TRL0.23.1；仅向项目进程暴露RX9070XT，非量化BF16/SDPA。无需重做SFT或重装环境。

```powershell
Set-Location D:\User\AI\reasoning-rl-lab
. .\scripts\desktop_env.ps1
$py = '.\.venv\Scripts\python.exe'
& $py -m pytest -q
& .\.cache\bootstrap\uv-bin\uv.exe pip check --python $py
```

主协议为 `experiments/desktopfilter_20261110/protocol_frozen.json`。调用下列只读验证会检查 `src+configs` 哈希、冻结执行文件、共同S2、开发ID、候选池与实际子集，且不会运行模型或读取保留评测集：

```powershell
& $py -c "import sys; sys.path.insert(0,'scripts'); from filter_common import read,EXP; from run_filter_controlled import verify; verify(read(EXP/'protocol_frozen.json')); print('Frozen execution inputs verified')"
```

模型11个文件逐项SHA256、S2真实前后向、Adam零梯度权重漂移和保存重载证据见 `experiments/desktopfilter_20261110/preflight.json`。共同S2 checkpoint SHA256为 `e8da01dae86ea2d7a95e19a645440510ec7dce2e4ceb546b26b3ff2dc9ac0bff`。诊断适配器不作为训练起点。

## 实际执行链路

以下是本轮实际执行的命令链，**不是覆盖已有目录的重跑指令**。脚本、run、log、冻结文件均拒绝覆盖；本轮已经完成，先检查状态和结果，不得重复启动同名家族。

```powershell
& $py scripts\prepare_filter_family.py
& $py scripts\audit_filter_signal.py
& $py scripts\filter_preflight.py
& $py scripts\run_filter_phase.py --label S2-development --timeout 900 -- -m reasoning_rl evaluate --name desktopfilter_20261110-S2-development --hardware rocm --config configs\desktopfilter_20261110_calibration.yaml --adapter checkpoints\desktoprepair_20261106_152712-S2
& $py scripts\calibrate_filter.py
& $py scripts\freeze_filter_capacity.py
& $py scripts\run_filter_phase.py --label probe --timeout 1100 -- scripts\run_filter_probe.py
& $py scripts\filter_environment_receipt.py
& $py scripts\freeze_filter_experiment.py
& $py scripts\run_filter_controlled.py
# 原控制器因交互中断失联后，保留原失败、同配置用新前缀恢复：
& $py analysis\resume_filter_after_interruption.py
# 全部六个独立重载结果完成后的离线复算、审计与可读图形：
& $py scripts\analyze_filter_experiment.py
& $py analysis\audit_filter_artifacts.py
& $py analysis\render_filter_results.py
# interpretation_final.json为核对全部结果后写下的结论，不是额外模型调用：
& $py analysis\write_filter_handoff.py
& $py analysis\finalize_filter_artifacts.py
```

`commands.jsonl` 保存真实参数向量，`process_costs.jsonl` 保存每个自有子进程启动/退出时间和失败。`session.json` 的四小时截止从预登记起计，比只限制模型计算更保守。早期一次 `-m` 分隔符命令行错误发生在模型启动前，后来使用显式 `--` 修正；沙箱Git DNS失败通过外部只读fetch恢复，uv不在PATH后使用项目内可执行文件/已安装依赖核验，均未重装依赖。

实际恢复事件：交互中断后原U/13仅留下136步日志且无终点，进程消失时间不能精确恢复。`interruption_receipt.json` 记录已观察运行下界及首次确认消失的保守上界；原日志、rollout和状态全部保留。`analysis/resume_filter_after_interruption.py` 调用原冻结控制器，仅将输出前缀改为 `desktopfilter_20261110-r1`、进程阶段名加 `r1-`，重新从同一S2执行全部六个终点。原配置、共同池、实际筛选、顺序和预算不变，总截止不延长。成功终点对照与包含失败的实际运行成本必须分列，不能隐藏已消耗但未留下终点的尝试。

## 冻结配置与计费

共同候选池64题、F保留16题；probe seed=20261110、K=4，原始成功次数0/1/2/3/4对应8/5/3/8/40题。资格只由原始次数1/2/3决定；全对/全错不能通过Jeffreys平滑被重新纳入。

每个终点的训练/筛选额度1200秒，独立重载评测预留570秒，外围及自然边界余量60秒，共同上限1830秒。完整固定探测物理执行一次，实测292.832841秒；每个F主比较均扣除全额，剩余约907.17秒用于自己的加载、缓存核验、rollout、优化和保存。U可将1200秒用于自己的训练链路。报告同时列出物理一次性成本与每F终点97.610947秒的摊销值；摊销值不对应一套额外训练结果，不能把节省额度假装已用于更多训练。

各种子的U/F仅由 `prefilter_manifest_path` 及组别标签区分，奖励、生成、损失和优化参数相同。G4、temperature0.8、top_p0.95、top_k20、最多512新token、LR5e-6、梯度累积4、token重要性比率、单轮更新、beta0、截断completion屏蔽沿用实测配置。两臂均为有放回均匀采样，不启用动态补采。

通用YAML仍保留 `uniform_mix`、`prior` 和缓存相关字段，但当前 `--strategy uniform` 与 `uniform_frozen_pool_v1` 不使用这些软加权/动态参数。实际选择日志应始终为U的1/64、F的1/16，sampler updates为0；Jeffreys平滑既不决定F成员，也不决定保留题之间的权重。配置的pool_size=64表示共同候选池，F的实际16题由冻结manifest及comparison_contract记录。

`run_filter_controlled.py` 先训练全部六个终点并写 `all_training_endpoints_fixed.json`，再逐一用新进程加载适配器，在固定开发128题上greedy评测。不会看成绩后追加训练，评测未用足的预留不回填；自然原子生成/优化边界的训练超额和共同上限未用足均实报。

## 在同一冻结子集上重复终点

若要验证运行可复现，使用**新的运行名**。以下例子重复seed13终点；seed29/47只改相应配置和新名字。F读取已冻结的原始探测文件并再次核验哈希与原始计数，仍扣除完整292.832841秒，不把缓存当免费。训练及评测进程的外围计时需参照父runner记录，不能只拿Trainer吞吐当端到端预算。

```powershell
& $py -m reasoning_rl grpo --name filter-repro-U-s13-01 --hardware rocm --config configs\desktopfilter_20261110_U_s13.yaml --strategy uniform --sft-checkpoint checkpoints\desktoprepair_20261106_152712-S2
& $py -m reasoning_rl grpo --name filter-repro-F-s13-01 --hardware rocm --config configs\desktopfilter_20261110_F_s13.yaml --strategy uniform --sft-checkpoint checkpoints\desktoprepair_20261106_152712-S2
& $py -m reasoning_rl evaluate --name filter-repro-U-s13-01-development --hardware rocm --config configs\desktopfilter_20261110_U_s13.yaml --adapter checkpoints\filter-repro-U-s13-01
& $py -m reasoning_rl evaluate --name filter-repro-F-s13-01-development --hardware rocm --config configs\desktopfilter_20261110_F_s13.yaml --adapter checkpoints\filter-repro-F-s13-01
```

更换机器或重新研究探测随机性时，应建立新的运行家族、独立定标和新冻结协议，重新测量探测成本。不要在本轮协议内重抽探测种子、按开发准确率重新筛题或覆盖原始探测。固定墙钟预算会受吞吐波动影响，故不承诺跨机器逐比特或相同步数复现。

## 工件解释

- `experiments/desktopfilter_20261110/`：预登记、ID、完整冻结协议、源码快照/哈希、实际筛选、进程账本、终点成本、全部终点固定凭据。
- `runs/desktopfilter_20261110-{U,F}-s{13,29,47}/`：原始reward/rollout、completion token ID、实际advantage与mask、每微步输入信号、梯度、Adam一阶矩、逐步权重差、选题和成本。
- `runs/desktopfilter_20261110-probe/`：全部256条随机探测输出、原始成功次数、token和成本。定标16题探测另列，不混成K8。
- `runs/desktopfilter_20261110-*-development/`：独立加载模型精度、适配器哈希、实际评测ID和128条原始预测。
- `checkpoints/desktopfilter_20261110-*/`：适配器与配置；训练终点额外保存Adam/RNG诊断状态。尚无自动精确恢复dataloader/sampler位置的实现，不承诺精确断点续训。
- `reports/comparisons/desktopfilter_20261110/`：离线复算汇总、逐终点CSV、三种子配对差、原始误差样例和图。

表中“有效优势样本”为原始非零advantage且实际loss mask非空的completion；rollout组、completion、optimizer step分别计算。零梯度不代表Adam状态为空或权重不变。训练随机成功率与开发greedy准确率分列；三种子以及开发数据的局限必须与结果同时报告。
