# 更大干净候选池的固定预筛与随机子集复验

> 2026-11-13实际状态：训练35/35、完整评估22/37；RX9070XT设备不在场，当前诊断停止。5格重试已耗尽，10格仍有额度；禁止直接按下面历史阶段命令重跑。先读[阻塞报告](../reports/comparisons/desktopfilterrep_2026-11-10_235416/gpu_runtime_blocked_status.md)和`AGENT_WORK_STATE.md`最新覆盖说明。

唯一家族：`desktopfilterrep_2026-11-10_235416`，北京时间2026-11-10登记。继续时先读对应`runner_state.json`、`heartbeat.json`及PID收据，避免重复启动；不重跑已经完成的家族或同名attempt。仅在本项目内复用`.venv`、BF16/SDPA基础模型和真实S2，不重做SFT，不打开正式test/SVAMP。

## 已执行的准备与定标

```powershell
Set-Location D:\User\AI\reasoning-rl-lab
. .\scripts\desktop_env.ps1
$py = '.\.venv\Scripts\python.exe'
$family = 'desktopfilterrep_2026-11-10_235416'
# 以下为已发生的阶段，不是授权覆盖旧目录的重跑命令。
& $py scripts/prepare_large_filter.py --family $family --stage register
& $py scripts/prepare_large_filter.py --family $family --stage audit
& $py scripts/large_filter_data.py --family $family --stage calibrate_setup
& $py scripts/large_filter_data.py --family $family --stage review_pack
& $py scripts/run_large_filter.py --family $family --stage calibration
# Agent读原题/原解，review_batch_01..09逐条记录计算或疑义理由。
# candidate_review_completed.jsonl是固定顺序中的前256个通过题及之前的排除题。
& $py scripts/large_filter_data.py --family $family --stage freeze
& $py scripts/run_large_filter.py --family $family --stage probes
```

历史账本核验四个已发布源码／证据归档的完整SHA256，并逐文件核对checkout；缺失或不同版本恢复到本家族`history_restored`，不覆盖当前文件。`history_sources.jsonl`给出逐ID证据来源和来源版本SHA256，`history_exposure_ledger.jsonl`区分实际预测/训练记录、名义池和保守历史提及。原validation512整体排除；不声称只扫描当前runs即可证明全历史未曝光。

新开发集位于`data/fresh_development/<family>/`，是原train中后置隔离的开发子集，不能称正式测试或公开模型预训练未见。普通数据加载器会排除其保留ID；只有带全部终点锁和具体对象key的专用评估入口可读。原始manifest与旧split不变。

## 冻结和完整矩阵入口

三次完整探测及原始输出写完后，才能运行预算冻结。`freeze_large_filter.py`按预登记公式从实测P_j、旧开发计时和实际N确定B/E/O/C、一次失败重试的全额预留和全会话硬截止。它同时冻结35配置、F/R全部ID、共同S2、源码快照与哈希。U只有五个独立终点，跨三个选择块复用；不得额外训练15个U。

为提高37对象完整评估的可靠性，主训练和新开发评估前另登记`evaluation_reserve_amendment_before_main.json`：E采用旧64题外推耗时的2倍加300秒，再向上取整300秒，替代初始登记的1.35倍加120秒。原登记保留；这项修订只扩大评估余量，不改变B、训练变量或任何选择规则，也不回填训练。冻结前还通过连接的GitHub只读核对默认分支最新提交，仍为用户给定的4ea43b03，未发现远端更晚家族。

```powershell
& $py scripts/freeze_large_filter.py --family $family
& $py scripts/run_large_filter.py --family $family --stage controlled
```

控制器只串行运行一个本项目GPU模型任务，以隐藏窗口子进程落日志。每单元最多两个完整attempt，均从同一S2和同一配置开始，采用首次成功终点，保留失败输出与成本。Adam/RNG诊断文件不代表支持精确sampler/dataloader续训。尚未提供重启自动恢复接口；异常时先核对旧子进程是否仍在运行，不能直接重开整个阶段或覆盖PID/日志收据。

`all_training_endpoints_fixed.json`在35单元全部成功或耗尽一次重试后写入，登记每个成功权重的完整哈希和缺失单元。之后才开启`fresh_development_opened.json`，再按预定顺序由37个独立进程加载A/S2与35个训练对象。第一次开启新开发后没有任何追加训练路径。失败终点／评估不能静默删除，未完成时不能给完整矩阵结论。

F每个终点扣相应完整P_j；R也扣相同P_j，以配平可训练时间。R该扣额是实验配平项，实际随机抽样成本另记；不能推断没有实际运行的“R免扣探测并多训练”效果。共享三次探测物理开销、逐终点算法扣额、R配平及包含失败的会话实际成本分开汇总。

## 工程检查与离线复算

```powershell
& $py -m pytest -q
```

新增回归覆盖新数据隔离、冻结门禁、完整探测扣费、R从全池同规模抽样、U复用和联合簇重采样依赖、失败日志不可覆盖。报告、原始预测、逐终点CSV、盲化审阅、图表和离线复算将在全部模型任务结束后填入同一家族。远端发布需后续明确指令。

## 全部模型结束后的盲化Agent审阅

先离线审计全部证据，按原登记范围生成隐藏方法名的案例包，再准备精确文本去重的逐响应审阅单元：

```powershell
& $py analysis/analyze_large_filter.py --family $family
& $py analysis/large_filter_blind_units.py --family $family --mode prepare
```

Agent逐份阅读`experiments/$family/blind_review_unique_units.jsonl`，在`blind_unit_review_batch_*.jsonl`写明每个`review_unit_id`的`category`、具体`notes`、`reviewer_kind=Agent`、`human_reviewed=false`和`method_identity_hidden_during_review=true`，必要时记`question_or_gold_issue`。不提前人工读取crosswalk或密封身份表。题目ID、题干、金标、参考解、响应原文和截断状态必须全部精确相同才合并展示；不合并近似文本，不删任何原始case或alias。脚本不生成语义错误类别，所有唯一响应必须先有明确的Agent判断。

```powershell
& $py analysis/large_filter_blind_units.py --family $family --mode materialize
& $py analysis/finalize_large_filter_reviews.py --family $family
```

`materialize`缺少任何唯一响应判断即拒绝回填；`finalize`只有全部原始案例齐全、判断哈希固定后才读方法身份表。重复展示数与实际唯一响应审阅数分别报告；不称作人工审阅，不修改主评分器或金标。随后才渲染图表、完成实际视觉核验、撰写中文报告并执行最终工件审计。
