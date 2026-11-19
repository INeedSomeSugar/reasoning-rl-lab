# 驱动重装检查及保存后暂停（2026-11-13）

用户在恢复运行器启动后要求先不启动训练，并明确选择当前独立评估保存后暂停。当前只有F-j20261112-s29-a1评估在运行，无训练；下一GPU任务已由不同所有者锁拦住。当前35/35训练、22/37完整评估、5格重试耗尽，暂停尚待当前评估保存后核验。

## 实际恢复检查

- 用户清理并重新安装驱动；11:05重启后，同一RX9070XT设备实例恢复`Present=true, Problem=0, Status=OK`。
- 独显驱动从`32.0.32015.12012`改为`32.0.32015.16001`，对应[AMD Adrenalin26.11.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-11-1.html)。这是用户实施的环境恢复，后续评估按新驱动分段登记；不能写成所有评估处在完全相同驱动版本下。
- 项目Torch仍为`2.13.0+rocm10.0.0`、HIP`7.15.26333`，只暴露同一RX9070XT；原非量化BF16/SDPA、模型、S2、tokenizer、评分器和数据保持原SHA。
- 小张量BF16/SDPA前后向、有限非零梯度及BF16→FP32转换通过，用时5.444秒；没有优化器更新或保存模型。
- 原已使用开发64题以固定S2和原greedy配置完整重放，用时254.831秒。64/64回答原文及判对标记相同，预测文件SHA256完全一致：`b4568567aaae183a0483db53f12b9d1e56f0f320a530005dd67b587def19052f`。旧64题52/64；该重复检查不构成新的泛化证据，也不能证明整个新开发集在两个驱动下逐位一致。
- 全部35终点和S2适配器、基础模型/tokenizer、冻结源码/配置、数据与协议重新核验通过。恢复检查不使用fresh_development或GSM8K test/SVAMP。

重装后的只读系统事件快照仅见Display4107信息事件（显示模式枚举），截至该快照未见新的Display/WHEA错误。此前驱动故障前后存在看门狗、原生0xC0000409退出、HIP invalid image，以及显卡设备不在场；这些证据尚不能区分具体驱动缺陷、系统兼容、PCIe/供电或硬件原因。通过一次短检查不能证明长时间运行稳定。

## 运行与暂停的实际边界

11:32:50按此前仍生效的继续指令启动评估恢复运行器；11:33:05开始F-j20261112-s29-a1。用户随后要求检查完先不启动训练；未启动任何训练，并立即拦住后续GPU调度。用户明确选择“等当前评估保存后暂停”。11:34:32观察器设置下一任务启动锁，当前评估保留完整原进程成本及原评分，不截停、不拼接部分输出。

不得启动后续GPU任务；当前项保存后由观察器终止控制器。用户明确继续前，不启动任何训练或后续评估。

仍保留的5个耗尽对象：F-j20261111-s47, F-j20261113-s29, R-j20261113-s29, R-j20261112-s29, U-s29。冻结每对象最多两次，未增加第三次尝试；这些缺失格继续显式保留，不能声称完整37对象或完整预登记5×3比较。新的驱动分段、失败及准备成本也要计入最终限制说明。原B/E/O/C与硬截止不变。

当前阶段：`armed`；完整评估22/37。主统计和原定seed29盲化审阅尚未完成，不用部分结果选模型或换seed。GSM8K test/SVAMP继续封存，远端未发布。

## 证据与复现

- [恢复检查收据](../../../experiments/desktopfilterrep_2026-11-10_235416/driver_recovery_readiness_2026-11-13_032618_UTC/completed.json)及其中两项GPU检查的原始日志和物理成本。
- [暂停请求](../../../experiments/desktopfilterrep_2026-11-10_235416/driver_recovery_evaluation_pause_armed.json)；完成后使用`pause_after_driver_recovery_evaluation_completed.json`，历史Nov12暂停收据不覆盖。
- [全部评估状态](../../../experiments/desktopfilterrep_2026-11-10_235416/evaluation_units.json)、[全部训练前锁定权重](../../../experiments/desktopfilterrep_2026-11-10_235416/all_training_endpoints_fixed.json)、[先前故障审计](gpu_runtime_blocked_status.md)。
- `analysis/check_large_filter_driver_recovery.py`仅用于此次已执行检查；`analysis/resume_large_filter_after_gpu_recovery.py`导入未改的冻结控制器，增加运行前设备核验及新失败后停止的外围调度保护；`analysis/pause_large_filter_after_driver_check.py`仅用于当前保存边界暂停。所有源码快照和SHA保存在各收据目录，不将这些历史命令作为覆盖重跑入口。

运维可见性限制仍保留：此前显示S2聚合正确数及seed47部分ID/判对进度行；未读取seed29回答正文或盲化身份映射，冻结金标/评分器/阈值不改。记录时间2026-11-13T03:37:14.059791+00:00。
