# 本地交接与提交状态

全部模型任务、离线复算和交接文档已完成。本轮未提交、未推送、未发布Release；旧提交与负面工件保留。

主要新代码为固定原始K4筛选、完整探测扣费、实际loss-mask/advantage对账和逐优化步参数改变量记录。原中断U/13有136步且无终点，主结果来自r1新前缀的六个完成终点；原失败和恢复过程均保留。

正式GSM8K test/SVAMP仍封存。不要把优化器诊断状态称为已支持精确断点续训。

下列是收尾时Git状态（文件内容摘要见artifact_manifest.json）：

```text
 M HANDOFF.md
 M PROJECT_CONTEXT.md
 M README.md
 M src/reasoning_rl/training.py
?? analysis/audit_filter_artifacts.py
?? analysis/collect_filter_training_signal.py
?? analysis/filter_live_status.py
?? analysis/finalize_filter_artifacts.py
?? analysis/render_filter_results.py
?? analysis/resume_filter_after_interruption.py
?? analysis/write_filter_handoff.py
?? configs/desktopfilter_20261110_F_s13.yaml
?? configs/desktopfilter_20261110_F_s29.yaml
?? configs/desktopfilter_20261110_F_s47.yaml
?? configs/desktopfilter_20261110_U_s13.yaml
?? configs/desktopfilter_20261110_U_s29.yaml
?? configs/desktopfilter_20261110_U_s47.yaml
?? configs/desktopfilter_20261110_calibration.yaml
?? configs/desktopfilter_20261110_calibration_train.yaml
?? docs/REPRODUCING_FILTER_20261110.md
?? environment/desktopfilter_20261110-hardware.json
?? environment/desktopfilter_20261110-interrupted-status.json
?? environment/desktopfilter_20261110-pytest.txt
?? environment/desktopfilter_20261110-r1-status.json
?? environment/desktopfilter_20261110-status.json
?? environment/desktopfilter_20261110-verification.json
?? experiments/desktopfilter_20261110/
?? logs/desktopfilter_20261110-S2-development.log
?? logs/desktopfilter_20261110-U-s13.log
?? logs/desktopfilter_20261110-calibration-probe.log
?? logs/desktopfilter_20261110-calibration-train.log
?? logs/desktopfilter_20261110-probe.log
?? logs/desktopfilter_20261110-r1-F-s13-development.log
?? logs/desktopfilter_20261110-r1-F-s13.log
?? logs/desktopfilter_20261110-r1-F-s29-development.log
?? logs/desktopfilter_20261110-r1-F-s29.log
?? logs/desktopfilter_20261110-r1-F-s47-development.log
?? logs/desktopfilter_20261110-r1-F-s47.log
?? logs/desktopfilter_20261110-r1-U-s13-development.log
?? logs/desktopfilter_20261110-r1-U-s13.log
?? logs/desktopfilter_20261110-r1-U-s29-development.log
?? logs/desktopfilter_20261110-r1-U-s29.log
?? logs/desktopfilter_20261110-r1-U-s47-development.log
?? logs/desktopfilter_20261110-r1-U-s47.log
?? reports/comparisons/desktopfilter_20261110/
?? runs/desktopfilter_20261110-S2-development/
?? runs/desktopfilter_20261110-U-s13/
?? runs/desktopfilter_20261110-calibration-probe/
?? runs/desktopfilter_20261110-calibration-train/
?? runs/desktopfilter_20261110-preflight/
?? runs/desktopfilter_20261110-probe/
?? runs/desktopfilter_20261110-r1-F-s13-development/
?? runs/desktopfilter_20261110-r1-F-s13/
?? runs/desktopfilter_20261110-r1-F-s29-development/
?? runs/desktopfilter_20261110-r1-F-s29/
?? runs/desktopfilter_20261110-r1-F-s47-development/
?? runs/desktopfilter_20261110-r1-F-s47/
?? runs/desktopfilter_20261110-r1-U-s13-development/
?? runs/desktopfilter_20261110-r1-U-s13/
?? runs/desktopfilter_20261110-r1-U-s29-development/
?? runs/desktopfilter_20261110-r1-U-s29/
?? runs/desktopfilter_20261110-r1-U-s47-development/
?? runs/desktopfilter_20261110-r1-U-s47/
?? scripts/analyze_filter_experiment.py
?? scripts/audit_filter_signal.py
?? scripts/calibrate_filter.py
?? scripts/filter_common.py
?? scripts/filter_environment_receipt.py
?? scripts/filter_preflight.py
?? scripts/freeze_filter_capacity.py
?? scripts/freeze_filter_experiment.py
?? scripts/prepare_filter_family.py
?? scripts/run_filter_controlled.py
?? scripts/run_filter_phase.py
?? scripts/run_filter_probe.py
?? src/reasoning_rl/prefiltering.py
?? tests/test_prefiltering.py
```
