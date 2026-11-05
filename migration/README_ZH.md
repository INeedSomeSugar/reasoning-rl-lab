# RX 9070 XT 台式机迁移

希望由台式机Codex自动完成下载和合并时，使用[自动拉取与接续指令](BOOTSTRAP_CODEX_ZH.md)。固定安装位置为 `D:\User\AI\reasoning-rl-lab`；Codex会克隆main、下载Release材料、核验哈希并恢复LoRA，然后按既有提示词检查AMD环境。源码ZIP作为历史归档保留，不覆盖当前Git工作树。

本机已完成短SFT/GRPO和采样预实验，较长生成在不同后端反复CUDA非法内存访问，当前建议迁移剩余扩大训练、三种子同预算对照与独立测试。它不是完全不能训练，也没有证据保证换卡自动解决所有问题。已推送到[GitHub私有仓库](https://github.com/INeedSomeSugar/reasoning-rl-lab)，[预发布Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/local-pilot-20261105)提供下述两个包与SHA256，远端摘要已核对。推荐克隆main获取最新交接文档，离线包保留本机首次发布前的快照。

复制并解压 `dist/reasoning-rl-lab-desktop-transfer-20261105.zip` 到台式机独立目录，打开该目录，将[CODEX_PROMPT_ZH.md](CODEX_PROMPT_ZH.md)整段交给Codex。源码包包含固定数据、依赖约束、配置、原始文本预测/日志、报告和学习材料。先运行 `python scripts/verify_transfer.py` 核对SHA256，再允许改变目标机配置。

另有 `dist/reasoning-rl-lab-laptop-lora-20261105.zip`（约83.3MiB），包含16份真实适配器的权重、配置和元数据；范围包括工程、SFT、GRPO与恢复检查点，不是16个正式实验终点。不包含公开基础模型或完整优化器状态。可解压到同一父目录合并 `reasoning-rl-lab/checkpoints/`，然后运行 `python scripts/verify_checkpoints.py`；也可不解压先用 `python scripts/verify_checkpoints.py --archive <zip路径>`。实际48个文件已核对SHA256，见 `environment/checkpoint_archive_verification.json`。这是本机权重备份，不是AMD运行结果。

GitHub方式：在台式机登录有仓库权限的账号，执行 `git clone https://github.com/INeedSomeSugar/reasoning-rl-lab.git`，打开项目目录并粘贴同一提示词。源码、固定数据、许可、指标和原始文本证据随Git版本保存；虚拟环境和公开基础模型在目标机器重新建立。需要复查笔记本权重时，从[Release](https://github.com/INeedSomeSugar/reasoning-rl-lab/releases/tag/local-pilot-20261105)下载LoRA包并解压到克隆目录的父目录，合并项目内checkpoints；建立项目环境后运行检查点核验。Git克隆以Git提交为准；TRANSFER_MANIFEST只核验Release源码ZIP解压后的原始快照。

不需要远程桌面、账号共享或跨机器访问。模型按锁定revision在台式机重新下载；也可手动复制笔记本已完成的 `models/qwen3_17b`，复制后核对 `rrl_download_metadata.json` 中的权重SHA256。笔记本 `checkpoints/` 用于复查历史结果；台式机更换精度/硬件后重新建立A/B和共同FP16 SFT起点，不把跨硬件结果归因于采样收益。

迁移包包含：源码、依赖约束、smoke/pilot/main 配置、学习材料、交接状态、数据及审核文件、已完成运行的文本证据。排除 `.venv*`、CUDA 轮子、缓存、模型权重及大型训练状态，防止混用 CUDA/ROCm 运行环境。

重打源码包：`python scripts/export_transfer.py`；重打LoRA包：在已有真实权重的本机运行 `python scripts/export_checkpoints.py`。命令更新同名包及清单，不上传外部服务；改动后重新打包和核验。通过GitHub源码克隆不会自动得到二进制检查点；已上传的LoRA附件在上述Release，公开基础模型仍按锁定版本另行获取。

原生 Windows 和 Linux 都有 AMD 官方 PyTorch 支持路线，但不是同一套轮子。这里不执行驱动安装、重启或远程连接。`--hardware rocm` 仅切换项目的训练精度与量化设置，不会安装 ROCm。
