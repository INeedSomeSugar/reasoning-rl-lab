# RX 9070 XT 台式机迁移

当前优先完成本机实验，尚未生成迁移压缩包。用户已授权以后通过GitHub下载接续，默认私有仓库；本机GitHub登录失效，用户选择稍后登录，因此当前没有已推送仓库的承诺。仓库可用后优先git clone，压缩包作为离线备选。

最方便的方式：把 `dist/reasoning-rl-lab-desktop-transfer-20261105.zip` 复制到台式机并解压成独立项目目录，用台式机 Codex 打开该目录，将 [CODEX_PROMPT_ZH.md](CODEX_PROMPT_ZH.md) 中的提示词粘贴给它。

GitHub方式：在台式机以自己的GitHub账号克隆已确认的私有仓库（具体地址见HANDOFF），打开项目目录并粘贴同一提示词。源码、固定数据、许可、指标和原始文本证据随Git版本保存；虚拟环境和公开模型权重在目标机器重新建立。LoRA检查点的可下载位置会在实际上传后记录，当前真实权重保存在笔记本checkpoints中，不能仅凭代码仓库推送声称权重已上传。

不需要远程桌面、账号共享或跨机器访问。源码与固定数据只有少量 MB，模型按锁定 revision 在台式机重新下载。也可手动复制 `models/qwen3_17b`，但请先完整完成笔记本下载，复制后核对 `rrl_download_metadata.json` 中的权重 SHA256。`checkpoints/` 可另外复制用于复查笔记本结果；台式机正式对照应建立共同 FP16 SFT 起点。

迁移包包含：源码、依赖约束、smoke/pilot/main 配置、学习材料、交接状态、数据及审核文件、已完成运行的文本证据。排除 `.venv*`、CUDA 轮子、缓存、模型权重及大型训练状态，防止混用 CUDA/ROCm 运行环境。

打包命令：`python scripts/export_transfer.py`。该命令生成文件清单与 SHA256，更新同名压缩包；内容保持在本项目目录内。台式机解压后执行 `python scripts/verify_transfer.py`，然后让 Codex 检查实际环境。

原生 Windows 和 Linux 都有 AMD 官方 PyTorch 支持路线，但不是同一套轮子。这里不执行驱动安装、重启或远程连接。`--hardware rocm` 仅切换项目的训练精度与量化设置，不会安装 ROCm。
