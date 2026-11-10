# 台式机自动拉取、核验与接续

这份指令用于复查 2026-11-05 的初次迁移，固定项目路径为 `D:\User\AI\reasoning-rl-lab`，公开仓库为 `https://github.com/INeedSomeSugar/reasoning-rl-lab`（2026-11-10 核验），历史 Release 标签为 `local-pilot-20261105`。该次迁移及后续台式机开发实验已完成，当前结果与复现入口见 [README](../README.md) 和 [最新复现说明](../docs/REPRODUCING_REPAIR_20261106.md)。下文保留历史材料恢复流程；云GPU、付费API仍需另行确认。

## 自动准备项目

1. 先读取 `D:\User\AI\AGENTS.md`（存在时），检查 Git、GitHub CLI 和目标目录。父目录不存在时可创建。只修改本项目；保持 AI 下其他项目不变。
2. 公开仓库可通过 Git HTTPS 克隆，公开 Release 附件可直接下载，无须先登录 GitHub。已有 GitHub CLI 登录时可复用下面的 `gh release download`；未登录时使用公开 Release 下载链接并按同样规则核验。不得索取、显示或把令牌写入项目。
3. 目标目录不存在时，克隆到精确路径：`git clone https://github.com/INeedSomeSugar/reasoning-rl-lab.git D:\User\AI\reasoning-rl-lab`。若已存在，先核对目录、remote 和工作树；匹配且干净时安全复用或仅作 fast-forward 更新，保留本机修改。不删除现有目录，不强制重置，不把另一份源码嵌套进项目目录。
4. 进入项目后阅读 `AGENTS.md`、`README.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md`、本文件与 `migration/CODEX_PROMPT_ZH.md`，记录实际 Git commit。Git main 是当前源码与交接文档的依据。

## 自动下载并核验材料

5. 在项目 `dist/` 中下载该 Release 的三个附件；已有文件先比对哈希，匹配则复用：

   ```powershell
   gh release download local-pilot-20261105 --repo INeedSomeSugar/reasoning-rl-lab --pattern reasoning-rl-lab-desktop-transfer-20261105.zip --pattern reasoning-rl-lab-laptop-lora-20261105.zip --pattern SHA256SUMS.txt --dir dist --skip-existing
   ```

   不覆盖尚未核验的同名文件。`environment/publication.json`保存了发布时的大小和SHA256；用 `Get-FileHash -Algorithm SHA256` 或标准库进行比对，同时核对 `SHA256SUMS.txt`，错误则停止使用该附件并记录原因。
6. 源码ZIP是历史快照，留在 `dist/` 用于归档，**不解压覆盖main工作树**。Git克隆包含当前代码、固定数据、报告与原始文本证据。`TRANSFER_MANIFEST.json`只适用于历史源码ZIP的原始解压副本，不用它校验已更新的Git工作树。
7. 检查LoRA ZIP条目及 `migration/CHECKPOINTS_MANIFEST.json`，先解压到项目内独立暂存目录。确认每个解析后的绝对路径均在暂存目录内，拒绝路径穿越；只把48个清单文件复制到本项目的 `checkpoints/` 对应位置。已有文件哈希相同则跳过，不同则保留并报告冲突，不能直接覆盖。不要把顶层 `reasoning-rl-lab/` 再嵌套到项目根目录，也不要用包内旧文档覆盖当前文档。
8. 按清单核验16份适配器。项目包尚未安装时先用PowerShell或Python标准库核验文件哈希；建好环境并安装本项目后，再运行 `python scripts/verify_checkpoints.py` 和 `python scripts/audit_evidence.py`。这些权重包含能力检查、SFT、GRPO与恢复工件，不是16个正式实验终点；不含完整优化器状态。

## 自动准备台式机训练

9. 按 `migration/CODEX_PROMPT_ZH.md` 继续：实查操作系统、RX9070XT、驱动、显存、Python、RAM与磁盘，核对AMD官方支持组合。独立环境、缓存和公开模型均放在项目内；不复制笔记本CUDA环境，不把NVIDIA NF4配置直接用于AMD。涉及系统驱动或重启时先说明实际必要条件，由用户处理。
10. 下载锁定版本的免费公开Qwen3-1.7B基础检查点（已后训练，非Base），或核验已有完整权重。Git与上述Release均不包含这份基础模型；它与LoRA适配器是两个部分。固定数据沿用现有版本与划分，测试/外部保留集不用于开发或选样。
11. 建立目标机独立环境，完成前向、反向、LoRA更新、保存重载和短SFT/GRPO。再按实测吞吐确定pilot/main，重建目标机A/B与共同SFT，继续三种子同预算对照、静态消融及冻结后的独立测试。不要只下载文件后就把迁移记为训练完成。
12. 保存下载、哈希、环境、真实运行与未完成事项，更新HANDOFF和PROJECT_CONTEXT。笔记本历史结果、目标机新结果分别记录；无提升也保留，不改简历、不声称新算法。不因常规可逆步骤反复询问确认。

用户只需在台式机上启动有 `D:\User\AI` 写权限的 Codex 并交付任务；系统驱动条件确实缺失时再由用户处理，不要求手工下载、解压或复制检查点。
