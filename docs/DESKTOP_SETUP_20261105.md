# RX 9070 XT 台式机接续记录

日期：2026-11-05。源码来自 main `c011b71b54b66f028cc545904e45c28e8da8f85a`，浅克隆；原始工作树干净。父目录 `D:\User\AI\AGENTS.md` 不存在，已读取项目 AGENTS、README、PROJECT_CONTEXT、HANDOFF 与两份迁移指令。所有改动仅在本项目。

## 已完成迁移

已有 GitHub 系统密钥环登录有效，沙箱内的失效提示不代表需要重新登录。首次 Git/TLS 下载中断，保留失败文件；改用 HTTP/1.1 克隆与分段续传。没有显示或保存明文凭据。

Release `local-pilot-20261105` 三个附件的大小、SHA256 与 `environment/publication.json`、远端 asset digest、`SHA256SUMS.txt` 一致。LoRA ZIP 全部成员已做路径与符号链接检查，只暂存并恢复 manifest 中的 48 个文件、16 份适配器；目标冲突会停止，不覆盖。源码历史 ZIP 没有解压覆盖 main。证据：`environment/desktop_release_verification.json`、`logs/desktop_release_restore.log`。这些是笔记本工件，不含完整优化器续训状态，也不是 AMD 训练结果。

## 实查主机

原始快照：`environment/desktop_host_20261105.json`。

- Windows 11 专业版，build 26300.9818，DisplayVersion 26H2；注册表旧 ProductName 为 Windows 10 Pro，保留两者。
- Ryzen 7 9700X，8 核 16 线程。物理 RAM 24 GiB，系统可用总量约 23.1 GiB；初查空闲量会随运行变化。
- RX 9070 XT 当前 Status=OK、ConfigManagerErrorCode=0；驱动 `32.0.32015.12012`，日期 2026-10-28。[AMD 26.10.2 发行说明](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-10-2.html)与该 Driver Store 版本对应。
- 独立显卡注册表 64 位显存字段为 17,095,983,104 字节；PyTorch 已核对为 15.92 GiB，首次空闲 15.77 GiB。不能使用 Win32_VideoController 的 32 位 AdapterRAM 当作真实显存，也不能把集显的共享内存写成独显显存。
- D 盘快照空闲 226,771,542,016 字节；本项目环境、缓存、模型和日志均在 D 盘项目内。
- `wsl --version` 返回未安装适用于 Linux 的 Windows 子系统；存在 Lxss 注册表项不等于有可运行 Linux 环境。

## 官方组合与证据边界

查阅日期为 2026-11-05，区分文档支持与这台机器的实际验证：

| 路线 | 官方文档 | 本轮选择与限制 |
| --- | --- | --- |
| 原生 Windows ROCm 7.2.1 | [兼容矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)、[限制](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/limitations/limitationsrad.html) | 旧文档明确不支持 ML training；未把该推理支持解释为训练认证，未安装这条旧路线 |
| 原生 Windows ROCm 10.0.0 | [矩阵](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html)、[PyTorch 安装](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)、[AMD LoRA 指南](https://developer.amd.com/playbooks/pytorch-finetuning/) | 固定 torch 2.13.0+rocm10.0.0、device-gfx1201、Python 3.12.14；其指南已包含 Windows LoRA 路径。先验证 FP16；真实 SFT 数值检查失败后，另建非量化 BF16/SDPA/AdamW 轮次 |
| WSL/Linux | [WSL 7.2.1 矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)与各自安装说明 | 当前没有可运行 WSL；不把 Windows 结果写成 Linux 结果，没有安装系统组件 |

10.0.0 矩阵列 Windows 11 **25H2** 和 Adrenalin **26.8.1**；本机是 26H2、26.10.2，**不是文档逐项相同的已验证组合**。先保留系统状态并完成项目内实测；不能承诺兼容，也不凭“驱动更新”推断需要降级。PyTorch/ROCm 官方包可用不代表固定 PEFT/TRL 获得 AMD 逐项认证。若 GPU 初始化或训练失败，保留确切异常后再确定是否需要系统驱动/WSL/重启条件。

使用 AMD 指南固定的 10.0.0 路线，而不是追逐当前最新 10.1.0，是为了沿用已发布 Windows LoRA 示例的框架组合；锁定公共模型和项目上层依赖仍保持原版本。两者能否共同工作以本机前后向、LoRA 和真实短训练为准。

## 项目独立环境与接续入口

uv 0.12.10 官方 ZIP 的 SHA256 为 `f65744f94072152b1f86ba2aace4d01f1124d9a8ecb235805039e3718c36cac2`。Python 3.12.14 位于 `.cache/python`，`.venv` 独立；创建时使用 `--no-registry --no-bin`。`scripts/desktop_env.ps1` 设置项目缓存及临时目录。没有修改全局 PATH、安装系统驱动、关闭安全特性或使用付费服务。

运行前 dot-source `scripts/desktop_env.ps1`。初始 `configs/desktop_rocm10_smoke.yaml` 的两个 FP16 家族均在 SFT 阶段失败，记录保留。成功的短链路为 `configs/desktop_rocm10_bf16_smoke.yaml`：seed 13，短 SFT 池与 GRPO 池不相交，C/D/E 共用实际台式机 SFT。扩大开发轮次使用独立的 `configs/desktop_rocm10_bf16_pilot.yaml`，不覆盖初始配置。正式 test/SVAMP 仍封存。

锁定 Qwen3-1.7B 已下载完毕，共 11 个文件。两块权重合计 4,063,515,592 字节，权重与 tokenizer 的官方 LFS SHA256、小文件的官方 Git blob SHA1 均通过；另记录所有本地 SHA256，见 `environment/desktop_model_verification.json`。模型修订与项目 lock 一致，没有改用 Base 或其他模型。

六个 AMD 大体积 wheel 已完成下载和 ZIP CRC 检查，官方索引没有发布 wheel SHA256，因此 `environment/desktop_amd_wheels.json` 中的摘要是本地完整性记录，不能写成通过厂商 SHA256 验证。已安装 torch 2.13.0+rocm10.0.0，首次原生探测返回 `torch.version.hip=7.15.26333`、GPU 可用，设备 0 为 RX 9070 XT，设备 1 为集成显卡；见 `logs/desktop_torch_initial_probe.log`。包的 ROCm 版本名和运行时 HIP 字段均原样记录；`torch.cuda` 命名空间在此调用 HIP，不表示安装了 CUDA。

普通依赖首次通过 uv 在线安装时，大 wheel 的流式下载多次超时。仅停止经确认属于本项目的安装进程，保留失败日志；随后按解析出的精确版本从 PyPI 分段下载 52 个 wheel，逐个核验发布方 SHA256、大小及 ZIP CRC，再离线安装。`environment/desktop_common_wheels.json` 保存完整收据，`environment/requirements-desktop-rocm10.lock.txt` 和 `environment/packages-desktop-rocm10.json` 保存最终实际依赖；pip check 通过。本地环境没有安装 bitsandbytes、flash-attn 或 vLLM。

## 首次实测与后备精度

36 项工程测试通过；FP16/SDPA 的前后向、有限非零 LoRA 梯度、骨干无梯度、真实更新和保存重载通过，重载 logits 最大差为 0。首次 A 验证 7/8，仅用于开发诊断。随后 B 在 Trainer 自动启用独显和集显的 DataParallel 后原生访问异常。按[官方 GPU 隔离说明](https://rocm.docs.amd.com/en/docs-10.0.0/reference/system-optimization/gpu-isolation.html)，仅在本项目进程设置 HIP_VISIBLE_DEVICES=0；没有在设备管理器禁用集显或更改系统环境。

新单卡 FP16 运行再次通过能力和 A 检查，但 B 第一次优化前检测到非有限 LoRA 梯度并停止。未将失败运行记作训练完成，也不推断为显存不足或驱动故障。进一步采用 `configs/desktop_rocm10_bf16_smoke.yaml`，保持非量化、SDPA 和其他参数，独立重跑 A/B/C/D/E。[AMD 精度文档](https://rocm.docs.amd.com/en/docs-10.0.0/reference/precision-support.html)列 RDNA4 支持 BF16，本机 is_bf16_supported 返回 True；完整模型验证仍以新运行结果为准。CLI 现保留显式非量化配置中的精度；旧 NF4 配置的 --hardware rocm 仍默认转 FP16/非量化。

## 已完成的 BF16 短链路

家族 `desktoprocm10bf16_20261105` 已实际完成能力检查、A、4 步 SFT、各 4 步 C/D/E，以及各自的独立重载验证。40 项工程测试通过。能力检查中 1,605,632 个参数可训练，LoRA 梯度有限且非零，冻结骨干无梯度，真实优化后 LoRA 改变，保存重载 logits 最大差为 0。

共同 8 道开发题、256-token 上限得到 A 8/8，B/C/D/E 均 7/8。C 的组内奖励零方差率 100%，LoRA delta=0；D/E 有真实更新，但两次 D 探测给出的采样权重相同，尚不能作为动态刷新生效证据。对应 GRPO 实际时间 C 36.6 秒、D 67.6 秒、E 51.8 秒，不能以这组不同成本和极小样本宣称方法收益。

见 [短链路报告](../reports/comparisons/desktoprocm10bf16_20261105/findings.md)、`environment/desktop_verification_20261105.json`。后者独立于历史笔记本的 `environment/verification.json`。

## 按实测成本扩大的开发轮次

家族 `desktoprocm10pilot_20261105` 使用 128 道 SFT 题、32 步、1 epoch，以及互不相交的 28 道 GRPO 题。GRPO 候选 32 道的条件/标签由 agent 在训练前审阅，四项共同排除保存在 `experiments/desktop_expanded_train_review_20261105.json`；没有改动原始数据，不能写成用户人工审阅或完整标签质量保证。

A/B 改为共同 32 道验证题、512-token 上限，单独重跑，不能和短链路的 256-token 结果混算。B 验证耗时 124.64 秒；按预先声明的取整规则确定 180 秒最终诊断验证预留，加上 300 秒训练及探测额度，得到每组 480 秒 GRPO 总时间上限。在任何 C/D/E 结果产生前写出 `experiments/desktoprocm10pilot_20261105-frozen-development.json`，共同冻结源码、池、超参、SFT 哈希和验证预留。

C/D 按 seed 13/29/47 配对重复，E 只运行 seed 13。探测和训练内验证计入 GRPO 总时间；未完成的探测计入耗时但不用来更新选择权重。加载、独立重载验证另计。自然边界、输出长度和未完成探测会导致实际耗时不同；共同上限不能写成精确相同计算量。完整活动/完成边界见 `environment/desktoprocm10pilot_20261105-status.json`，完成的终点由 [开发报告](../reports/comparisons/desktoprocm10pilot_20261105/findings.md) 汇总。

目前已确认 SFT 的开发负面结果：A 28/32、B 16/32，B 有 9/32 输出截断、10/32 解析失败；审阅记录在 `reports/comparisons/desktoprocm10pilot_20261105/sft_degradation_review.json`。输出重复、超长及可解析的算错均有出现；不能仅归因为评分器，也未确认具体训练根因。D seed 29 已完成全池的后续探测，28 个采样权重均变化，证明动态刷新机制在本机真实执行；这一事实不等于正确率提升。

开发冻结明确不授权保留集访问。正式协议、具体检查点、最终评测预算尚未冻结，GSM8K test 与 SVAMP 保持封存；没有云 GPU 或付费 API 费用，未计量本机电费。

## 本轮完成状态

扩大开发轮次在 2026-11-05 23:50（UTC+8）完成，状态 `completed_development_pilot`，无活动训练。整轮墙钟 74 分 42 秒。A/B 为 28/32、16/32；C 的 seed13/29/47 为19/19/18，D为19/16/19，E seed13为17（均以32为分母）。C/D均值为58.33%/56.25%，D−C配对差均值−2.08个百分点，没有动态优势或相对原模型的提升证据。静态仅一次，不补写重复或显著性。

194次 GRPO 优化前检查的梯度均有限，冻结骨干无梯度，其中134次非零；七个终点均真实改变 LoRA。D seed29 的后续探测完整且权重变化；D seed13/47 的后续探测分别只完成15/11题，未用于更新。D seed29 的再下一次探测也只完成13题，照计耗时。全部原始探测、采样、rollouts、源码与精度快照可检查。

七组 GRPO 实际合计3108.52秒，另计独立重载评测1045.00秒、训练加载21.42秒；共同B的 SFT 实测20.23秒、28381处理token/15160监督token。整轮墙钟还包含A/B评测和进程准备等，不能只把局部阶段之和称为全程GPU成本。训练峰值 PyTorch allocated/reserved 为4.62/7.07 GiB；没有观察到本轮OOM，不代表对所有未来负载的稳定性保证。

最终证据审计覆盖66份历史及新增运行，状态pass，无缺失检查点。正式main与最终独立评测未完成。下一轮应先在新的开发家族排查SFT退化，按实测成本确定更充分的动态刷新配置；保留本轮完整负面结果，不改动冻结配置或据验证结果回看保留集。最新接续入口为 [HANDOFF.md](../HANDOFF.md)。
