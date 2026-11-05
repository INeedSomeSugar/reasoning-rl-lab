# 来源与固定版本

2026-11-05核查；实际下载SHA256见对应manifest或模型metadata。

|对象|第一方来源|锁定/使用方式|
|---|---|---|
|GSM8K数据与MIT许可|[openai/grade-school-math](https://github.com/openai/grade-school-math)|Git revision `3101c7d5072418e28b9008a6636bde82a006892c`，原始train/test分别下载|
|SVAMP数据与MIT许可|[arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP)|精确revision在 `data/raw/sources.lock.json`，保留完整LICENSE|
|Qwen3-1.7B|[官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)|`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；已后训练，Apache2.0|
|Qwen3-0.6B备选|[官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)|`c1899de289a04d12100db370d81485cdf75e47ca`；未与1.7B混算|
|PyTorch CUDA wheel|[官方cu128索引](https://download.pytorch.org/whl/cu128/torch/)|2.8.0+cu128，cp312 Windows；下载脚本内固定索引提供的SHA256|
|PEFT|[v0.17.1源码](https://github.com/huggingface/peft/tree/v0.17.1)|LoRA与k-bit训练准备；真实依赖元数据在environment/package-requirements.json|
|TRL|[v0.23.1源码](https://github.com/huggingface/trl/tree/v0.23.1)、[版本文档](https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer)|训练目标在固定GRPOTrainer实现；loss_type与默认值不同，已显式记录|
|AMD Radeon Linux|[ROCm7.2.1矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)|RX9070XT支持；只说明供应商矩阵，项目未在台式机运行|
|AMD Radeon Windows|[ROCm7.2.1矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)|Windows11、Python3.12、PyTorch2.9；完整ROCm栈尚未全部支持|
|bitsandbytes|[v0.48.1文档](https://huggingface.co/docs/bitsandbytes/v0.48.1/installation)|本机CUDA候选；不假设其AMD支持与最新版一致|
|云GPU|[AutoDL](https://www.autodl.com/)、[Runpod价格](https://www.runpod.io/pricing)、[存储规则](https://docs.runpod.io/pods/pricing)|只用于待确认方案，不租卡或付费|

模型大文件的官方连接出现短流中断后，允许经 `hf-mirror.com` 续传同一revision权重；**官方Hugging Face API返回的LFS SHA256保持校验依据**。下载路线在 `models/<key>/rrl_download_route.json`，权重metadata和完整许可证随模型保留，不把镜像当作新的模型版本。

课程学习/适应性采样是已有研究方向。本项目先实现清楚的中等成功率权重并做静态/动态消融，不声称完成系统查新或提出新算法。
