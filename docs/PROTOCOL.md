# 实验协议 v0.1

## 预注册问题与范围

H1：在同一GPU、固定端到端增量训练时间预算下，D的独立数学题正确率高于C。C/D共享固定SFT检查点、训练池、奖励、GRPO目标与优化器，仅数据采样及其必要探测开销不同。E的初始权重和D相同，但训练中不再探测；D–E用于区分初始筛选与动态更新。没有新算法主张。

阶段顺序是A验证集基线→B SFT→C均匀GRPO→D动态→E静态。smoke单种子只验证链路。pilot先检验奖励变化、梯度、组内方差、选择分布与吞吐；由实际成本确定main的池大小、探测频率和时限，然后冻结具体配置。正式种子为13、29、47；不足三次就按实际次数报告，不补写重复。

## 数据边界

Git提交保留文件原始字节，`.gitattributes`禁用自动换行转换；跨Windows/Linux克隆后核对SHA256，不因checkout换行规则破坏固定数据或旧源码快照。准备数据时优先验证并复用已准备划分，源revision同时固定在configs/data.sources.lock.json，不从变化中的main重新选择版本。

数据锁定在 `data/manifest.json`，源Git revision及文件SHA256均保存。GSM8K官方train按近重复连通簇拆分，不拆开高度相似题。近重复检测阈值0.90固定在模型运行前，仅作题目卫生审计；不是根据测试正确率筛选题目。当前没有需要从官方train隔离的保留集近重复题。归一化不移除数字，因此数字变化较大的语义模板重复仍可能存在。

训练、探测、估计题目成功率仅可使用 `load_split('train', 'train'/'select')`。验证集只用于模型/超参数开发与预定检查点规则。`test`和`external`只允许已冻结方案的最终评测。原始保留题/答案、模型预测及其派生数据均不得回流训练池。代码的隔离是防止意外误用的工程约束，不是操作系统权限沙箱。

SVAMP来自官方 `arkilpatel/SVAMP`、MIT许可，一题一行，Body+Question组成输入，Answer为数值标签；只作外部迁移评测。两套公开题的预训练污染未知，解释为公开基准表现，不能据此证明新题泛化或排除记忆。

## 模型、提示与生成

优先Qwen3-1.7B，官方完整revision在 `configs/models.lock.json`。该模型已经后训练；“-Base”是另一个检查点，当前未采用。所有组使用相同的checkpoint、量化/精度、tokenizer、prompt版本、`enable_thinking=False`与生成预算。若切换到0.6B或改变精度，建立独立配置与基线，不能将变化归因于采样。

验证/最终评测固定greedy和相同max_new_tokens；训练及探测使用temperature0.8、top_p0.95、top_k20。明确创建GenerationConfig，不继承未记录默认值。不把非thinking设置写成“模型没有推理过程”：输出仍可包含可见计算步骤。

## 优化目标与截断

SFT使用Transformers Trainer的CausalLM交叉熵。prompt/padding位置label=-100，仅监督完整标准解及EOS；过长样本整条丢弃并记录，不能监督被截断的gold答案。LoRA只更新q_proj/v_proj的低秩矩阵，bias和骨干冻结，dropout=0。

TRL固定0.23.1。首版显式 `loss_type='grpo'`、`scale_rewards='group'`（每组样本标准差+1e-4）、`beta=0`、`epsilon=0.2`、`importance_sampling_level='token'`、`num_iterations=1`、AdamW、constant LR、weight_decay0、max_grad_norm1、`mask_truncated_completions=True`。默认GRPOConfig在该版本的loss_type是dapo，故本项目显式覆盖。beta=0不构建额外KL参考模型以减少本机显存，但这偏离包含KL项的原始配置；不声称逐项复现DeepSeekMath。

二值奖励只检查最终可解析数值是否与gold精确相等。评分器 `numeric_final_v2` 先取最后一个####之后的最终答案区域：其中必须只有一个无歧义数值，允许跨行、Markdown和单位文字；其次取boxed、明确的Answer:末行或最后一行独立数值。不从一般叙述中随意取最后一个数字。含多个候选数字、空占位符或无效显式标记不会回退到早先计算。数值用Decimal/Fraction规范化，支持千位逗号、负号、小数、分数；不执行任意表达式，不接受NaN。严格格式仍要求同一行 `#### 数字`，其合规率独立统计、不加格式奖励。

该解析规则在验证集开发时校准：v1未接受####后另起一行的答案，造成格式错误与正确性混淆。v2离线重评分保留原始预测与原指标，不重复生成，审计文件为 `rescoring_audit.json`。这一步在任何保留集启用前完成；最终评测同时冻结评分器版本与代码哈希。

生成达到上限而无EOS记为截断；TRL将其全部completion loss mask置0，其答案正确性仍可单独记录，组内奖励归一化遵循固定TRL实现。记录组内零方差比例和非零优势数量；全组对/错会产生零优势，loss接近0不说明模型成功。

## 难度估计

训练池每题K次真实采样，`p=(successes+0.5)/(K+1)`。`g(p)=4p(1-p)`，`q(i)=u/N+(1-u)*g(p_i)/sum_j g(p_j)`，u=0.25。正的均匀份额保证每题有非零被选概率，不保证短期内每题都被选。

D在step0与每refresh_interval个optimizer step进行全池刷新，采用最新K次估计，不累积早期观测；E仅step0。探测有独立并恢复的torch RNG状态，C/E/D的训练采样随机数规则保持可追踪。实际题目选择发生在TRL生成边界，不受DataLoader提前读取影响；记录每个生成批次选择的题ID、权重和刷新次数。当前仅支持单进程单GPU，拒绝未实现的多卡分组。

## 计算成本公平

扩大本机pilot的开发配置为`local_pilot.yaml`：SFT池是固定训练随机序列前128题，GRPO池初定其后16题，在B验证耗时511秒后缩为其后8题并将末次验证预留校准为540秒；二者不相交，C/D/E共享后者。调整只依据实际资源成本，在GRPO开始前确定，避免少量GRPO题已被SFT逐题监督；仍全部来源于GSM8K官方train，不涉及验证题或保留题。

上述NF4/SDPA轮次在D验证时中断，不能拼接后端修改后的结果。新NF4/eager开发计划为`experiments/local_nf4_eager_development_plan.json`：重建A/B、固定前16道验证题和256-token上限；SFT仍为同128题/32步。训练题标签疑义审阅导致候选8题中预先排除gsm8k-train-05134，保留7题，原始数据与旧奖励不改。新C/D/E共享600秒优化/探测预算，末次验证预留按B实测时长执行预先固定的取整规则。具体配置、源码、数据、SFT哈希及题池在新冻结文件中保存。该规模仅作开发诊断，不能代表全量训练池的效果。

可声明`final_validation_reserve_seconds`，依据先前固定验证的实际时长为末次验证预留预算。训练/探测在剩余预算到达该预留量时停止；末次验证的真实成本照常计入grpo_total。预留不是精确预测，实际超额或未用尽的时间均需报告，不能把共同上限直接当作实际耗时相等。

主轴是**同机端到端GRPO增量wall time**，从首次探测前开始，包含所有探测、生成、优化与预定验证。共同SFT作为共享前置成本单列。模型加载/下载与云租用时间另列，不从账单中抹除。C不需要探测，可将省下预算用于更多更新；不能强制C与D只跑相同步数然后称预算公平。

该上限实际覆盖`grpo_total`阶段，不覆盖进程启动、Trainer准备、检查点序列化及另一次独立重载评测。加载有单独计量；汇总给出加载、GRPO、重载评测之和，并标明未计量的外围阶段。保留原始Trainer吞吐字段，但时间早停后它可能仍用计划max_steps计算，实测吞吐必须用实际完成步数或真实token重算。

同时保存生成token、探测prompt/生成token、策略loss输入token、验证token、CUDA event device elapsed与本进程峰值allocated/reserved显存。token相加只是辅助预算代理，不能当等价FLOPs。CUDA event elapsed不是活跃kernel计时。嵌套probe/validation成本已包含在grpo_total，汇总不能重复相加。不同GPU/驱动/精度的耗时分表，不直接比较速度收益。

预算停止在自然批次/验证边界，报告超额；probe同样受预算限制。若预算无法容纳完整初始探测，缩小池后重新冻结预实验配置，不隐瞒探测失败或使用测试集替换。main的steps=100000只是防止相同步数过早限制C的保护上限，真实停止以时间/token为先。

## 评测与归因

精度按实际模块记录，不仅抄写配置：本机实测TRL0.23.1的`prepare_peft_model`会把4-bit BF16模型的FP32 embedding/lm_head转成BF16，LoRA与norm保持FP32；见`reports/trl_precision_audit.json`。训练内验证因此仅为诊断。A/B/C/D/E的可比较终点均由共同独立加载器重载后生成，固定非量化层FP32及NF4计算BF16。当前运行入口保存`model_precision.json`；早期运行的精度通过单独实测审计解释，不伪造历史侧文件。

逐题保存原始prediction、gold、解析结果、正确性、格式、token数和截断标记。评测指标包括正确率及Wilson95%区间（题目抽样不确定性，不是种子间不确定性）、长度、截断、提取/格式错误、训练reward、组内零方差、采样覆盖及各类题曝光。

至少三个C/D配对种子；每对相同SFT起点。报告逐种子结果、均值/标准差，以及相同可观测成本点的正确率。成本曲线只画实际验证点，单点画散点，不伪造过程曲线。若用插值作辅助，必须明确且不改变未观测结果。对同一批题的比较可做配对bootstrap；不能仅凭单组Wilson区间不重叠来宣布显著。

自动标签只能确认截断、提取失败、格式和重复迹象。算术错误、条件遗漏及推理歧义由人工结合题目和输出审阅，不把“答案错”自动标成算术错。记录human_reviewed字段，未审阅不冒充人工结论。按训练题运算/长度等可解释特征比较曝光比例，检查过度偏向；小样本只作诊断。

正式方案在验证集确定后，冻结模型、配置、具体端点检查点、全部种子和数据哈希。官方test和SVAMP只评一次，不参与checkpoint选择或筛选有利结果。无提升结果保留；用长度/截断、奖励稀疏、零方差、探测成本、题型偏置等实际证据归因，不能拿loss下降替代泛化提高。

参考：[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[TRL0.23.1 GRPO文档](https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer)、[LoRA](https://arxiv.org/abs/2106.09685)。这些是方法来源；项目的实际配置和证据以保存文件为准。
