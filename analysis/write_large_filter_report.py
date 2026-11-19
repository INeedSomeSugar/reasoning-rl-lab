"""Chinese final report from audited evidence and completed blinded Agent review."""
import argparse
from collections import Counter
import csv
import datetime as dt
from pathlib import Path
import statistics
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file, jsonl_read
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, TRAIN_SEEDS, PROBE_SEEDS


def pct(value):
    return "缺失" if value is None else f"{value*100:.3f}%"


def pp(value):
    return "缺失" if value is None else f"{value:+.3f}pp"


def table(headers, rows):
    return "|"+"|".join(headers)+"|\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("|"+"|".join(map(str,r))+"|" for r in rows)+"\n"


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    s = read(out / "summary.json")
    plan = read(exp / "protocol_frozen.json")
    audit = read(out / "evidence_audit.json")
    data_audit = read(exp / "history_audit.json")
    if s["missing_evaluation_objects"]:
        from write_large_filter_incomplete_report import main as write_incomplete
        write_incomplete(family)
        return
    review_path = out / "blind_review_summary.json"
    if not review_path.exists():
        raise RuntimeError("Complete the preregistered blinded Agent case review before declaring the final report complete")
    review = read(review_path)
    if review["pending_cases"]:
        raise RuntimeError("Blinded review is not complete")
    exact_units = review.get("exact_text_deduplication")
    review_dedup_note = (f"其中按题目ID、题干、金标、参考解、响应原文和截断状态全部精确相同的规则，"
        f"实际逐份审阅{exact_units['unique_response_units_actually_annotated']}份唯一响应。程序仅将已明确写下的Agent判断回填到重复展示，"
        "保留全部原始case/alias映射，不进行语义合并，不自动生成错误类别；全部唯一响应判断固定后才允许回填和解盲。"
        if exact_units else "全部案例均有明确的逐响应Agent判断。")
    N = s["n"]
    endpoints = {r["endpoint"]: r for r in s["endpoints"]}
    diagnostics = s["training_signal"]
    budget = plan["budget"]
    rows = []
    for anchor in ("A", "S2"):
        r = endpoints[anchor]
        rows.append([anchor, str(r["correct"])+f"/{N}", pct(r["accuracy"]), "单一固定锚点"])
    for arm in ("U", "F", "R"):
        blocks = [None] if arm == "U" else PROBE_SEEDS
        for j in blocks:
            rs = [endpoints[f"U-s{seed}" if j is None else f"{arm}-j{j}-s{seed}"] for seed in TRAIN_SEEDS]
            rows.append([arm if j is None else f"{arm} / {j}", " / ".join(str(r["correct"]) for r in rs),
                pct(statistics.mean(r["accuracy"] for r in rs)), f"{statistics.stdev(r['accuracy']*100 for r in rs):.3f}pp"])
    main_table = table(["方法／探测块", f"正确数（训练seed13/29/47/71/97；分母各{N}）", "均值正确率", "5训练seed描述性SD"], rows)
    matrices = []
    for metric, label in (("paired_F_minus_U_matrix_pp", "F−U"), ("paired_F_minus_R_matrix_pp", "F−R")):
        m = s[metric]
        matrices.append(f"### {label}的5×3配对差\n\n"+table(["训练种子"]+[str(j) for j in PROBE_SEEDS],
            [[seed]+[pp(m[a][b]) for a in range(3)] for b, seed in enumerate(TRAIN_SEEDS)]))
    interval_rows = []
    for name, metric in (("F-U", "F_minus_U_mean_pp"), ("F-R", "F_minus_R_mean_pp"), ("F-S2", "F_minus_S2_mean_pp"), ("F-A", "F_minus_A_mean_pp")):
        cond = s["conditional_intervals"][name]["percentile95_pp"]
        crossed = s["crossed_exploratory_intervals"][name]["percentile95_pp"]
        interval_rows.append([name, pp(s[metric]), f"[{cond[0]:+.3f}, {cond[1]:+.3f}]pp", f"[{crossed[0]:+.3f}, {crossed[1]:+.3f}]pp"])
    interval_table = table(["比较", "平均差", "条件95%簇bootstrap区间", "交叉bootstrap探索性95%区间"], interval_rows)
    probe_rows = [[p["probe_seed"], p["M"], "/".join(str(p[f"raw_success_{i}_items"]) for i in range(5)),
        f"{p['full_process_seconds']:.6f}", p["random_subset_seed"], p["R_overlap_with_F"]] for p in s["probe_summary"]]
    probe_table = table(["探测seed", "M_j", "原始成功次数0/1/2/3/4对应题数", "完整P_j秒", "R抽样seed", "R∩F题数"], probe_rows)
    selection = read(exp / "selection_diagnostics.json")
    jaccard_table = table(["F集合"]+[str(j) for j in PROBE_SEEDS],
        [[j]+[f"{selection['F_jaccard'][str(j)+'/'+str(k)]:.4f}" for k in PROBE_SEEDS] for j in PROBE_SEEDS])
    sensitivity_rows = []
    for scope, key in (("探测块", "probe_block_sensitivity"), ("训练种子", "training_seed_sensitivity")):
        for seed, values in s[key].items():
            for contrast in ("U", "R"):
                sensitivity_rows.append([scope, seed, "F−"+contrast, pp(values[f"F_minus_{contrast}_mean_pp"]),
                    " / ".join(pp(v) for v in values[f"F_minus_{contrast}_range_pp"]), f"{values[f'F_minus_{contrast}_sd_pp']:.3f}pp"])
    sensitivity_table = table(["方向", "seed", "比较", "均值", "范围min/max", "描述性SD"], sensitivity_rows)
    signal_rows, transition_rows = [], []
    for arm in ("U", "F", "R"):
        keys = [k for k, r in endpoints.items() if r["arm"] == arm]
        ds = [diagnostics[k] for k in keys]
        steps = sum(d["optimizer_steps"] for d in ds)
        zero = sum(d["all_correct_groups"]+d["all_wrong_groups"] for d in ds)
        rates = s["charged_signal_rates"][arm]
        signal_rows.append([arm, len(keys), steps, f"{zero/steps:.3%}", sum(d["effective_completions"] for d in ds),
            f"{rates['effective_completions_per_charged_second']:.7f}", sum(d["nonzero_gradient_steps"] for d in ds),
            f"{rates['nonzero_gradient_steps_per_charged_second']:.7f}"])
        trans = [s["transitions_from_S2"][k] for k in keys]
        transition_rows.append([arm, f"{statistics.mean(t['S2_correct_then_wrong'] for t in trans):.3f}",
            f"{statistics.mean(t['S2_wrong_then_correct'] for t in trans):.3f}",
            f"{statistics.mean(t['net_correct_difference'] for t in trans):+.3f}"])
    signal_table = table(["方法", "实际训练终点数", "优化步总数", "全对/全错零方差组比例", "有效completion总数", "有效completion/完整计费秒", "非零梯度步", "非零梯度步/计费秒"], signal_rows)
    transition_table = table(["方法", "S2正确→错误（每终点均值题数）", "S2错误→正确（每终点均值题数）", "新学会−回归"], transition_rows)
    with (out / "training_signal_windows.csv").open(encoding="utf-8-sig") as f:
        windows = list(csv.DictReader(f))
    window_rows = []
    for arm in ("U", "F", "R"):
        for window in ("first", "middle", "last"):
            selected = [r for r in windows if r["arm"] == arm and r["window"] == window]
            groups = sum(int(r["groups"]) for r in selected)
            eff = sum(int(r["effective_completions"]) for r in selected)
            mixed = sum(int(r["mixed_groups"]) for r in selected)
            correct_groups = sum(int(r["all_correct_groups"]) for r in selected)
            wrong_groups = sum(int(r["all_wrong_groups"]) for r in selected)
            tokens = sum(int(r["generated_tokens"]) for r in selected)
            truncated = sum(int(r["truncated_completions"]) for r in selected)
            label = {"first": "前1/3", "middle": "中1/3", "last": "后1/3"}[window]
            window_rows.append([arm, label, groups, f"{correct_groups/groups:.2%}" if groups else "无组",
                f"{wrong_groups/groups:.2%}" if groups else "无组", f"{mixed/groups:.2%}" if groups else "无组",
                f"{eff/(4*groups):.2%}" if groups else "无组", f"{tokens/(4*groups):.1f}" if groups else "无组",
                f"{truncated/(4*groups):.2%}" if groups else "无组"])
    window_table = table(["方法", "时间窗", "组数", "全对组", "全错组", "混合组", "有效completion比例", "平均生成token", "截断比例"], window_rows)
    FU = s["F_minus_U_mean_pp"]
    FR = s["F_minus_R_mean_pp"]
    FS2 = s["F_minus_S2_mean_pp"]
    ci = s["conditional_intervals"]["F-U"]["percentile95_pp"]
    fr_ci = s["conditional_intervals"]["F-R"]["percentile95_pp"]
    eng = s["engineering_signal_threshold_met"]
    dev = s["development_evidence_threshold_met"]
    h1 = ("达到" if eng else "未达到")+"预登记工程信号复现标准；"+("达到" if dev else "未达到")+"预登记新开发证据标准。"
    if fr_ci[0] > 0:
        h2 = "固定模型条件下F−R区间下界大于0，支持本池和本批选择块中的成功率筛选优势。"
    elif fr_ci[1] < 0:
        h2 = "固定模型条件下F−R区间上界小于0，本池和本批选择块中同规模随机子集的结果优于成功率筛选。不能把F相对U的任何差异归因于筛选优于随机选题。"
    else:
        h2 = "F−R的条件区间未建立筛选优于同规模随机子集的证据；不能把F相对U的任何差异全部归因于成功率筛选。区间跨零也不证明两者等效。"
    h3 = "F相对S2的平均准确率没有正增长，应优先按新学会与回归的平衡解释F−U。" if FS2 <= 0 else "F相对S2有平均净正确数增长；其条件区间和逐种子回归/新学会分解限制了可作的能力增长结论。"
    next_step = "达到前瞻开发阈值，值得另立第二独立候选池复验；本报告不自动追加运行或启封正式测试。" if dev else "未达到前瞻开发阈值，本家族到此结束；保留全部负向和跨零结果，不追加种子、阈值或训练直到得到正结果。"
    if eng and not dev:
        next_step += "工程信号与新开发证据未同时满足前瞻标准，下一步优先检验训练目标与迁移之间的差距。"
    costs = s["physical_costs"]
    sigma = sum(d["zero_gradient_adam_changed_steps"] for d in diagnostics.values())
    peak = max(d["peak_allocated_bytes"] for d in diagnostics.values())/2**30
    peak_reserved = max(d["peak_reserved_bytes"] for d in diagnostics.values())/2**30
    operational_gain = s["failure_inclusive_F_vs_U_effective_signal_relative_improvement_bounds"]
    operational_gain_text = (f"{operational_gain[0]:+.2%}至{operational_gain[1]:+.2%}" if operational_gain is not None
        else "未定义（U有效信号为0）")
    signal_improvement = f"{s['F_vs_U_effective_signal_relative_improvement']:+.2%}" if s["F_vs_U_effective_signal_relative_improvement"] is not None else "未定义（U基线有效信号为0）"
    text = rf"""# 256题固定预筛复验与同规模随机子集消融

家族`{family}`；本地RX9070XT，固定Qwen3-1.7B与同一真实S2。35个训练终点及A/S2共37对象均由独立新进程在同一新开发{N}题上评估。全部35个训练单元终止并登记权重哈希后，才首次开启该开发集。所有预登记训练／探测种子均报告，没有选择最好种子或最好checkpoint。

## 主要发现与判断

- 主比较F−U为**{FU:+.3f}pp**，联合题目簇bootstrap条件95%区间**[{ci[0]:+.3f}, {ci[1]:+.3f}]pp**。该区间固定本次训练和探测种子、检查点及单一候选池。
- F−R为{FR:+.3f}pp，F−S2为{FS2:+.3f}pp，F−公开模型A为{s['F_minus_A_mean_pp']:+.3f}pp。{h2}
- 完整计费端到端口径下，F相对U有效信号利用率变化**{signal_improvement}**；三探测块方向均正：{s['all_three_signal_block_directions_positive']}。{h1}
- {h3} {next_step}

这是原官方train中后置隔离的**新开发子集**，不是GSM8K正式test；只能称项目记录未曝光，公开模型预训练是否见过未知。GSM8K正式test与SVAMP继续封存。

## 全部种子终点

{main_table}
U只有5个独立训练终点，在三个探测块的主配对中复用。F/R各15个实际训练终点，构成5个训练seed与3个选择块的交叉设计；15个终点不代表15次独立的候选池或筛选集合抽样。每个准确率的分母均为{N}，不能把15×{N}、rollout或completion数当作独立样本量；上述SD仅描述种子间离散，不是题目区间。

![全部终点](endpoint_accuracies.png)

{''.join(matrices)}
![全部配对差](paired_differences.png)

## 不确定性与种子敏感性

{interval_table}
以新开发近重复簇为抽样单位，对37个固定模型联合重采样{plan['statistics']['conditional_cluster_bootstrap_replicates']}次。每次使用完全相同的簇抽样权重，按抽中簇中实际题数重新计算分母；同批题和共享U的依赖得到保留。条件95%区间是百分位区间，训练／探测种子固定。主比较为F−U，其他比较是机制或诊断，没有把多个对比包装成相互独立的显著发现。

交叉bootstrap另重采样训练seed块、配对F_j/R_j选择块和题目簇；每次训练seed抽样中的U在全部抽中选择块中共享。题目簇重采样seed20261116，交叉块抽样seed20261118，均由冻结源码预先固定。该区间仅作探索性敏感性展示：训练seed只有5个、选择块只有3个，而且仍只有一个候选池，不能据此估计跨新池的普遍收益。

{sensitivity_table}
![条件区间](conditional_intervals.png)

## 相对共同S2：净新学会还是较少回归

{transition_table}
每份终点都有逐题S2正确→错误和错误→正确记录，二者之差严格等于该终点正确数相对S2的变化。不能把S2原本答对且训练后仍答对的题算作新学会。上表是固定评分器下的行为转移，不证明内部推理机制或预训练记忆变化。

把同seed的F−U正确数差拆开，来自S2原错题的新增判对差为{s['F_minus_U_behavioral_decomposition_pp']['additional_S2_wrong_then_correct_pp']:+.4f}pp，来自较少S2原对题回归的差为{s['F_minus_U_behavioral_decomposition_pp']['avoided_S2_correct_then_wrong_pp']:+.4f}pp，二者严格相加为主比较均值。这是行为账目分解；正值分别表示F比对应U多纠正题目或少回归题目，负值含义相反。[15格分解](paired_s2_transition_decomposition.csv)保留共享U对应关系，[完整转移ID](transition_ids_from_S2.json)支持离线逐题核查。

## 训练信号与优化器证据

{signal_table}
有效completion严格定义为实际TRL优势非零且实际completion loss mask非空，不能只按混合奖励组估算。全部原始奖励经原评分器复算；实际优势、EOS／截断mask、每步梯度有限性、冻结骨干无梯度、Adam状态与保存后LoRA相对S2的参数差均核对。零梯度Adam步仍保留，观察到{sigma}步在梯度为零时权重继续变化，不能将这些步当作没有优化器行为。

训练峰值PyTorch allocated/reserved为{peak:.3f}/{peak_reserved:.3f}GiB，属于进程统计，不是整机显存占用。Adam/RNG诊断文件保存，但没有验证精确sampler/dataloader断点续训，因此失败采用同配置、同S2的新attempt完整重试。

![信号效率](signal_efficiency.png)

{window_table}
窗口按每个终点已观测训练墙钟等分，表中比例按各窗口的实际组数聚合。全对、全错和混合组同时列出，避免仅凭零方差比例上升就判定模型已学会；生成长度和截断保留为吞吐及信号变化诊断。
按每终点实际训练时间分成首／中／末三个等长窗口，以组完成边界归属窗口。不同窗口的组数可能不同；仅绘制实际信号和覆盖，没有制造未评估的准确率学习曲线。后段信号变化详见逐终点CSV，不能由单个终点的波动推断所有固定筛选都已过时。

![信号时间窗](signal_time_windows.png)
![覆盖和集中度](coverage_concentration.png)

## 三次独立固定S2探测与随机机制对照

{probe_table}
F严格保留每题原始K4成功次数1/2/3，大小M_j全部来自真实输出。没有按约64题凑数、改K、换seed或补采。R用对应固定seed从完整256题均匀无放回抽同样M_j题，允许与F重叠；随后F/R与U均在各自固定池中均匀有放回采样，更新权重／动态补采事件为0。

{jaccard_table}
![探测与入选频率](probe_selection.png)
入选频率和Jaccard衡量本批K4探测的选择稳定性，不等同于模型能力或筛选因果效应。三个R块也只是三个固定随机子集，F−R不能推广为对所有随机子集都占优。

## 数据曝光、近重复与疑义审计

沿用锁定数据版本和原始manifest。核验四份已发布源码／证据ZIP的完整SHA256，并与checkout逐文件比对；未发现缺失归档运行证据，另存10份不同历史版本。曝光账本覆盖名义及实际SFT、GRPO、校准、探测、开发评估、全部保存预测、归档运行及保守历史提及。原validation512整批排除，未从中假设额外1024题。

共{data_audit['historically_referenced_or_configured_ids']}题有历史提及或配置证据；包括已知疑义题与近重复簇闭包在内排除{data_audit['blocked_ids_including_cluster_closure']}题。剩余{data_audit['eligible_rows']}题／{data_audit['eligible_clusters']}簇。沿用word(1,2) TF-IDF cosine≥0.90原审计边，并增加只用official train的独立重拟合检查（{data_audit['train_only_near_pairs']}对）；两套边和完全重复共同构簇，归一化完全重复0对。该规则不能排除低于阈值的语义模板相似，更不能证明预训练无污染。

在任何新探测前登记抽样seed20261114/20261115、候选顺序、资格和已知疑义清单。前283题经同一Agent质量规则审阅，排除27题，取首256个通过题。审阅保留题干、原解、计算检查、排除原因和Agent身份；`human_reviewed=false`。全部320题审阅／备用包也排除在新开发集之外。新开发按完整剩余簇冻结{N}题（{s['development_clusters']}簇），与新池、名义SFT128及所有历史曝光闭包隔离。

专用`fresh_development` manifest明确来源为原train后置隔离；旧split和原manifest未覆盖。普通train/select/develop不能返回新开发ID，正式test/external权限没有放宽。新开发评估必须携带全部35训练单元的不可变终点锁和具体对象key，加载前核验成功权重及基础模型/tokenizer文件。

## 成本、预算和失败

三次完整探测实测P_j后，按`B=max(3600,300*ceil(3*max(P_j)/300))`冻结**B={budget['B_seconds']}秒**。U使用B做自身加载、训练、采样维护和保存；F_j与R_j使用B−P_j。B从父进程完整源码／模型／tokenizer缓存核验前起计，外层时间戳传入子进程预算钟，包含解释器启动、加载、训练和保存；旧题两步真实集成检查验证了该传递。初始评估预留登记后，为提高37对象完整评估可靠性，在主训练／新开发前保存前瞻修订：E取旧64题外推耗时2倍加300秒后取整。最终**E={budget['E_seconds']}、O={budget['O_seconds']}、C={budget['C_seconds']}秒**。B仅由时间决定，评估余量不回填训练。

每个F终点扣相应**完整P_j**；R的同额扣减仅作实验配平，不能声称随机选题在实际应用中必然需要完整探测，也不能推断没有运行的“R不扣探测并训练更久”的结果。逐终点账本分开记录算法探测扣额和R配平；共享探测物理上各执行一次，不把15个F扣额当作15次真实探测，也不把R配平项再加进会话物理成本。

- 完整退出收据记录的模型进程合计{costs['measured_owned_model_process_seconds']/3600:.3f}小时；另有{costs['interrupted_process_count']}个中断进程，耗时界定为{costs['interrupted_model_process_seconds_lower_bound']:.3f}—{costs['interrupted_model_process_seconds_upper_bound']:.3f}秒。全部模型尝试物理进程成本为{costs['all_attempt_model_process_seconds_lower_bound']/3600:.3f}—{costs['all_attempt_model_process_seconds_upper_bound']/3600:.3f}小时，包括定标、共享探测、训练、独立评估和失败；中断进程没有被按0计费，未知退出码保持未知。
- 共享探测物理时长（含探测失败，如有）：{costs['shared_probe_physical_seconds_including_probe_failures']/3600:.3f}小时。失败模型进程总成本{costs['failed_model_process_seconds_lower_bound']:.3f}—{costs['failed_model_process_seconds_upper_bound']:.3f}秒，分为{costs['completed_failed_process_count']}个完整失败收据和{costs['interrupted_process_count']}个中断。另有{costs['controller_preparation_failures_with_preserved_source']}次保留源码/日志的控制器准备错误，其时间计入会话墙钟，不冒充GPU活跃时间。
- F逐终点完整探测扣额之和：{costs['F_standalone_probe_debit_seconds_sum_not_extra_shared_executions']:.3f}秒；R实验配平项之和：{costs['R_matching_debit_seconds_sum_not_extra_physical_cost']:.3f}秒。两者不能重复加到物理账本。
- R三次实际子集抽样合计{costs['R_subset_draw_physical_seconds']:.9f}秒；各端点自身manifest核验／读取计入各自训练进程。
- 本次分析时从登记起的会话墙钟{costs['registered_session_elapsed_wall_seconds_at_analysis']/3600:.3f}小时，覆盖准备、推理、训练、失败、系统重启停机、用户请求暂停和分析间隔。最终交接另写`session_final.json`。登记前初始定位没有完整进程计时，只保留最早确认时钟，不能伪称精确GPU活跃时间。未计量本机电费；没有租云GPU或调用项目外付费模型API。

共同上限并不表示实际耗时完全相等。`endpoints.csv`逐项列出自然边界训练超额、端到端超额及未用额度；加载、缓存检查、保存、独立评估及父进程核验均进入计费端到端秒。全会话硬上限和每失败单元一次完整重试预留在主训练前冻结，未因新开发成绩增加。

冻结主信号指标的`charged_end_to_end_seconds`是保留的首次成功终点路径扣额（含完整P/R配平），并列的`charged_end_to_end_including_failed_attempts_seconds_*_bound`加入该终点全部失败路径的已知成本界，不把丢弃权重中的训练信号计入分子。包含失败后F相对U的有效信号率提高范围为{operational_gain_text}；这是成本不确定性的敏感性范围，不是统计置信区间，也不替换冻结主决策。系统重启或重试造成的操作成本差不能归因于成功率筛选。跨单元排队与用户暂停进入实际会话墙钟，不能重复计作每个终点的GPU活跃时间。

本轮U-s47-a0在系统重启前有原始活动时间戳，首个后续OS启动给出进程存活上界，完整证据见[中断成本审计](../../../experiments/{family}/interruption_cost_audit.json)。其唯一完整重试a1从同一S2/配置重新开始；用户随后要求等待保存后暂停，观察器保留真实退出码/时间并在终点核验后停止控制器。继续时从下一单元开始，未中途恢复sampler或optimizer。系统重启/升级、暂停与恢复均有记录；即使模型、数据与代码哈希一致，不能把整段会话描述为未经外部中断的相同系统状态。

![成本组成](cost_breakdown.png)

## 盲化案例审阅与已知问题

按预登记固定训练seed29、三个探测块，审阅F/U、F/R、F/S2每一对的全部不一致题，并用固定seed20261117分别抽16个共同判对和16个共同判错样例。方法名、比较名及自动正误标签在审阅包中隐藏，回答别名和顺序随机；原始参考解与金标保持不变。

本轮审阅{review['completed_cases']}个案例／{review['reviewed_responses']}份回答展示，覆盖{review['unique_question_ids']}个唯一问题；待审阅案例{review['pending_cases']}。{review_dedup_note} 审阅者为Agent，未标作人工审阅。分类区分读题／建模、计算、解析、截断、推导支持的正确、只给答案而无法核验推导、推导不支持但数值碰巧正确、疑义和其他，详见`blind_review_summary.json`和逐例JSONL。其他训练seed的未审阅数值错误保留“待审阅”，不自动叫算术错。案例抽样不改变主要评分、金标、训练或检查点。

主要局限为单一候选池、3个探测／随机选择块、5个训练seed、K4的抽样噪声、有限近重复规则和公开预训练污染未知。固定时间训练也可能有自然原子边界和设备状态差异，实际成本与超额均保留。开发题准确率提升即使跨零被排除，也仍不等于正式测试或广泛泛化证明。

## 下一步

{next_step} 若未来出现可复现后段信号枯竭或筛选滞后，再另立动态过滤消融；本家族不加入动态刷新、奖励塑形、新学习率或跳过零梯度Adam步。

## 证据、检查点和复算

主证据入口：[结构化summary](summary.json)、[全部37对象逐终点CSV](endpoints.csv)、[15格配对差](paired_differences.csv)、[20k次联合簇重采样原始数组](joint_cluster_bootstrap.npz)、[逐题覆盖](training_question_coverage.csv)、[信号时间窗](training_signal_windows.csv)、[S2行为转移](transitions_from_S2.json)、[含中断行的进程物理账本](process_cost_ledger.csv)、[中断成本界](interrupted_process_cost_bounds.csv)、[逐终点失败路径成本](failed_attempt_costs_by_endpoint.json)、[证据审计](evidence_audit.json)、[盲化审阅汇总](blind_review_summary.json)。

冻结与原始运行：[协议](../../../experiments/{family}/protocol_frozen.json)、[终点权重锁](../../../experiments/{family}/all_training_endpoints_fixed.json)、[模型任务完成](../../../experiments/{family}/all_model_tasks_complete.json)、[三次探测](../../../experiments/{family}/all_probes_complete.json)、[历史审计](../../../experiments/{family}/history_audit.json)、[逐ID曝光](../../../experiments/{family}/history_exposure_ledger.jsonl)、[原题质量审阅](../../../experiments/{family}/candidate_review_completed.jsonl)。全部LoRA与Adam/RNG诊断在项目`checkpoints/`，逐路径和哈希见终点锁及最终工件清单。

从项目内独立环境执行：

```powershell
. .\scripts\desktop_env.ps1
$py = '.\.venv\Scripts\python.exe'
$family = '{family}'
& $py analysis/analyze_large_filter.py --family $family
& $py analysis/render_large_filter.py --family $family
& $py analysis/write_large_filter_report.py --family $family
```

离线复算不生成新模型输出，不重训，也不修改冻结评分器／金标。详细命令和未覆盖失败的约束见[复现说明](../../../docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。源码、配置、数据、模型、检查点及原始工件的完整SHA256在家族最终manifest内；本轮只准备本地可审阅变更，未远端发布。
"""
    (out / "findings.md").write_text(text, encoding="utf-8", newline="\n")
    json_write(exp / "report_written.json", {"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report": (out / "findings.md").relative_to(ROOT).as_posix(), "sha256": sha256_file(out / "findings.md"),
        "all35_training37_evaluation": True, "blinded_agent_review_completed": True, "remote_published": False})
    print(str(out / "findings.md"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
