"""Write the local Chinese findings when the registered matrix has failed cells."""
from collections import Counter
import csv
import datetime as dt
import statistics

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file
from large_filter_common import read, PROBE_SEEDS, TRAIN_SEEDS
from write_large_filter_report import pct, pp, table


def bounds(values):
    return f"[{values[0]:+.3f}, {values[1]:+.3f}] pp" if values is not None else "缺失"


def rate_text(row, key="effective_completions_per_second"):
    return f"{row[key]:.5f}" if row else "缺失"


def relative(f, u, key="effective_completions_per_second"):
    return f"{f[key]/u[key]-1:+.2%}" if f and u and u[key] else "未定义"


def main(family):
    exp = ROOT / "experiments" / family
    out = ROOT / "reports/comparisons" / family
    summary = read(out / "summary.json")
    review = read(out / "blind_review_summary.json")
    assert review["pending_cases"] == 0 and review["human_reviewed"] is False
    assert summary["original_full_matrix_inference_available"] is False and summary["successful_training"] == 35
    assert summary["development_evidence_threshold_met"] is None and summary["engineering_signal_threshold_met"] is None
    assert read(out / "evidence_audit.json")["all_scores_recomputed_with_unchanged_scorer"]
    plan = read(exp / "protocol_frozen.json")
    history = read(exp / "history_audit.json")
    episodes = read(exp / "evaluation_units.json")
    n = summary["n"]
    endpoints = {r["endpoint"]: r for r in summary["endpoints"]}
    comparisons = summary["available_pair_comparisons"]
    fu = comparisons["F-U"]["available"]
    fr = comparisons["F-R"]["available"]
    s2 = comparisons["F-S2"]["available"]
    sa = comparisons["F-A"]["available"]
    signal = summary["signal_sensitivity"]
    total_signal = signal["all35_charged_training_only"]
    complete_signal = signal["complete_seed_block_end_to_end"]
    costs = summary["physical_costs"]
    missing = summary["missing_evaluation_objects"]
    evaluated = summary["successful_evaluation"]
    local = dt.timezone(dt.timedelta(hours=8))
    deadline = dt.datetime.fromtimestamp(plan["session_limit"]["deadline_unix"], local)
    model_done = dt.datetime.fromisoformat(read(exp / "all_model_tasks_complete.json")["completed_utc"]).astimezone(local)
    budget = plan["budget"]
    limit = plan["session_limit"]
    base = f"../../../experiments/{family}"
    missing_table = table(["评估对象", "尝试次数", "失败标签", "结果"], [
        [key, len(episodes[key]["attempts"]), " / ".join(a["label"] for a in episodes[key]["attempts"]), "缺失"] for key in missing])

    endpoint_rows = []
    for arm, j in [("U", None)]+[(arm, j) for j in PROBE_SEEDS for arm in ("F", "R")]:
        selected = [endpoints[f"U-s{s}" if j is None else f"{arm}-j{j}-s{s}"] for s in TRAIN_SEEDS]
        correct = ["缺失" if r["correct"] is None else str(r["correct"]) for r in selected]
        endpoint_rows.append([arm if j is None else f"{arm} / {j}", *correct, f"{sum(r['correct'] is not None for r in selected)}/5"])
    endpoint_table = table([f"正确数，每份分母 {n}", *map(str, TRAIN_SEEDS), "完整评估"], endpoint_rows)
    pair_tables = []
    interval_rows, seed_rows, driver_rows = [], [], []
    for label, comparison in comparisons.items():
        a = comparison["available"]
        interval_rows.append([label, f"{a['pairs']}/15", pp(a["mean_pp"]), bounds(a["conditional_percentile95_pp"]),
            bounds(comparison["full_matrix_sensitivity"]["bounds_pp"])])
        for block_type, field in (("探测", "by_probe_seed"), ("训练", "by_training_seed")):
            if label not in ("F-U", "F-R"):
                continue
            for seed, stat in comparison[field].items():
                seed_rows.append([label, block_type, seed, stat["pairs"], pp(stat["mean_pp"]), bounds(stat["range_pp"]),
                    f"{stat['sd_pp']:.3f} pp" if stat["sd_pp"] is not None else "—"])
        for same, name in (("same_driver", "同版本"), ("different_driver", "跨版本")):
            stat = comparison[same]
            driver_rows.append([label, name, stat["pairs"], pp(stat["mean_pp"])])
        matrix = comparison["matrix_pp_probe_by_training"]
        if label in ("F-U", "F-R"):
            pair_tables.append(f"### {label} 的配对正确率差\n\n"+table(["训练种子", *map(str, PROBE_SEEDS)],
                [[seed, *(pp(matrix[i][k]) for i in range(len(PROBE_SEEDS)))] for k, seed in enumerate(TRAIN_SEEDS)]))
    interval_table = table(["比较", "有完整预测的配对", "可观测平均差", "条件 95% 区间", "完整矩阵极值界"], interval_rows)
    seed_table = table(["比较", "种子类型", "种子", "配对数", "平均差", "最小与最大", "描述性 SD"], seed_rows)
    driver_table = table(["比较", "评估驱动", "配对数", "平均差"], driver_rows)
    pair_tables_md = "\n\n".join(pair_tables)

    transitions = summary["transitions_from_S2"]
    transition_rows = []
    for arm in ("U", "F", "R"):
        rows = [transitions[key] for key in transitions if endpoints[key]["arm"] == arm]
        nums = [statistics.mean(r[c] for r in rows) for c in ("S2_correct_then_wrong", "S2_wrong_then_correct", "net_correct_difference")]
        transition_rows.append([arm, len(rows), *(f"{v:.2f}" for v in nums)])
    transition_table = table(["方法", "完整终点", "S2 正确→错误平均题数", "S2 错误→正确平均题数", "净正确数变化"], transition_rows)
    with (out / "paired_s2_transition_decomposition.csv").open(encoding="utf-8-sig") as f:
        decomposition = list(csv.DictReader(f))
    assert len(decomposition) == fu["pairs"]
    learned, avoided = (statistics.mean(float(r[k]) for r in decomposition) for k in (
        "additional_S2_wrong_then_correct_pp", "avoided_S2_correct_then_wrong_pp"))
    fu_signs = Counter((v > 0)-(v < 0) for row in comparisons["F-U"]["matrix_pp_probe_by_training"] for v in row if v is not None)
    matched_seed_count = len(signal["complete_end_to_end_training_seed_blocks"])
    e2e_lift = relative(complete_signal["F"], complete_signal["U"])
    e2e_gradient_lift = relative(complete_signal["F"], complete_signal["U"], "nonzero_gradient_steps_per_second")

    diagnostics = summary["training_signal"]
    signal_rows = []
    for arm in ("U", "F", "R"):
        ds = [diagnostics[key] for key in diagnostics if endpoints[key]["arm"] == arm]
        steps = sum(d["optimizer_steps"] for d in ds)
        zero = sum(d["all_correct_groups"]+d["all_wrong_groups"] for d in ds)
        groups = zero + sum(d["mixed_groups"] for d in ds)
        r = total_signal[arm]
        signal_rows.append([arm, len(ds), steps, f"{zero/groups:.2%}", r["effective_completions"],
            rate_text(r), r["nonzero_gradient_steps"], rate_text(r, "nonzero_gradient_steps_per_second")])
    signal_table = table(["方法", "训练终点", "优化步", "零方差组", "有效 completion", "有效 completion/计费训练秒", "非零梯度步", "非零梯度步/计费训练秒"], signal_rows)
    matched_rows = []
    for j, block in signal["matched_end_to_end_by_probe"].items():
        for comp, rates in block.items():
            base_arm = comp[2:]
            matched_rows.append([j, comp, "/".join(map(str,rates["training_seeds"])), rate_text(rates["F"]),
                rate_text(rates[base_arm]), relative(rates["F"], rates[base_arm]),
                rate_text(rates["F"], "nonzero_gradient_steps_per_second"), rate_text(rates[base_arm], "nonzero_gradient_steps_per_second")])
    matched_table = table(["探测种子", "比较", "有完整计费的配对训练种子", "F 有效 completion/秒", "对照/秒", "F 相对变化", "F 非零梯度步/秒", "对照/秒"], matched_rows)
    pre_blocks = signal["all35_training_only_by_probe_seed"]
    training_directions = "、".join(relative(pre_blocks[str(j)]["F"], total_signal["U"]) for j in PROBE_SEEDS)
    with (out / "training_signal_windows.csv").open(encoding="utf-8-sig") as f:
        windows = list(csv.DictReader(f))
    window_rows, window_effective = [], {}
    for arm in ("U", "F", "R"):
        for window, name in (("first", "前"), ("middle", "中"), ("last", "后")):
            rs = [r for r in windows if r["arm"] == arm and r["window"] == window]
            groups = sum(int(r["groups"]) for r in rs)
            shares = [sum(int(r[field]) for r in rs)/groups for field in ("all_correct_groups", "all_wrong_groups", "mixed_groups")]
            items = sum(int(r["effective_completions"]) for r in rs)
            window_effective[arm, window] = items/(4*groups)
            window_rows.append([arm, name, groups, *(f"{v:.2%}" for v in shares), f"{window_effective[arm, window]:.2%}"])
    window_table = table(["方法", "时间窗", "组数", "全对", "全错", "混合", "有效 completion 比例"], window_rows)
    coverage_rows = []
    for arm in ("U", "F", "R"):
        ds = [r for r in endpoints.values() if r["arm"] == arm]
        coverage = [r["training_distinct_training_ids"] for r in ds]
        top10 = [r["training_effective_top10_question_share"] for r in ds]
        coverage_rows.append([arm, f"{min(coverage)}–{max(coverage)}", f"{min(top10):.2%}–{max(top10):.2%}"])
    coverage_table = table(["方法", "单个终点实际覆盖题数", "单个终点前十题的有效输出占比"], coverage_rows)
    peak = max(d["peak_allocated_bytes"] for d in diagnostics.values())/2**30
    reserved = max(d["peak_reserved_bytes"] for d in diagnostics.values())/2**30
    adam = sum(d["zero_gradient_adam_changed_steps"] for d in diagnostics.values())

    probes = summary["probe_summary"]
    probe_table = table(["探测种子", "0 / 1 / 2 / 3 / 4 次成功题数", "M", "完整探测秒", "随机种子", "F∩R"], [
        [r["probe_seed"], " / ".join(str(r[f"raw_success_{i}_items"]) for i in range(5)), r["M"],
        f"{r['full_process_seconds']:.2f}", r["random_subset_seed"], r["R_overlap_with_F"]] for r in probes])
    jacc = read(exp / "selection_diagnostics.json")["F_jaccard"]
    jaccard = "、".join(f"{jacc[f'{j}/{k}']:.3f}" for j,k in ((PROBE_SEEDS[0], PROBE_SEEDS[1]), (PROBE_SEEDS[0], PROBE_SEEDS[2]), (PROBE_SEEDS[1], PROBE_SEEDS[2])))
    stage_names = {"calibration_and_recovery_checks": "准备、定标和恢复检查", "shared_probes": "共享探测", "training": "训练含失败", "fresh_development_evaluation": "新开发评估含失败"}
    cost_table = table(["阶段", "模型与运行检查进程数", "实际进程时长下界（小时）", "上界（小时）"], [
        [stage_names[key], values["attempts"], f"{values['seconds_lower_bound']/3600:.4f}", f"{values['seconds_upper_bound']/3600:.4f}"]
        for key, values in costs["process_by_stage"].items()])
    overshoot = [r["end_to_end_cap_overshoot_seconds"] for r in endpoints.values() if r["end_to_end_cap_overshoot_seconds"] is not None]
    retry_overshoot = [r["failure_inclusive_C_overshoot_seconds_upper_bound"] for r in endpoints.values()
        if r["charged_end_to_end_seconds"] is not None]
    pre_overshoot = [r["training_wall_budget_overshoot_seconds"] for r in endpoints.values() if r["training_wall_budget_overshoot_seconds"] is not None]
    driver_counts = "、".join(f"{version} 共 {count} 份" for version, count in summary["across_driver_evaluation_counts"].items())

    scope_counts = review["case_counts_by_scope"]
    review_counts = table(["盲化范围", "案例数"], [
        ["原定种子 29 的可用配对", scope_counts.get("registered_seed29", 0)],
        ["补充同驱动 F/U 配对", scope_counts.get("exploratory_same_driver_seed13", 0)]])
    categories = {"supported_correct": "推导支持正确答案", "correct_without_checkable_reasoning": "答案正确但推导不可核验",
        "read_or_modeling_error": "读题或建模", "calculation_error": "计算", "parsing_error": "答案解析",
        "truncated": "截断", "lucky_numeric_correct": "推导有误但数值碰巧正确", "uncertain": "无法确定", "other": "其他"}
    review_responses = jsonl_read(out / "blind_review_unblinded_responses.jsonl")
    by_category = Counter(r["category"] for r in review_responses)
    examples = []
    for category in ("read_or_modeling_error", "calculation_error", "parsing_error", "truncated", "lucky_numeric_correct"):
        matches = sorted((r for r in review_responses if r["category"] == category), key=lambda r: (r["id"],r["case_id"]))
        if matches:
            r = matches[0]
            examples.append([categories[category], r["id"], r["endpoint_key"], r["notes"].replace("|", "／").replace("\n", " ")])
    example_table = table(["人工智能代理判别", "题目 ID", "解盲对象", "可核查理由"], examples) if examples else "样例见逐响应审阅记录。"
    category_table = table(["类别", "响应展示次数"], [[categories[k], by_category[k]] for k in categories])
    unique_annotations = review["exact_text_deduplication"]["unique_response_units_actually_annotated"]
    flags = jsonl_read(out / "blind_review_question_gold_flags.jsonl")
    flagged_questions = len({r["id"] for r in flags})

    report = f"""# 256 题固定预筛复验与同规模随机子集消融

本地家族 `{family}`。35/35 个训练终点已复核，{evaluated}/37 个对象完成同一批 {n} 道新开发题的独立评估。F 为固定 S2/K4 预筛，U 为完整训练池均匀采样，R 为同规模随机子集。

**在端到端记录齐全的 {matched_seed_count} 个训练种子整块中，F 的有效 completion/计费端到端秒相对 U 变化 {e2e_lift}。新开发准确率 F−U 在 {fu['pairs']}/15 个可观测配对中平均 {pp(fu['mean_pp'])}，固定模型和种子的条件 95% 区间为 {bounds(fu['conditional_percentile95_pp'])}。** {len(missing)} 个对象因运行故障耗尽重试，原定完整矩阵的准确率和端到端工程阈值无法判定。

F−R 在 {fr['pairs']}/15 个可观测配对中平均 {pp(fr['mean_pp'])}，条件区间 {bounds(fr['conditional_percentile95_pp'])}。F−S2 为 {pp(s2['mean_pp'])}，F−公开模型 A 为 {pp(sa['mean_pp'])}。可用配对中，F 对同规模 R 或起点 S2 的准确率优势均缺少证据。 [比较与区间](paired_differences.csv)

35 条计费训练路径汇总后，F 的有效 completion/秒比 U 变化 **{relative(total_signal['F'],total_signal['U'])}**。三个探测块相对同批 U 的训练口径分别为 {training_directions}。这些值包含完整探测扣额，适用于已完成训练的信号分析。完整评估的端到端效率和缺失范围另见下文。 [信号与成本](endpoints.csv)

新开发集来自 GSM8K 官方 train 的后置隔离子集。项目记录未显示这些题被曝光，公开模型预训练是否见过仍未知。GSM8K 正式 test 和 SVAMP 保持封存。

## 全部种子和失败对象

A 为 {endpoints['A']['correct']}/{n}（{pct(endpoints['A']['accuracy'])}），共同 S2 为 {endpoints['S2']['correct']}/{n}（{pct(endpoints['S2']['accuracy'])}）。两者使用冻结权重。新开发集首次开启前已锁定 35 个训练终点和这两个锚点。 [权重锁]({base}/all_training_endpoints_fixed.json)

{endpoint_table}

{missing_table}

这 {len(missing)} 个缺口保留原始失败日志、退出码和已写出的部分行。完整预测与部分预测在评估账本中分别登记。11 月 13 日发生原生评估进程异常和后续 HIP 错误，当时 Windows 无法枚举 RX 9070 XT。根因仍未确认。用户清理并重装驱动后，旧开发 64 题的 S2 输出与原文件 SHA256 一致。用户曾在已保存终点的边界暂停。11 月 18 日晚从 28 份恢复，按冻结顺序完成最后四项，于北京时间 {model_done:%Y-%m-%d %H:%M:%S} 保存最后一项。 [故障审计](gpu_runtime_blocked_status.md) · [重装与暂停记录](driver_recovery_check_20261113.md) · [逐对象尝试]({base}/evaluation_units.json) · [任务结束收据]({base}/all_model_tasks_complete.json)

![全部注册种子的终点评估](endpoint_accuracies.png)

{pair_tables_md}

![全部 F−U 和 F−R 配对格](paired_differences.png)

## 区间、缺失敏感性与环境变化

{interval_table}

以 {summary['development_clusters']} 个新开发题近重复簇为抽样单位，使用种子 20261116 对所有已得到的固定模型联合重采样 {plan['statistics']['conditional_cluster_bootstrap_replicates']} 次。每次在相同簇样本上重算各模型正确率，再求可观测配对的平均差。每个训练种子的共同 U 在三个选择块中共享同一次抽样结果。训练和探测种子固定，且只有一个候选池。完整矩阵极值界允许每个缺失模型的正确率在 0 到 100% 间变化，按其在 15 格的实际权重计算，反映缺失可导致的最大幅度。原定完整矩阵保留为缺失，本次未计算交叉 bootstrap。 [缺失分析口径]({base}/missing_evaluation_analysis_plan_20261118.json) · [联合重采样数组](joint_cluster_bootstrap.npz)

![可观测配对的条件区间](conditional_intervals.png)

{seed_table}

F−U 的 {fu['pairs']} 个可观测配对中，{fu_signs[1]} 个为正、{fu_signs[-1]} 个为负、{fu_signs[0]} 个持平。训练种子 13 和 47 的可观测均差为正，71 和 97 为负，种子 29 无完整 F/U 配对。三个探测块的 F−U 均差均为负。少量种子及单一候选池限制了跨运行推断。

{driver_table}

完整预测按驱动登记为 {driver_counts}。驱动版本与实际评估顺序及种子重合。同版本子集是敏感性检查，无法独立识别驱动影响。旧 64 题重放的结果仅覆盖共同 S2 和该检查用集合。 [端点与驱动](endpoints.csv) · [驱动分段记录]({base}/evaluation_driver_segments.jsonl)

## 共同 S2 上的新增判对与回归

{transition_table}

同一批可观测 F/U 配对中，F−U 的 {pp(fu['mean_pp'])} 可拆为新增判对差 {pp(learned)} 和少回归贡献 {pp(avoided)}。F 的新增判对数接近 U，回归题略多。F−S2 的条件区间包含零，未观察到明确的净能力增长。缺失对象无转移记录。 [逐题转移 ID](transitions_from_S2.json) · [逐配对分解](paired_s2_transition_decomposition.csv)

## 实际梯度信号、覆盖与训练时间窗

{signal_table}

计费训练秒包含自身加载、训练、保存、父进程核验与应计的 F 探测或 R 配平时间。有效 completion 要求实际优势非零且实际 loss mask 非空。核验覆盖逐组奖励、实际优势和掩码、每步梯度有限性、冻结骨干无梯度、Adam 状态以及保存后 LoRA 相对 S2 的参数变化。实际有 {adam} 个零梯度 Adam 步仍改变了权重。训练峰值 allocated/reserved 为 {peak:.3f}/{reserved:.3f} GiB。Adam/RNG 文件供审计，sampler 和 dataloader 的精确续训未验证。 [全部训练诊断](training_diagnostics.json)

{matched_table}

七条端到端路径均成功的训练种子为 {'、'.join(map(str,signal['complete_end_to_end_training_seed_blocks'])) or '无'}。在这些整块内，U、F、R 的有效 completion/计费端到端秒分别为 {rate_text(complete_signal['U'])}、{rate_text(complete_signal['F'])}、{rate_text(complete_signal['R'])}，F 相对 R 变化 {relative(complete_signal['F'],complete_signal['R'])}。F 的非零梯度步/计费端到端秒相对 U 变化 {e2e_gradient_lift}。上表按每个探测块各自可配对的种子计算。缺失评估的端到端时间保留空值。 [逐端点成本与信号](endpoints.csv)

![端到端有效信号](signal_efficiency.png)

{window_table}

窗口按每个终点的实际训练时长三等分，按组完成时间归属，汇总比例以实际组数为分母。F 的有效比例从 {window_effective['F','first']:.2%} 降到 {window_effective['F','last']:.2%}，后段仍高于 U 的 {window_effective['U','last']:.2%} 和 R 的 {window_effective['R','last']:.2%}。在本轮实际训练时长内，未见 F 信号枯竭。

{coverage_table}

F 与 R 基本采遍各自的训练子集，F 的有效输出比同规模 R 更分散。逐终点表记录实际生成长度与 token、解析和格式失败、截断、覆盖题数及重复次数。 [时间窗](training_signal_windows.csv) · [逐题覆盖](training_question_coverage.csv)

![实际训练信号时间窗](signal_time_windows.png)
![训练题覆盖与集中度](coverage_concentration.png)

## 三次探测与数据隔离

{probe_table}

固定 S2 每题 K=4，只留原始成功次数 1、2、3。R 从共同 256 题无放回抽同一 M，允许与 F 重叠。三个 F 集合两两 Jaccard 依次为 {jaccard}。U/F/R 各自固定题池内均匀有放回训练。 [原始探测与进程收据]({base}/all_probes_complete.json) · [逐题集合频率]({base}/selection_diagnostics.json)

![K4 成功数和入选频率](probe_selection.png)

历史账本核验四个已发布源码和证据归档的 SHA256，逐文件比对并保留不同历史版本。覆盖名义及实际训练、校准、探测、开发评估和留存预测。历史记录有 {history['historically_referenced_or_configured_ids']} 个题目 ID，加上疑义题及近重复簇闭包共排除 {history['blocked_ids_including_cluster_closure']} 题，剩余 {history['eligible_rows']} 题。原 validation 512 整批排除。近重复沿用 word(1,2) TF-IDF cosine≥0.90，合并原记录和官方 train 重拟合边并检查归一化完全重复。 [历史审计]({base}/history_audit.json) · [逐题历史账本]({base}/history_exposure_ledger.jsonl) · [簇与成员]({base}/clusters.jsonl)

收尾时从已核验的发布归档恢复原始 GSM8K train 文件，SHA256 与原始 manifest 一致。按官方顺序核查了原 train 与 validation 的题号、题干及参考解。恢复过程只提取训练源文件。 [原始训练数据恢复收据]({base}/raw_official_train_source_recovery_20261119.json)

数据抽样种子 20261114 和 20261115 在新探测前登记。Agent 按冻结顺序审核候选题干和参考解，首 283 题中排除 27 题，取首批 256 个合格题。原题、金标、审阅身份与理由均保留，已登记疑义题及关联簇先排除。新开发集按完整簇冻结，隔离于候选池、共同 S2 和历史曝光。专用入口要求权重锁和对象 ID，原划分仍可按原用途使用。 [候选审核]({base}/candidate_review_completed.jsonl) · [候选池]({base}/candidate_pool.json) · [新开发 manifest](../../../data/fresh_development/{family}/manifest.json)

## 盲化 Agent 审阅

{review_counts}

审了所列可用配对的所有判对不一致题，并按登记种子每对抽取最多 16 道共同判对和 16 道共同判错题。原定训练种子 29 的 {len(review['unavailable_registered_pairs'])}/{review['registered_pair_count']} 个比较缺预测，缺口保留。补充种子 13 的固定 F/U 同驱动配对在缺失方案中单列。题目、参考解和回答以匿名标签交给 Agent，{unique_annotations} 份唯一原文响应逐份判断并固定注释哈希后解盲。审阅由 Agent 完成，`human_reviewed=false`。相同响应的重复展示保留映射。 [审阅范围、分层计数和冲突](blind_review_summary.json)

{category_table}

{example_table}

上表是按分层抽样后看到的案例。类别计数包含同一响应的重复展示，适用于定位错误类型。未审阅的数值错误不归为计算错误。前期运维界面显示过 S2 聚合正确数及 seed47 的部分题目 ID 和判对标记，匿名语义审阅未读该日志的响应原文。Agent 在已审案例中标出 {flagged_questions} 道有疑义或参考解笔误的题，合计 {len(flags)} 条案例标注。主结果沿用冻结评分器和金标。 [盲评细节](blind_review_unblinded_responses.jsonl) · [题干或参考答案疑义](blind_review_question_gold_flags.jsonl)

## 实际成本与原预算

三次完整探测实测决定 B={budget['B_seconds']} 秒。旧开发 64 题的独立评估用时 {budget['eval_old64_process_seconds']:.2f} 秒，按 N 线性外推为 {budget['eval_newN_linear_estimate_seconds']:.2f} 秒。主训练前登记保守余量，取外推的两倍加 300 秒再向上取整到 300 秒，冻结 E={budget['E_seconds']}、O={budget['O_seconds']}、C={budget['C_seconds']} 秒。评估未用额度不回填训练。全会话硬上限 {limit['total_session_hard_seconds']/3600:.2f} 小时，其中每个单元预留一次完整重试，共 {limit['failure_reserve_one_full_attempt_for_every_training_and_evaluation_unit_seconds']/3600:.2f} 小时。 [冻结预算]({base}/protocol_frozen.json) · [评估预留登记]({base}/evaluation_reserve_amendment_before_main.json)

F 每个终点扣完整 P_j，R 的同额扣除服务于相同可训练时间。三次共享探测实际总耗时 {costs['shared_probe_physical_seconds_including_probe_failures']/60:.2f} 分钟。各终点的 F 探测扣额合计 {costs['F_standalone_probe_debit_seconds_sum_not_extra_shared_executions']/3600:.3f} 小时，R 实验配平合计 {costs['R_matching_debit_seconds_sum_not_extra_physical_cost']/3600:.3f} 小时。三次 R 抽样实际 CPU 时间合计 {costs['R_subset_draw_physical_seconds']:.6f} 秒。R 获得完整 B 训练时长的效果需要单独实测。 [探测耗时](probe_summary.csv)

{cost_table}

全部尝试的物理进程时长为 **{costs['all_attempt_model_process_seconds_lower_bound']/3600:.3f}–{costs['all_attempt_model_process_seconds_upper_bound']/3600:.3f} 小时**，含失败时长 {costs['failed_model_process_seconds_lower_bound']/60:.2f}–{costs['failed_model_process_seconds_upper_bound']/60:.2f} 分钟。界限来自 U-s47-a0 重启中断时的未知精确退出。评估失败路径另有父进程核验界，U-s47 首次完整成功来自唯一一次重试。注册到统计出表的实际墙钟为 {costs['registered_session_elapsed_wall_seconds_at_analysis']/3600:.2f} 小时，包含项目内准备、用户暂停、重启和等待。会话硬截止为北京时间 {deadline:%Y-%m-%d %H:%M:%S}。有完整评估的路径中，成功路径和计入失败成本后的最大 C 超额分别为 {max(overshoot):.2f} 秒和 {max(retry_overshoot):.2f} 秒，训练自然边界最大超额为 {max(pre_overshoot):.2f} 秒。 [全部尝试账本](process_cost_ledger.csv) · [失败端点路径界](failed_attempt_costs_by_endpoint.json) · [中断成本界](interrupted_process_cost_bounds.csv)

![成功路径计费与共享探测](cost_breakdown.png)

## 实验判断与复算

前瞻开发阈值是完整 F−U 至少 +1 pp 且条件区间下界大于零。工程阈值基于完整端到端信号提高至少 20% 且三个探测块方向均正。完整矩阵的两项决策均因缺失留空。现有配对复现了更多有效训练信号，准确率差在各训练种子间变号，与同规模 R 的条件区间也包含零。可用结果不足以把预筛作为提分手段推进。下一步优先核查探测时成功率和训练奖励为何没有转化为新题上的正确率，以及共同 S2 原本判对的题在哪些情况下回归。另行登记第二独立候选池前，先验证长评估在当前设备上可稳定完成。

上一轮 64 题池的 F−U 为 +1.5625 pp，条件区间也包含零。其端到端有效 completion/秒提高 72.45%。两轮都看到信号效率提高的方向，各自的开发题、池规模和训练预算不同。 [上一轮报告](../desktopfilter_20261110/findings.md)

复算在项目独立环境内运行，使用锁定数据、源码、权重、原始输出及已固定审阅注释。全部 35 个检查点及其他尝试的工件保存在项目内。最终按文件登记 SHA256，正式保留集仍封存。本轮交付为可审阅的本地变更。 [复现命令](../../../docs/REPRODUCING_FILTER_REPLICATION_20261110.md) · [结构化汇总](summary.json) · [证据审计](evidence_audit.json) · [图表检查]({base}/figure_visual_audit_20261119.json)
"""
    path = out / "findings.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    json_write(exp / "report_written.json", {"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "all35_training37_evaluation": False,
        "successful_training": 35, "successful_evaluations": evaluated, "missing_evaluation_objects": missing,
        "blinded_agent_review_completed": True, "complete_registered_blind_scope": review["complete_registered_scope"],
        "remote_published": False})
    print(str(path))
