"""Populate the final Chinese handoff only after every fixed confirmation has ended."""
import datetime as dt
import json
import time
from collections import Counter
from pathlib import Path
from reasoning_rl.runtime import ROOT,json_write,jsonl_read,code_hash,sha256_file

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"reports/comparisons"/FAMILY
MARK_START="<!-- REPAIR_20261106_CURRENT_BEGIN -->"
MARK_END="<!-- REPAIR_20261106_CURRENT_END -->"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def insert_current(path,text,history_label=None):
    original=Path(path).read_text(encoding="utf-8")
    if MARK_START in original:
        before,rest=original.split(MARK_START,1)
        _,after=rest.split(MARK_END,1)
        result=before+MARK_START+"\n"+text+"\n"+MARK_END+after
    else:
        first,rest=original.split("\n",1)
        label="\n\n"+history_label+"\n" if history_label else "\n"
        result=first+"\n\n"+MARK_START+"\n"+text+"\n"+MARK_END+label+rest
    Path(path).write_text(result,encoding="utf-8",newline="\n")


def main():
    confirmation=read(ROOT/"environment"/(FAMILY+"-confirmation-status.json"))
    assert confirmation["status"]=="completed_one_stage_confirmation"
    assert read(ROOT/"environment"/(FAMILY+"-controlled-status.json"))["status"]=="completed_balanced_development_comparison"
    summary=read(OUT/"summary.json")
    audit=read(OUT/"evidence_audit.json")
    plan=read(EXP/"sampling_frozen.json")
    assert summary["all_nine_confirmation_complete"] and audit["one_stage_confirmation_complete"] and audit["status"]=="pass"
    assert code_hash()==plan["code_sha256"]
    rows=summary["endpoint_rows"]
    aggregates=summary["aggregates"]
    def numbers(arm,subset):
        return "/".join(str(r[subset+"_correct"]) for r in rows if r["arm"]==arm)
    def effect(subset,contrast):
        return next(r for r in summary["paired_effects"] if r["subset"]==subset and r["contrast"]==contrast)
    cfx=effect("confirmation","D-C")
    dfx=effect("confirmation","D-E")
    attribution=read(OUT/"grpo_error_attribution.json")
    cf_pairs=[r for r in attribution["records"] if r["subset"]=="confirmation" and r["contrast"]=="D-C"]
    loss_partition=Counter()
    for record in cf_pairs:
        loss_partition.update(record["left_regression_surface_partition"])
    comp=effect("comparison","D-C")
    base=summary["baseline"]
    a=base["confirmation"]["A"]["correct"]
    b=base["confirmation"]["S2"]["correct"]
    finished=dt.datetime.fromtimestamp(confirmation["finished_unix"],dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
    current=(f"## 最新完成状态：2026-11-06\n\n"
        f"家族 `{FAMILY}` 已完整完成：实际 tokenizer/标签/EOS/重载审计、未训练 LoRA 控制、四个预登记 SFT 候选、成本定标、C/D/E 各 seed13/29/47，以及所有终点独立重载和互斥128题的一次阶段确认。模型任务结束于北京时间 {finished:%Y-%m-%d %H:%M:%S}；本轮完整墙钟见 `experiments/{FAMILY}/session_final.json`。\n\n"
        f"S1 以完全相同的适配器权重复现历史 SFT 退化；同32步降低学习率后比较集从64/128变为106/128、截断40变0。按预登记规则选真实 S2（1e-5、8步、实际32条样本/3688监督token），没有把未训练适配器标为 SFT。所查边界、mask、EOS与重载未发现缺陷；重复输出的内部机制仍未确认。\n\n"
        f"固定 comparison128：A107、S2=110；C={numbers('C','comparison')}，D={numbers('D','comparison')}，E={numbers('E','comparison')}。D−C 均值 {comp['mean_difference_pp']:+.3f}pp。固定 confirmation128：A={a}、S2={b}；C={numbers('C','confirmation')}，D={numbers('D','confirmation')}，E={numbers('E','confirmation')}，D−C 均值 {cfx['mean_difference_pp']:+.3f}pp。分母均为128；两组都是 validation 开发数据，不能写成正式独立测试或稳定泛化收益。\n\n"
        f"新主线为 `rollout_cache_fifo_v1_dynamic`，44题训练池，D/E各自初始K4探测；D复用真实训练rollout的最近16次奖励，32步无新观测回归先验，E固定初始权重。每组600秒训练/采样钟、540秒重载评测预留及30秒外围预留。979个主优化步骤均检查有限、244步非零，九个主适配器确实更新。没有补跑同协议周期全池动态基线，因此不声称优于该历史方法。\n\n"
        f"工程47项测试、69个依赖兼容检查通过；本轮 {audit['runs_checked']} 份运行完成证据审计，含保存权重改变量和精确选题/缓存重放。当前进程隔离为单 RX9070XT，非量化BF16/SDPA；公开模型11个文件已复核，版本与数据均沿用锁定来源。\n\n"
        f"`configs/main.yaml` 已改为实测容量配置 `main_capacity_calibrated_development_only`，不再沿用未测量2048题/NF4草案。最终正式方案、端点选择及独立测试预算仍未冻结；GSM8K test/SVAMP继续封存。具备真实开发对照的工程条件，尚无足够证据宣称动态策略优势或直接启封正式测试。\n\n"
        f"[完整中文报告](reports/comparisons/{FAMILY}/findings.md) · [复现命令](docs/REPRODUCING_REPAIR_20261106.md) · [逐终点成本](reports/comparisons/{FAMILY}/standalone_endpoints.csv) · [证据审计](reports/comparisons/{FAMILY}/evidence_audit.json)。本轮源码、配置、报告及原始工件仍待Git提交/推送；本地适配器备份不含公开基础模型、环境或完整优化器状态。此前已发布的历史结果保持原样。")
    insert_current(ROOT/"README.md",current)
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    readme=readme.replace("## 已有真实结果与当前实验","## 2026-11-05 历史真实结果")
    readme=readme.replace("`main.yaml` 目前是容量上限草案，正式冻结前需要 pilot 定标。", "`main.yaml` 已按2026-11-06本机实测改为开发容量配置，尚非最终正式协议；本轮不得使用上面的历史示意命令启封test/SVAMP。")
    (ROOT/"README.md").write_text(readme,encoding="utf-8",newline="\n")
    insert_current(ROOT/"PROJECT_CONTEXT.md",current, "## 历史上下文（2026-11-05 轮次与迁移原文保留）")
    insert_current(ROOT/"HANDOFF.md",current+f"\n\n继续时优先阅读本节、完整报告和 `experiments/{FAMILY}/sampling_frozen.json`。本轮模型任务已结束，不续跑旧状态、不覆盖旧家族。待提交/备份清单见 `reports/comparisons/{FAMILY}/pending_artifacts.md`；最终墙钟、归档校验和本地未推送状态以收尾收据为准。", "## 历史交接（上一轮与笔记本记录原样保留）")
    # These two files live under docs, so their relative report links need ../.
    protocol=(f"## 2026-11-06 当前开发协议\n\n"
        f"当前已执行方案以 [冻结文件](../experiments/{FAMILY}/sampling_frozen.json) 为准：真实S2共同起点、train44题、C均匀/D rollout_cache_fifo_v1_dynamic/E初始静态、三个配对种子13/29/47，独立重载比较128题，端点全部固定后一次互斥128题阶段确认。600秒算法钟包括加载准备、初始K4探测、rollout、优化和维护；540秒独立评测和30秒进程外围预留形成1170秒共同上限，实际未用足与自然边界超额逐项报告。\n\n"
        f"D只用训练rollout更新最近16奖励缓存，32个优化步无新观测时回到p=0.5，概率含0.25均匀下界；E初始探测和初始权重与D一致，此后不更新。后续全池生成次数为0，不冒称完成了周期全池刷新。所有探测/成功率估计只使用train。下文周期全池与NF4段落保留为历史v0.1协议，不能替代本轮冻结文件。\n\n"
        f"[中文结果与局限](../reports/comparisons/{FAMILY}/findings.md) · [复现](REPRODUCING_REPAIR_20261106.md)。两组128题均为开发validation；未冻结最终正式协议，test/SVAMP仍封存。")
    insert_current(ROOT/"docs/PROTOCOL.md",protocol)
    setup=(f"## 2026-11-06 后续实测\n\n"
        f"本机复用该环境完成 `{FAMILY}`：四SFT候选、C/D/E各三种子及所有终点独立重载、互斥128题阶段确认。47项工程测试和69个依赖检查通过，所有运行只暴露RX9070XT、实际非量化BF16/SDPA。`main.yaml` 已按成本定标，不是最终保留集授权。[新结果](../reports/comparisons/{FAMILY}/findings.md) 与 [复现说明](REPRODUCING_REPAIR_20261106.md) 是当前入口；下文为迁移与上一轮历史。")
    insert_current(ROOT/"docs/DESKTOP_SETUP_20261105.md",setup)
    report=(OUT/"findings.md").read_text(encoding="utf-8")
    status_start=report.index("状态：")
    status_end=report.index("\n\n",status_start)
    report=report[:status_start]+f"状态：本轮全部模型任务已于北京时间 {finished:%Y-%m-%d %H:%M:%S} 完整结束；九组训练、两组128题独立重载评测、一次阶段确认均完成。运行上限为北京时间15:27:12–21:27:12，整轮最终墙钟（包含收尾归档）见 [会话账本](../../../experiments/{FAMILY}/session_final.json)。"+report[status_end:]
    report=report.replace("图片右侧阶段确认尚在写入，最终报告将使用全部固定终点更新。图只使用真实已完成观测；没有插值验证曲线或补写中间成绩。", "图中左右两组128题均已完成，所有训练终点都由新进程独立重载。只使用真实已完成观测，没有插值验证曲线或补写中间成绩。")
    start=report.index("## 阶段确认与当前边界")
    stop=report.index("工程上已经具备",start)
    confirmation_lines=["## 阶段确认与当前边界","",f"确认集在九个主checkpoint全部固定后才按预登记开启；包括A、S2与九个C/D/E，所有终点独立加载。[开启前清单](../../../experiments/{FAMILY}/confirmation_opened.json)。此阶段没有再选择参数、种子或检查点。","",
        "|组别 / seed|13|29|47|均值±种子样本标准差|","|---|---:|---:|---:|---:|"]
    for arm in ("C","D","E"):
        vals=[r["confirmation_correct"] for r in rows if r["arm"]==arm]
        ag=aggregates["confirmation"][arm]
        confirmation_lines.append(f"|{arm}|{vals[0]}/128|{vals[1]}/128|{vals[2]}/128|{ag['mean_accuracy_pct']:.3f}% ±{ag['sample_seed_sd_pct']:.3f}pp|")
    confirmation_lines += ["",f"A={a}/128（{a/128*100:.3f}%），S2={b}/128（{b/128*100:.3f}%）。S2相对A为{(b-a)/128*100:+.3f}pp，比较集的三题优势没有在确认集复现，不能强求已后训练模型额外SFT必须改善。", "",
        f"确认集D−C的三个配对差为{', '.join(f'{x:+.3f}' for x in cfx['paired_seed_differences_pp'])}pp，均值{cfx['mean_difference_pp']:+.3f}pp；同题条件bootstrap95%区间[{cfx['question_paired_bootstrap95_pp'][0]:.3f}, {cfx['question_paired_bootstrap95_pp'][1]:.3f}]pp。D−E均值{dfx['mean_difference_pp']:+.3f}pp。比较集D−C为+0.781pp，确认集的结果必须与其分开解释，不挑选更有利的一个集合当结论。", "",
        f"三个固定种子的逐题配对中，C对而D错合计{sum(r['right_correct_left_wrong'] for r in cf_pairs)}次，D对而C错{sum(r['right_wrong_left_correct'] for r in cf_pairs)}次。回归项互斥分为：{loss_partition['nontruncated_wrong_numeric_requires_review']}次未截断且数值错、{loss_partition['nontruncated_parse_failure']}次未截断解析失败、{loss_partition['truncated']}次截断。因此差异不能仅归为终止或解析问题。这里是同128题跨3种子的384个相关配对，不是384个独立问题；数值错不能自动标成算术错。[逐题误差账本](grpo_error_attribution.json)与[待语义审阅样例](grpo_confirmation_case_pack.jsonl)保留原始输出。", "",
        "SFT保住比较集表现与采样机制真实执行均已验证；是否带来稳定的额外能力收益，当前证据不足。缓存成功率、选题覆盖和稀疏梯度是机制诊断，不是独立泛化指标。确认结果保持负面或中性也不触发追加搜索。",""]
    report=report[:start]+"\n".join(confirmation_lines)+"\n"+report[stop:]
    report=report.replace("最终交接将补充完整确认、终审、归档摘要和 [复现命令](../../../docs/REPRODUCING_REPAIR_20261106.md)。历史运行、旧配置和负面结果保留，不与本轮均值混算。",
        f"本轮{audit['runs_checked']}份运行完成 [证据终审](evidence_audit.json)，所有新训练终点、未训练控制和一次阶段确认均保留。[复现命令](../../../docs/REPRODUCING_REPAIR_20261106.md)、[失败/偏差记录](failures_and_deviations.json) 与 [待提交/归档清单](pending_artifacts.md) 一并交接。历史运行、旧配置和负面结果保留，不与本轮均值混算。")
    report=report.replace("[全终点逐题预测](all_endpoint_predictions.csv)","[主比较与确认逐题预测](all_endpoint_predictions.csv)")
    if "GRPO阶段的进程峰值" not in report:
        report=report.replace("九个适配器都真实改变。",f"九个适配器都真实改变。GRPO阶段的进程峰值allocated为{min(r['peak_allocated_gib'] for r in rows):.2f}–{max(r['peak_allocated_gib'] for r in rows):.2f}GiB，reserved为{min(r['peak_reserved_gib'] for r in rows):.2f}–{max(r['peak_reserved_gib'] for r in rows):.2f}GiB，不是系统总显存占用。")
    if "按策略累计非零梯度步骤" not in report:
        signal=attribution["learning_signal_by_arm"]
        report=report.replace("二值奖励的零方差、截断 loss mask 与 Adam 更新的存在需要分开解释。",f"二值奖励的零方差、截断 loss mask 与 Adam 更新的存在需要分开解释。按策略累计非零梯度步骤，C/D/E为{signal['C']['nonzero_gradient_steps']}/{signal['D']['nonzero_gradient_steps']}/{signal['E']['nonzero_gradient_steps']}，虽然C完成更多optimizer step，非零学习步骤并没有相应更多；D与E也没有清晰的信号数量优势。该诊断不能替代端点结果或相同步数消融。")
    report=report.replace("后续应据本次完整确认结果决定正式实验是否值得继续，而不是反复调到正收益。", "进入正式验证不要求先在开发集调到正收益；当前欠缺的是具体正式协议、端点与预算冻结，而不是必须发现正结果。44题小池与吞吐波动应作为正式方案的范围选择和局限记录，不能通过继续看确认集来优化参数。")
    (OUT/"findings.md").write_text(report,encoding="utf-8",newline="\n")
    failures=jsonl_read(EXP/"failures.jsonl") if (EXP/"failures.jsonl").exists() else []
    json_write(OUT/"failures_and_deviations.json",{"runtime_failed_phases":failures,"all_model_phases_completed":True,
        "negative_results_preserved":["S1 exact reproduction of SFT collapse","S2 comparison gain not repeated on confirmation","Sparse GRPO gradient signal and small/inconsistent sampler gains"],
        "engineering_recovery_incidents":jsonl_read(EXP/"recovery_incidents.jsonl"),
        "additional_analysis_incident":{"error":"One ad-hoc Python -c read command hit an unterminated string literal due to PowerShell quoting","resolution":"Read-only command replaced with native PowerShell/here-string; no experiment was restarted or changed"},
        "source_evolution_note":"During early SFT candidates, GRPO-only code and provenance collectors evolved. Whole-source per-run hashes are retained. SFT/encoding/collator/generation ASTs are identical, and S1 saved weights and old32 raw predictions match the historical B exactly. The later frozen main/confirmation execution source is identical across all arms.",
        "planned_budget_deviations":"Small natural-boundary600s overshoot and unused independent-evaluation reserve are measured, not hidden. No endpoint/checkpoint selection after stage confirmation.",
        "holdout_or_paid_service_used":False})
    json_write(ROOT/"environment"/(FAMILY+"-verification.json"),{"engineering_tests_passed":47,
        "engineering_test_log":f"logs/{FAMILY}-engineering-check.log","dependency_packages_checked":69,
        "dependency_log":f"logs/{FAMILY}-dependency-check.log","new_runs_evidence_audited":audit["runs_checked"],
        "code_sha256":code_hash(),"scientific_execution_source_unchanged_since_main_freeze":True,
        "confirmation_complete":True,"formal_test_opened":False})
    print(json.dumps({"confirmation_D_minus_C_pp":cfx["mean_difference_pp"],"new_runs_audited":audit["runs_checked"],"documents_updated":True},indent=2))


if __name__=="__main__":
    main()
