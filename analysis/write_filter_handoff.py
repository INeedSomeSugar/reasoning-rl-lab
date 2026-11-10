"""Write reviewed final results into the existing Markdown report and handoff files."""
import datetime as dt
import json
from pathlib import Path
import re
import sys
from reasoning_rl.runtime import ROOT, json_write, sha256_file
sys.path.insert(0,str(ROOT/"scripts"))
from filter_common import EXP,FAMILY,read

OUT=ROOT/"reports/comparisons"/FAMILY


def pct(value):
    return f"{value*100:.3f}%"


def replace_block(path, body):
    path=ROOT/path
    text=path.read_text(encoding="utf-8")
    start="<!-- FILTER_20261110_CURRENT_BEGIN -->"
    end="<!-- FILTER_20261110_CURRENT_END -->"
    block=start+"\n"+body.strip()+"\n"+end
    if start in text:
        text=re.sub(re.escape(start)+r".*?"+re.escape(end),lambda _:block,text,flags=re.S)
    else:
        first,remaining=text.split("\n",1)
        text=first+"\n\n"+block+"\n"+remaining
    path.write_text(text,encoding="utf-8",newline="\n")


def main():
    if (EXP/"report_written.json").exists():
        raise FileExistsError("Final report already written; edit reviewed corrections in place")
    assert read(ROOT/"environment"/(FAMILY+"-status.json"))["status"] == "completed_three_paired_seed_development"
    s=read(OUT/"summary.json")
    audit=read(OUT/"evidence_audit.json")
    assert audit["all_checks_passed"]
    interpretation=read(EXP/"interpretation_final.json")  # explicitly reviewed after actual outcomes
    rows=s["endpoints"]
    u,f=s["by_arm"]
    assert [u["arm"],f["arm"]] == ["U","F"]
    differences=s["paired_differences"]
    paired_text="、".join(f"{r['F_minus_U_pp']:+.4f}" for r in differences)
    conditional=s["conditional_question_bootstrap_95_pp"]
    intro=(f"**{interpretation['main_finding']}。** U/F 的开发准确率三种子均值为 **{pct(u['mean_accuracy'])} / {pct(f['mean_accuracy'])}**，"
        f"F−U 配对差均值 **{s['mean_paired_F_minus_U_pp']:+.4f} 个百分点**。共同真实S2本机独立重载为 {s['s2_baseline']['correct']}/128。"
        "所有六个终点从新进程加载并完成同一冻结开发128题评测；GSM8K test/SVAMP保持封存。\n\n")
    result="## 三种子真实终点与主要判断\n\n"
    result+="|seed|U正确 /128|F正确 /128|F−U百分点|\n|---|---:|---:|---:|\n"
    for r in differences:
        result+=f"|{r['seed']}|{r['U_correct']}|{r['F_correct']}|{r['F_minus_U_pp']:+.4f}|\n"
    result+=f"\n三种子配对差依次为 {paired_text}pp，样本标准差 {s['paired_difference_seed_sample_sd_pp']:.3f}pp。"
    result+=f"固定这三个种子、按同题联合重采样的描述性95% bootstrap区间为 [{conditional[0]:+.3f}, {conditional[1]:+.3f}]pp。"
    result+="这不是种子不确定性或正式泛化区间，三个种子和同一128题的384个相关配对不能当作384道独立问题。validation内的近重复簇也可能违反逐题独立近似。沿用confirmation128是已经使用过的开发数据。\n\n"
    result+="![全部配对种子终点](accuracy_paired_accessible.png)\n\n"
    result+="[三种子配对差CSV](paired_differences.csv) · [结构化原始汇总](summary.json)。\n\n"
    result+=interpretation["accuracy_interpretation"]+"\n\n"
    result+="## 机制、有效信号与真实更新\n\n"
    result+="|组别|全对 /全错 /混合组|零方差组比例|非零梯度步 /优化步|有效优势completion|含探测与重载评测的有效completion/秒|\n|---|---|---:|---|---:|---:|\n"
    for a in (u,f):
        result+=(f"|{a['arm']}|{a['all_correct_groups']} /{a['all_wrong_groups']} /{a['mixed_groups']}|{pct(a['zero_variance_group_fraction'])}|"
            f"{a['nonzero_gradient_steps']} /{a['optimizer_steps']}|{a['effective_advantage_completions']}|{a['effective_completions_per_paid_end_to_end_second']:.5f}|\n")
    result+="\n"+interpretation["mechanism_interpretation"]+"\n\n"
    for a in (u,f):
        result+=(f"{a['arm']}的有效completion/付费训练链路秒为 {a['effective_completions_per_paid_training_second']:.5f}，"
            f"非零梯度步/含评测端到端秒为 {a['nonzero_gradient_steps_per_paid_end_to_end_second']:.5f}；"
            f"{a['zero_gradient_steps']}个零梯度步中，{a['zero_gradient_steps_with_actual_weight_change']}步仍实际改变参数。")
        result+="\n\n"
        result+=(f"{a['arm']}零优势completion为{a['zero_advantage_completions']}/{a['completions']}（{a['zero_advantage_completions']/a['completions']*100:.3f}%），"
            f"截断屏蔽为{a['truncated_completions']}/{a['completions']}，其中{a['nonzero_advantage_completions']-a['effective_advantage_completions']}条原有非零优势；"
            f"零梯度optimizer step为{a['zero_gradient_steps']}/{a['optimizer_steps']}（{a['zero_gradient_steps']/a['optimizer_steps']*100:.3f}%）。"
            "组、completion和optimizer step为不同口径；本配置一组对应一步已由实际日志核验。\n\n")
    result+="![实际机制与含筛选成本的效率](mechanism_and_cost_accessible.png)\n\n"
    result+="[逐题训练曝光](item_exposure.csv)保留实际覆盖与重复采样次数。\n\n"
    result+=interpretation["signal_drift_interpretation"]+"\n\n"
    result+="[实际信号首尾四分之一](signal_drift.json)与[U中被初始探测排除题的训练信号](screened_out_training_signal.json)只描述各自真实轨迹；不能把U上的信号当作F未发生的反事实结果。\n\n"
    result+="## 端到端成本、token与异常\n\n"
    result+="|终点|优化步|训练进程秒|F探测扣费秒|独立评测秒|付费端到端秒 /共同1830秒上限|训练生成token|截断 /解析失败completion|\n|---|---:|---:|---:|---:|---:|---:|---|\n"
    for r in rows:
        result+=(f"|{r['label']}|{r['optimizer_steps']}|{r['training_process_seconds']:.2f}|{r['probe_standalone_debit_seconds']:.2f}|"
            f"{r['evaluation_process_seconds']:.2f}|{r['charged_end_to_end_seconds']:.2f}|{r['training_generated_tokens']}|"
            f"{r['truncated_completions']} /{r['parse_failures']}（分母{r['completions']}）|\n")
    result+="\n所有行的训练进程包含启动、加载、缓存核验、采样、前后向、优化与保存；评测进程包含独立加载、greedy生成与落盘，外围核验也计入端到端列。"
    result+="F每行额外扣完整292.832841秒探测；物理共享探测只执行一次，各F的97.610947秒摊销仅作成本解释，没有一套假想的额外训练结果。"
    result+="原子rollout/optimizer边界造成的训练额度小幅超额与共同上限未用足均保留在CSV，评测剩余额度不回填。\n\n"
    for a in (u,f):
        result+=(f"{a['arm']}三终点训练生成 {a['training_generated_tokens']} token，独立评测生成 {a['evaluation_generated_tokens']} token；"
            f"单终点口径分配的探测token合计 {a['assigned_probe_generated_tokens']}，含重复分配的总token为 {a['standalone_total_generated_tokens']}。")
        result+="\n\n"
    result+=f"实际完整探测只生成 {s['probe']['generated_tokens']} token；探测解析失败{s['probe']['parse_failures']}条、截断{s['probe']['truncations']}条，分母256。"
    result+=f"全部六个开发终点评测解析失败合计{sum(r['eval_parse_failures'] for r in rows)}条、截断{sum(r['eval_truncations'] for r in rows)}条，分母768。\n\n"
    result+=(f"全会话所有已落盘输出的物理生成token合计至少{audit['physical_generated_tokens_observed_lower_bound']}，共享探测只计一次；"
        f"其中中断尝试已记录80926 token。最后一个未完成生成组可能另有0–{audit['unlogged_inflight_generation_tokens_upper_bound']} token未落盘，因此不把已观察合计伪装成精确总量。前后向诊断没有生成新token。\n\n")
    result+=interpretation["cost_interpretation"]+"\n\n"
    result+=(f"原始U/13中断尝试没有终点，不混入上述主比较；其已观察耗时至少{audit['interrupted_attempt_observed_lower_seconds']:.2f}秒，"
        f"按执行会话失联检测区间保守记账{audit['interrupted_attempt_conservative_upper_seconds']:.2f}秒，不能把它写成精确退出耗时。"
        f"已包装模型进程实测合计{audit['wrapped_model_process_seconds']:.2f}秒，包含该失败记账及未包装前检准备的保守合计为{audit['conservative_model_related_process_upper_bound_seconds']:.2f}秒。"
        f"从预登记到最后模型任务结束的全会话墙钟为{audit['session_wall_through_last_model_seconds']:.2f}秒，未超过14400秒。成功终点的共同算法预算与含失败的实际运行成本必须分开理解；不能用忽略失败的数字宣称整轮运维成本优势。\n\n")
    result+="[逐终点完整CSV](standalone_endpoints.csv) · [原始进程成本](../../../experiments/desktopfilter_20261110/process_costs.jsonl) · [中断与恢复](../../../experiments/desktopfilter_20261110/interruption_receipt.json)。device elapsed包含设备空闲，不等同活跃kernel时间或FLOPs；PyTorch显存峰值不是系统总显存，能耗未计量。\n\n"
    result+="## 误差分析与下一步\n\n"+interpretation["error_interpretation"]+"\n\n"
    result+="[全部配对预测](all_endpoint_predictions.jsonl)与[配对不一致原始样例](paired_error_cases.jsonl)保留全部种子，没有只展示最好种子。自动分类只区分截断、解析失败和可解析数值错；未人工逐条判断的数值错不自动称为算术错误。\n\n"
    result+=interpretation["flagged_item_interpretation"]+"\n\n"
    result+="**下一步判断：** "+interpretation["next_step"]+"\n\n"
    result+="工程59项回归测试通过，69个项目环境包兼容；全部实际advantage/mask、梯度、均匀采样权重、预算、检查点哈希及独立加载协议通过[证据审计](evidence_audit.json)。"
    result+="[复现命令](../../../docs/REPRODUCING_FILTER_20261110.md)及[检查点清单](checkpoint_manifest.json)可直接定位权重与Adam/RNG诊断状态；尚未实现精确恢复dataloader/sampler位置的自动续训。本轮只保存本地成果，未推送GitHub或发布Release。\n\n"
    path=OUT/"findings.md"
    text=path.read_text(encoding="utf-8")
    text=text.replace("**本报告正在随真实运行补齐，U/F 主对照尚未完成。**","**三个配对种子U/F训练及六个独立进程终点评测已全部完成。**")
    text=text.replace("## 运行中发现及恢复（结果未完成）","## 中断恢复与冻结后数据质量发现")
    text=text.replace("新主实验会逐步记录梯度、Adam 一阶矩和实际权重差。","新主实验已逐步记录梯度、Adam 一阶矩和实际权重差。")
    text=text.replace("最终将匹配成功终点的算法比较与包含失败尝试的实际运行成本分列。","报告已将匹配成功终点的算法比较与包含失败尝试的实际运行成本分列。")
    text=text.replace("主对照结果、成本、误差分析、下一步判断与最终审计将在运行结束后写入本文件；当前不声明预筛带来准确率或效率收益。","主结果、成本和判断见报告开头；冻结协议与全部实际输出保持原样。")
    title,rest=text.split("\n",1)
    text=title+"\n\n"+intro+result+"---\n"+rest.lstrip()
    path.write_text(text,encoding="utf-8",newline="\n")
    short=(f"## 最新完成：固定S2预筛与均匀对照（2026-11-10）\n\n"
        f"`{FAMILY}` 已完成：64题共同池，固定S2/K4严格保留原始成功次数1/2/3的16题，U/F各seed13/29/47；全部六个训练终点固定后，由独立进程重载评测同一开发128题。"
        f"U正确数{'/'.join(str(r['U_correct']) for r in differences)}，F为{'/'.join(str(r['F_correct']) for r in differences)}；"
        f"均值{pct(u['mean_accuracy'])}/{pct(f['mean_accuracy'])}，F−U平均{s['mean_paired_F_minus_U_pp']:+.4f}pp。{interpretation['main_finding']}。\n\n"
        f"零方差组U/F为{pct(u['zero_variance_group_fraction'])}/{pct(f['zero_variance_group_fraction'])}，"
        f"计入完整筛选和独立重载的有效completion/秒为{u['effective_completions_per_paid_end_to_end_second']:.5f}/{f['effective_completions_per_paid_end_to_end_second']:.5f}。"
        "每F扣完整292.832841秒探测，每臂1200秒训练/筛选额度、共同1830秒端到端上限。原U/13交互中断的136步失败工件和成本区间保留；主结果来自同配置新前缀r1，不混入失败权重。\n\n"
        "冻结后发现训练题06831自指措辞疑义，已单列其信号贡献，本轮未改金标/删题。confirmation128明确是已使用的开发数据；GSM8K test/SVAMP继续封存。"
        f"下一步：{interpretation['next_step']}\n\n"
        f"[完整报告](reports/comparisons/{FAMILY}/findings.md) · [复现命令](docs/REPRODUCING_FILTER_20261110.md) · "
        f"[证据审计](reports/comparisons/{FAMILY}/evidence_audit.json) · [冻结协议](experiments/{FAMILY}/protocol_frozen.json)。"
        "全部模型任务已结束，成果保存在本项目，尚未提交/推送；不要重跑已完成家族或启封正式测试。")
    replace_block("HANDOFF.md",short)
    replace_block("PROJECT_CONTEXT.md",short)
    replace_block("README.md",short)
    readme=ROOT/"README.md"
    text=readme.read_text(encoding="utf-8").replace("## 最新开发实验：SFT 排查与缓存采样","## 上一轮开发实验：SFT 排查与缓存采样")
    readme.write_text(text,encoding="utf-8",newline="\n")
    json_write(EXP/"report_written.json",{"written_unix":__import__("time").time(),"report_sha256":sha256_file(path),
        "reviewed_interpretation_sha256":sha256_file(EXP/"interpretation_final.json"),"files":["HANDOFF.md","PROJECT_CONTEXT.md","README.md",str(path.relative_to(ROOT))]})
    print(json.dumps({"report_written":str(path),"paired_mean_difference_pp":s["mean_paired_F_minus_U_pp"]}))


if __name__=="__main__":
    main()
