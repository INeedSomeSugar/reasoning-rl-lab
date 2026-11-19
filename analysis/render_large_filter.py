"""Publication-style static figures from the independently audited endpoint tables."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

from reasoning_rl.runtime import ROOT, bootstrap
bootstrap()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read, TRAIN_SEEDS, PROBE_SEEDS

COLORS = {"U": "#65717D", "F": "#225F91", "R": "#B47727"}


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": "#89929B",
        "grid.color": "#E4E7EB", "grid.linewidth": 0.7, "axes.axisbelow": True})


def save(fig, directory, name):
    fig.savefig(directory / (name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(directory / (name+".svg"), bbox_inches="tight")
    plt.close(fig)


def groups(summary):
    result = [("U", "U", None)]
    for index, j in enumerate(PROBE_SEEDS, 1):
        result.extend([(f"F{index}", "F", j), (f"R{index}", "R", j)])
    return [(label, arm, j, [r for r in summary["endpoints"] if r["arm"] == arm and (j is None or r["probe_seed"] == j)])
        for label, arm, j in result]


def main(family):
    out = ROOT / "reports/comparisons" / family
    exp = ROOT / "experiments" / family
    s = read(out / "summary.json")
    by_key = {r["endpoint"]: r for r in s["endpoints"]}
    style()
    group_data = groups(s)
    # Every plotted point is a real endpoint. Shared U points are intentionally repeated by panel.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    valid = [r["accuracy"]*100 for r in s["endpoints"] if r["accuracy"] is not None]
    lo, hi = max(0, min(valid)-3), min(100, max(valid)+3)
    incomplete = s.get("available_pair_comparisons")
    for j_index, (ax, j) in enumerate(zip(axes, PROBE_SEEDS)):
        for offset, arm, marker in ((-.19, "U", "o"), (0, "F", "o"), (.19, "R", "^")):
            points = [by_key[f"U-s{x}" if arm == "U" else f"{arm}-j{j}-s{x}"] for x in TRAIN_SEEDS]
            vals = [r["accuracy"]*100 if r["accuracy"] is not None else np.nan for r in points]
            ax.scatter(np.arange(5)+offset, vals, s=54, marker=marker,
                facecolors="white" if arm == "R" else COLORS[arm], edgecolors=COLORS[arm], linewidths=1.5, label=arm, zorder=3)
            for x, value in enumerate(vals):
                if not np.isfinite(value):
                    ax.text(x+offset, .026, "×", transform=ax.get_xaxis_transform(), ha="center", color=COLORS[arm], fontweight="bold")
        for anchor, line, color in (("A", "--", "#353B43"), ("S2", ":", "#6D747D")):
            if by_key[anchor]["accuracy"] is not None:
                ax.axhline(by_key[anchor]["accuracy"]*100, linestyle=line, color=color, linewidth=1.2, label=anchor)
        ax.set(xticks=np.arange(5), xticklabels=TRAIN_SEEDS, xlabel="训练种子", title=f"探测种子 {j}\nM={s['probe_summary'][j_index]['M']}", ylim=(lo, hi))
        ax.grid(axis="y")
    axes[0].set_ylabel(f"新开发严格正确率（%，N={s['n']}）")
    axes[1].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    title = "全部注册种子的终点准确率｜U 的 5 个训练终点跨面板复用"
    if incomplete:
        title += "\n底部 × 表示评估缺失"
    fig.suptitle(title, y=1.10 if incomplete else 1.06)
    fig.subplots_adjust(wspace=.12)
    save(fig, out, "endpoint_accuracies")
    if incomplete or "paired_F_minus_U_matrix_pp" in s:
        values = [incomplete[name]["matrix_pp_probe_by_training"] for name in ("F-U", "F-R")] if incomplete else [
            s["paired_F_minus_U_matrix_pp"], s["paired_F_minus_R_matrix_pp"]]
        matrices = [np.array(m, dtype=float).T for m in values]
        lim = max([.1]+[abs(float(x)) for m in matrices for x in m.ravel() if np.isfinite(x)])
        cmap = LinearSegmentedColormap.from_list("signed", ["#B47727", "#F7F8FA", "#225F91"])
        cmap.set_bad("#E4E7EB")
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), constrained_layout=True)
        for ax, matrix, title in zip(axes, matrices, ("F−U", "F−R")):
            im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
            for i in range(5):
                for j in range(3):
                    value = matrix[i, j]
                    ax.text(j, i, f"{value:+.3f}" if np.isfinite(value) else "缺失", ha="center", va="center", fontsize=12,
                        color="white" if np.isfinite(value) and value > .8*lim else "#151B23")
            ax.set(xticks=np.arange(3), xticklabels=[str(x) for x in PROBE_SEEDS], yticks=np.arange(5), yticklabels=TRAIN_SEEDS,
                xlabel="探测块", ylabel="训练种子", title=title+"（百分点）")
        fig.colorbar(im, ax=axes, shrink=.85, label="同一新开发集上的正确率差（百分点）")
        save(fig, out, "paired_differences")
        names = ["F-U", "F-R", "F-S2", "F-A"]
        estimates = [incomplete[name]["available"]["mean_pp"] for name in names] if incomplete else [
            s["F_minus_U_mean_pp"], s["F_minus_R_mean_pp"], s["F_minus_S2_mean_pp"], s["F_minus_A_mean_pp"]]
        fig, ax = plt.subplots(figsize=(9.5, 4.7))
        for i, (name, estimate) in enumerate(zip(names, estimates)):
            bounds = incomplete[name]["available"]["conditional_percentile95_pp"] if incomplete else s["conditional_intervals"][name]["percentile95_pp"]
            if bounds is None:
                continue
            low, high = bounds
            ax.plot([low, high], [i, i], color=COLORS["F"], linewidth=2)
            ax.scatter([estimate], [i], color=COLORS["F"], s=55, zorder=3)
            ax.text(high+.05, i, f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]", va="center")
        ax.axvline(0, color="#4B5563", linestyle="--", linewidth=1)
        labels = [f"{name} ({incomplete[name]['available']['pairs']}/15)" for name in names] if incomplete else names
        ax.set(yticks=range(4), yticklabels=labels, xlabel="正确率差（百分点）",
            title="可观测配对的条件 95% 区间｜固定模型与选题块" if incomplete else "联合题目簇 bootstrap 条件 95% 区间")
        ax.invert_yaxis()
        ax.margins(x=.25)
        ax.grid(axis="x")
        save(fig, out, "conditional_intervals")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, label in zip(axes,
        ("effective_completions_per_charged_second", "nonzero_gradient_steps_per_charged_second"),
        ("有效 completion／计费端到端秒", "非零梯度步／计费端到端秒")):
        group_labels = []
        for x, (group, arm, j, rs) in enumerate(group_data):
            selected = [r for r in rs if r[metric] is not None]
            ys = [r[metric] for r in selected]
            ax.scatter(x+np.linspace(-.12, .12, len(ys)), ys, color=COLORS[arm], marker="^" if arm == "R" else "o", s=35)
            if ys:
                numerator = "effective_completions" if metric.startswith("effective") else "nonzero_gradient_steps"
                pooled = sum(r[numerator] for r in selected)/sum(r["charged_end_to_end_seconds"] for r in selected)
                ax.plot([x-.25, x+.25], [pooled]*2, color=COLORS[arm], linewidth=2)
            group_labels.append(f"{group}\n{len(ys)}/{len(rs)}" if incomplete else group)
        ax.set(xticks=range(7), xticklabels=group_labels, ylabel=label, ylim=(0, None))
        ax.grid(axis="y")
    fig.suptitle("有完整评估的计费路径｜点为独立终点，横线为组内汇总比率，R 含配平扣额")
    fig.tight_layout()
    save(fig, out, "signal_efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for ax, metric, ylabel in zip(axes, ("coverage_fraction", "effective_top10_question_share"),
        ("实际训练不同题数／训练池大小", "有效 completion 最多的前 10 题占比")):
        for x, (label, arm, j, rs) in enumerate(group_data):
            vals = [s["training_signal"][r["endpoint"]][metric] for r in rs if r["endpoint"] in s["training_signal"]]
            vals = [v for v in vals if v is not None]
            ax.scatter(x+np.linspace(-.12,.12,len(vals)), vals, color=COLORS[arm], marker="^" if arm == "R" else "o")
        ax.set(xticks=range(7), xticklabels=[g[0] for g in group_data], ylabel=ylabel, ylim=(0, 1.05))
        ax.grid(axis="y")
    fig.suptitle("题目覆盖与有效输出集中度｜U 池 256 题，F/R 在每块均为 M 题")
    fig.tight_layout()
    save(fig, out, "coverage_concentration")
    with (out / "training_signal_windows.csv").open(encoding="utf-8-sig") as f:
        windows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    for label, arm, j, rs in group_data:
        keys = {r["endpoint"] for r in rs}
        values, effective = [], []
        for window in ("first", "middle", "last"):
            selected = [r for r in windows if r["endpoint"] in keys and r["window"] == window]
            count = sum(int(r["groups"]) for r in selected)
            mixed = sum(int(r["mixed_groups"]) for r in selected)
            eff = sum(int(r["effective_completions"]) for r in selected)
            values.append(1-mixed/count if count else np.nan)
            effective.append(eff/(4*count) if count else np.nan)
        for ax, yy in zip(axes, (values, effective)):
            ax.plot(range(3), yy, label=label, color=COLORS[arm],
                linestyle="--" if arm == "R" else "-", marker="o" if j is None else ("s", "^", "D")[PROBE_SEEDS.index(j)],
                markerfacecolor="white" if arm == "R" else COLORS[arm], linewidth=1.4)
    for ax, label in zip(axes, ("零方差组占比（全对或全错）", "有效 completion／全部生成 completion")):
        ax.set(xticks=range(3), xticklabels=["前1/3时间", "中1/3时间", "后1/3时间"], ylabel=label, ylim=(0, 1))
        ax.grid(axis="y")
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(.5,-.17), frameon=False)
    fig.suptitle("全部 35 个训练终点的实际信号随时间变化")
    fig.tight_layout()
    save(fig, out, "signal_time_windows")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    components = [("training_process_seconds", "自身训练进程", "#225F91", None),
        ("evaluation_process_seconds", "独立评估进程", "#C8A43A", None),
        ("F_full_algorithm_probe_debit_seconds", "F完整探测扣额", "#B47727", None),
        ("R_experimental_matching_debit_seconds", "R实验配平项", "#BA718C", "//")]
    cost_groups = [[r for r in rs if r["charged_end_to_end_seconds"] is not None] for _, _, _, rs in group_data]
    bottoms = np.zeros(len(cost_groups))
    for field, label, color, hatch in components:
        vals = np.array([np.mean([r[field] for r in rows]) if rows else 0 for rows in cost_groups])/60
        axes[0].bar(range(7), vals, bottom=bottoms, color=color, label=label, hatch=hatch, edgecolor="white", linewidth=.5)
        bottoms += vals
    parent = []
    for rs in cost_groups:
        vals = [r["charged_end_to_end_seconds"]-sum(r[f] for f,_,_,_ in components) for r in rs]
        parent.append(np.mean(vals)/60 if vals else 0)
    axes[0].bar(range(7), parent, bottom=bottoms, color="#A0A7B0", label="父进程核验等外围")
    bottoms += np.array(parent)
    failed_low = np.array([np.mean([r["failed_attempt_own_path_seconds_lower_bound"] for r in rs]) if rs else 0 for rs in cost_groups])/60
    failed_high = np.array([np.mean([r["failed_attempt_own_path_seconds_upper_bound"] for r in rs]) if rs else 0 for rs in cost_groups])/60
    axes[0].bar(range(7), failed_low, bottom=bottoms, color="#B95353", hatch="xx", label="失败尝试成本下界")
    axes[0].errorbar(range(7), bottoms+failed_low, yerr=np.array([np.zeros(7), failed_high-failed_low]),
        fmt="none", ecolor="#6B2222", capsize=3, linewidth=1)
    labels = [f"{group[0]}\n{len(rs)}/{len(group[3])}" if incomplete else group[0] for group, rs in zip(group_data, cost_groups)]
    axes[0].set(xticks=range(7), xticklabels=labels, ylabel="计费端到端分钟／完整终点", title="成功路径均值及其对应失败成本界")
    axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(.5,-.13), frameon=False)
    axes[0].grid(axis="y")
    physical = [p["full_process_seconds"]/60 for p in s["probe_summary"]]
    axes[1].bar(range(3), physical, color="#B47727")
    for i, value in enumerate(physical):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom")
    axes[1].set(xticks=range(3), xticklabels=["j1", "j2", "j3"], ylabel="实际物理分钟", title="共享探测各实际执行一次")
    axes[1].grid(axis="y")
    fig.tight_layout()
    save(fig, out, "cost_breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    palette = ["#7A8793", "#8DAEC8", "#225F91", "#C49D50", "#B47727"]
    bottoms = np.zeros(3)
    for count in range(5):
        vals = [r[f"raw_success_{count}_items"] for r in s["probe_summary"]]
        axes[0].bar(range(3), vals, bottom=bottoms, color=palette[count], label=f"{count}/4")
        for i, value in enumerate(vals):
            if value >= 8:
                axes[0].text(i, bottoms[i]+value/2, str(value), ha="center", va="center", color="white" if count in (0,2,4) else "#17212C")
        bottoms += vals
    axes[0].set(xticks=range(3), xticklabels=PROBE_SEEDS, ylabel="题数（每块 256 题）", title="原始 K4 成功次数｜保留 1、2、3 次")
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(.5,-.12), frameon=False)
    frequency = Counter(read(exp / "selection_diagnostics.json")["per_item_F_selection_frequency"].values())
    axes[1].bar(range(4), [frequency[i] for i in range(4)], color="#225F91")
    for i in range(4):
        axes[1].text(i, frequency[i], str(frequency[i]), ha="center", va="bottom")
    axes[1].set(xticks=range(4), xlabel="三个独立探测块中的入选次数", ylabel="题数", title="固定S2筛选的题目入选频率")
    fig.tight_layout()
    save(fig, out, "probe_selection")
    print(str(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    main(p.parse_args().family)
