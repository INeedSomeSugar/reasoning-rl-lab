"""Reconcile the first real local pilot, without opening holdout content or running a model."""
import json

from reasoning_rl.runtime import ROOT, bootstrap, json_write, jsonl_read
from reasoning_rl.reporting import write_csv, _plot_style


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    bootstrap()
    endpoints = {
        "A": "local17v2_20261105_smoke-A",
        "B": "local17v2_20261105_smoke-B-val",
        "C": "local17signalv2_20261105_C_reloadval_r2",
        "D": "local17signalv2_20261105_D_reloadval",
        "E": "local17signalv2_20261105_E_reloadval",
    }
    rows, train, predictions = [], {}, {}
    for arm, name in endpoints.items():
        directory = ROOT / "runs" / name
        metric = read(directory / "validation/metrics.json")
        predictions[arm] = jsonl_read(directory / "validation/predictions.jsonl")
        config = read(directory / "config.json")
        if arm == "A":
            reference_config = config
        for field in ("model", "dtype", "quantization", "data_seed", "generation", "evaluation"):
            if config[field] != reference_config[field]:
                raise ValueError(f"Endpoint protocol differs: {arm}, {field}")
        if metric["scorer_version"] != "numeric_final_v2" or metric["decoding"] != "greedy" or metric["thinking"]:
            raise ValueError("Endpoint decoding or scorer differs")
        if [r["id"] for r in predictions[arm]] != [r["id"] for r in predictions["A"]]:
            raise ValueError("Endpoint validation IDs differ")
        rows.append({"arm": arm, "evaluation_run": name,
                     **{k: metric[k] for k in ("n", "correct", "accuracy", "mean_generated_tokens",
                                               "format_error_rate", "parse_error_rate", "truncation_rate")}})
        if arm in "CDE":
            path = ROOT / "runs" / f"local17signalv2_20261105_{arm}_s13"
            result = read(path / "training_result.json")
            costs = jsonl_read(path / "costs.jsonl")
            total = next(c for c in costs if c["phase"] == "grpo_total")
            probe_seconds = sum(c["wall_seconds"] for c in costs if c["phase"] == "difficulty_probe")
            train[arm] = {**result, "wall_seconds": total["wall_seconds"],
                          "probe_wall_seconds": probe_seconds,
                          "probe_wall_fraction": probe_seconds / total["wall_seconds"],
                          "device_elapsed_seconds": total["device_elapsed_seconds"],
                          "peak_allocated_gib": total["peak_allocated_bytes"] / 1024**3,
                          "peak_reserved_gib": total["peak_reserved_bytes"] / 1024**3}
    probe_directory = lambda arm: ROOT / "runs" / f"local17signalv2_20261105_{arm}_s13" / "probes"
    initial_equal = ((probe_directory("D") / "step_00000.jsonl").read_bytes()
                     == (probe_directory("E") / "step_00000.jsonl").read_bytes())
    if not initial_equal:
        raise ValueError("D/E initial probes should match under isolated RNG and identical initial weights")
    failure_rows = []
    for arm in "CDE":
        rollouts = jsonl_read(ROOT / "runs" / f"local17signalv2_20261105_{arm}_s13" / "rollouts.jsonl")
        failure_rows.extend({"arm": arm, "rollout_row_0based": i, **r} for i, r in enumerate(rollouts) if not r["reward"])
    payload = {"scope": "One-seed signal pilot, 8 development questions; unequal compute, no holdout evidence",
               "completed_seeds": [13], "formal_comparison_completed": False,
               "endpoint_protocol_matches": True, "initial_dynamic_static_probes_byte_identical": initial_equal,
               "endpoints": rows, "training": train, "negative_reward_rollouts": failure_rows}
    output = ROOT / "reports"
    json_write(output / "local_pilot_summary.json", payload)
    write_csv(output / "local_pilot_endpoints.csv", rows)
    plt = _plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained")
    colors = {"C": "#2563a6", "D": "#c07810", "E": "#657d39"}
    for arm in "CDE":
        t = train[arm]
        accuracy = next(r["accuracy"] for r in rows if r["arm"] == arm)
        axes[0].scatter(t["wall_seconds"] / 60, 100 * accuracy, s=75, color=colors[arm], label=arm)
        axes[0].annotate(f"{arm}: {round(accuracy*8)}/8", (t["wall_seconds"] / 60, 100*accuracy),
                         xytext=(7, -15 if arm == "E" else 7), textcoords="offset points")
        axes[1].bar(arm, t["wall_seconds"] - t["probe_wall_seconds"], color=colors[arm])
        axes[1].bar(arm, t["probe_wall_seconds"], bottom=t["wall_seconds"] - t["probe_wall_seconds"],
                    color=colors[arm], hatch="///", alpha=0.5)
    axes[0].set(title="Common standalone reload endpoints", xlabel="Observed GRPO time (min)",
                ylabel="Development accuracy (%)", ylim=(0, 100), xlim=(0, max(t["wall_seconds"] for t in train.values()) / 60 * 1.25))
    axes[1].set(title="Observed time, probe overhead hatched", xlabel="GRPO arm", ylabel="Wall time (seconds)")
    fig.suptitle("RTX 3060 Laptop | seed 13 | 8 steps each; unequal compute; no final test", fontsize=11)
    fig.savefig(output / "local_pilot_endpoints.png", dpi=160)
    fig.savefig(output / "local_pilot_endpoints.svg")
    plt.close(fig)
    lines = ["# 本机首轮真实预实验", "", "自动汇总：2026-11-05，RTX 3060 Laptop 6GB。仅seed13、每组8步和8道开发验证题；没有同预算优势或泛化提升结论。", "",
             "## 统一重载后的终点", "", "A/B/C/D/E均使用相同模型revision、NF4/BF16加载协议、非thinking提示、greedy、192-token上限和评分器v2。训练上下文验证不替代该表。", "",
             "|组|正确|平均输出token|截断率|格式错误率|提取失败率|", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['arm']}|{r['correct']}/{r['n']}|{r['mean_generated_tokens']:.2f}|{r['truncation_rate']:.1%}|{r['format_error_rate']:.1%}|{r['parse_error_rate']:.1%}|")
    lines += ["", "## 真正发生的训练与额外开销", "", "|组|步数|总秒数|探测秒数|策略生成token|探测生成token|奖励均值|零方差组|LoRA delta L2|峰值allocated GiB|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, t in train.items():
        lines.append(f"|{arm}|{t['completed_optimizer_steps']}|{t['wall_seconds']:.2f}|{t['probe_wall_seconds']:.2f}|{t['generated_tokens']}|{t['probe_generated_tokens']}|{t['mean_reward']:.5f}|{t['zero_variance_group_fraction']:.1%}|{t['adapter_delta_l2']:.8f}|{t['peak_allocated_gib']:.3f}|")
    lines += ["", "总时间包含探测、策略生成/优化和训练内验证；探测是子项，不能再加到总时间。独立重载的终点评测时间另存于各评测run。device elapsed、输入/填充/log-prob token及reserved显存详见JSON和原始costs.jsonl。", "",
              "C/D/E从同一个B检查点开始，16题训练池、代码快照、生成、奖励和优化配置相同；比较程序核对哈希。三组都产生了非零有限梯度及真实LoRA更新。", "",
              f"D做了{train['D']['probe_calls']}次探测、{train['D']['sampler_weight_change_events']}次后续权重变化；E仅初始探测。两组初始探测原始输出逐字节一致。D探测占本轮总时间{train['D']['probe_wall_fraction']:.1%}，尚未达到低开销的主实验设计。相同步数结果不能回答原研究问题。", "",
              "## 结果解释与下一步", "",
              "可见的负奖励样本主要暴露空答案标记、重复输出和截断；推导得到正确数值也可能因最终区域无效而获得0分。少量训练奖励和8题终点差异不支持推理能力提升结论。保留原始输出与评分规则，不因当前结果改评分器。", "",
              "动态组在首次刷新前的4步梯度均为0；该次权重变化不能归因为此前能力进步，K=2重采样噪声是需要检验的解释。静态组训练loss接近0却有真实参数更新，标量loss也不能单独判断GRPO是否学习。", "",
              "同样的初始探测，D/E实际用时不同，说明当前笔记本运行速度有波动。未记录的功耗/温度/其他进程负载不能事后猜测为原因；正式实验应增加设备状态记录、轮换组别顺序并保留多种子实际耗时。", "",
              "下一轮先校准更充分的SFT与验证规模，固定评分/生成协议，降低探测占比，再运行同一端到端时间预算的C/D/E。正式C/D至少seed13/29/47；最终冻结具体检查点后才开启GSM8K test与SVAMP。当前未进行正式对照或保留集评测。", "",
              "完整数据、前后向/重载与有限梯度审计见 [evidence_audit.json](evidence_audit.json)；错误实例见 [error_analysis.md](error_analysis.md)；当前交接见 [HANDOFF.md](../HANDOFF.md)。", ""]
    (output / "local_pilot_findings.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(output / "local_pilot_findings.md")


if __name__ == "__main__":
    main()
