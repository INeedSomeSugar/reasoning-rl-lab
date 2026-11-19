"""Summarize completed time-budget pilot runs and independent reloads; no model or holdout access."""
import argparse
import ast
import json
import statistics
from collections import defaultdict

from reasoning_rl.runtime import ROOT, bootstrap, json_write, jsonl_read
from reasoning_rl.reporting import write_csv, _plot_style


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    bootstrap()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="localbudget_20261105")
    parser.add_argument("--baseline", default="localpilot3_20261105_A_val")
    parser.add_argument("--sft-evaluation", default="localpilot2_20261105_B_val")
    args = parser.parse_args()
    output = ROOT / "reports/comparisons" / args.prefix
    output.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for arm, name in (("A", args.baseline), ("B", args.sft_evaluation)):
        directory = ROOT / "runs" / name
        baselines[arm] = read(directory / "validation/metrics.json")
        baselines[arm]["run"] = name
    reference_predictions = jsonl_read(ROOT / "runs" / args.sft_evaluation / "validation/predictions.jsonl")
    reference_ids = [r["id"] for r in reference_predictions]
    reference_config = read(ROOT / "runs" / args.sft_evaluation / "config.json")
    reference_precision = read(ROOT / "runs" / args.sft_evaluation / "model_precision.json")["standalone_loader"]

    def prompt_signature(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "render_prompt"
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"PROMPT", "PROMPT_VERSION"} for t in node.targets)]

    def verify_evaluation(directory, metric, baseline=False):
        config = read(directory / "config.json")
        for key in ("model", "dtype", "quantization", "data_seed", "generation", "evaluation"):
            if config[key] != reference_config[key]:
                raise ValueError("Endpoint evaluation configuration differs: " + key)
        if config.get("attention_implementation", "sdpa") != reference_config.get("attention_implementation", "sdpa"):
            raise ValueError("Endpoint attention implementation differs")
        precision = read(directory / "model_precision.json")["standalone_loader"]
        backbone_fields = ("input_embedding", "output_head", "norm_dtypes", "four_bit_compute_dtypes", "is_loaded_in_4bit")
        if any(precision[k] != reference_precision[k] for k in backbone_fields):
            raise ValueError("Standalone backbone precision differs")
        if not baseline and precision["lora_dtypes"] != reference_precision["lora_dtypes"]:
            raise ValueError("Standalone adapter precision differs")
        if metric["scorer_version"] != baselines["B"]["scorer_version"] or metric["thinking"] or metric["decoding"] != "greedy":
            raise ValueError("Scorer/decoding differs")
        # Runtime/logging fixes can differ between A/B. Prompt, parsing, and model revision must agree.
        for relative in ("src/reasoning_rl/scoring.py", "configs/models.lock.json"):
            if (directory / "source_snapshot" / relative).read_bytes() != (ROOT / "runs" / args.sft_evaluation / "source_snapshot" / relative).read_bytes():
                raise ValueError("Endpoint prompt/scorer/model source differs: " + relative)
        relative = "src/reasoning_rl/modeling.py"
        if prompt_signature(directory / "source_snapshot" / relative) != prompt_signature(ROOT / "runs" / args.sft_evaluation / "source_snapshot" / relative):
            raise ValueError("Endpoint prompt implementation differs")

    verify_evaluation(ROOT / "runs" / args.baseline, baselines["A"], baseline=True)
    verify_evaluation(ROOT / "runs" / args.sft_evaluation, baselines["B"])
    a_ids = [r["id"] for r in jsonl_read(ROOT / "runs" / args.baseline / "validation/predictions.jsonl")]
    if a_ids != reference_ids:
        raise ValueError("A/B validation IDs differ")
    rows, contracts = [], []
    for directory in sorted((ROOT / "runs").glob(args.prefix + "-*-s*")):
        if not (directory / "training_result.json").exists():
            continue
        evaluation = ROOT / "runs" / (directory.name + "-reloadval")
        if not (evaluation / "validation/metrics.json").exists():
            continue
        result = read(directory / "training_result.json")
        metric = read(evaluation / "validation/metrics.json")
        contract = read(directory / "comparison_contract.json")
        eval_protocol = read(evaluation / "evaluation_protocol.json")
        if eval_protocol["adapter_sha256"] != result["checkpoint_sha256"]:
            raise ValueError("Reload evaluated a different checkpoint")
        predictions = jsonl_read(evaluation / "validation/predictions.jsonl")
        if [r["id"] for r in predictions] != reference_ids:
            raise ValueError("GRPO endpoint validation IDs differ from A/B")
        verify_evaluation(evaluation, metric)
        costs = jsonl_read(directory / "costs.jsonl")
        total = next(c for c in costs if c["phase"] == "grpo_total")
        load_seconds = sum(c["wall_seconds"] for c in costs if c["phase"] == "load_model")
        probe_seconds = sum(c["wall_seconds"] for c in costs if c["phase"] == "difficulty_probe")
        validation_seconds = sum(c["wall_seconds"] for c in costs if c["phase"] == "validation")
        last_progress = [c for c in costs if c["phase"] == "budget_progress"][-1]
        reload_costs = jsonl_read(evaluation / "costs.jsonl")
        reload_seconds = sum(c["wall_seconds"] for c in reload_costs if c["phase"] in ("load_model", "validation"))
        cap = contract["protocol"]["grpo"]["max_wall_seconds"]
        rows.append({"run": directory.name, "strategy": contract["strategy"], "seed": contract["protocol"]["seed"],
                     "n": metric["n"], "correct": metric["correct"], "accuracy": metric["accuracy"],
                     "wilson95_low": metric["accuracy_wilson95"][0], "wilson95_high": metric["accuracy_wilson95"][1],
                     "mean_generated_tokens": metric["mean_generated_tokens"],
                     "truncation_rate": metric["truncation_rate"], "format_error_rate": metric["format_error_rate"],
                     "parse_error_rate": metric["parse_error_rate"],
                     "steps": result["completed_optimizer_steps"], "grpo_wall_seconds": total["wall_seconds"],
                     "actual_optimizer_steps_per_grpo_second": result["completed_optimizer_steps"]/total["wall_seconds"],
                     "actual_policy_generated_tokens_per_grpo_second": result["generated_tokens"]/total["wall_seconds"],
                     "declared_grpo_wall_cap": cap, "cap_minus_actual_seconds": cap - total["wall_seconds"],
                     "probe_wall_seconds": probe_seconds, "reload_eval_wall_seconds": reload_seconds,
                     "in_training_validation_seconds": validation_seconds,
                     "elapsed_before_final_validation_seconds": last_progress["cumulative_training_wall_seconds"],
                     "training_model_load_seconds": load_seconds,
                     "grpo_plus_reload_eval_seconds": total["wall_seconds"] + reload_seconds,
                     "load_grpo_reload_total_seconds": load_seconds + total["wall_seconds"] + reload_seconds,
                     "grpo_device_elapsed_seconds": total["device_elapsed_seconds"],
                     "generated_tokens": result["generated_tokens"], "processed_tokens": result["processed_tokens"],
                     "probe_generated_tokens": result["probe_generated_tokens"], "probe_prompt_tokens": result["probe_prompt_tokens"],
                     "probe_calls": result["probe_calls"], "weight_change_events": result["sampler_weight_change_events"],
                     "mean_reward": result["mean_reward"], "zero_variance_fraction": result["zero_variance_group_fraction"],
                     "adapter_delta_l2": result["adapter_delta_l2"], "adapter_updated": result["adapter_updated"],
                     "peak_allocated_gib": total["peak_allocated_bytes"] / 1024**3,
                     "peak_reserved_gib": total["peak_reserved_bytes"] / 1024**3})
        contracts.append(contract)
    if not rows:
        raise ValueError("No complete training + standalone reload pair")
    for key in ("sft_checkpoint_sha256", "data_manifest_sha256", "pool_ids", "code_sha256"):
        if any(c[key] != contracts[0][key] for c in contracts):
            raise ValueError("Comparison contract mismatch: " + key)
    canonical = {k: v for k, v in contracts[0]["protocol"].items() if k != "seed"}
    if any({k: v for k, v in c["protocol"].items() if k != "seed"} != canonical for c in contracts):
        raise ValueError("Training protocols differ beyond paired random seeds")
    b_protocol = read(ROOT / "runs" / args.sft_evaluation / "evaluation_protocol.json")
    if b_protocol["adapter_sha256"] != contracts[0]["sft_checkpoint_sha256"]:
        raise ValueError("B baseline is not the shared SFT checkpoint")
    groups = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)
    aggregates = []
    for strategy, items in groups.items():
        aggregates.append({"strategy": strategy, "seeds": [r["seed"] for r in items], "repeats": len(items),
                           "accuracy_mean": statistics.mean(r["accuracy"] for r in items),
                           "accuracy_sample_std": statistics.stdev(r["accuracy"] for r in items) if len(items) > 1 else None,
                           "grpo_seconds_mean": statistics.mean(r["grpo_wall_seconds"] for r in items)})
    paired = []
    for seed in sorted({r["seed"] for r in rows}):
        by_strategy = {r["strategy"]: r for r in rows if r["seed"] == seed}
        if {"uniform", "dynamic"} <= by_strategy.keys():
            c, d = by_strategy["uniform"], by_strategy["dynamic"]
            paired.append({"seed": seed, "dynamic_minus_uniform_pp": 100 * (d["accuracy"] - c["accuracy"]),
                           "dynamic_minus_uniform_grpo_seconds": d["grpo_wall_seconds"] - c["grpo_wall_seconds"]})
    json_write(output / "standalone_comparison.json", {"scope": "Development validation only; repeated questions across seeds; no independent test claim",
        "formal_comparison_completed": False, "baselines": baselines, "endpoints": rows,
        "aggregates": aggregates, "paired_seed_differences": paired,
        "cost_note": "Common GRPO-stage time cap with declared final-validation reserve; includes probes and in-training validation. Model loading and standalone reload evaluation are recorded separately. Listed phase sums exclude unmetered process startup and checkpoint serialization. A shared cap is not exact matched compute.",
        "throughput_note": "Use completed optimizer steps or real tokens divided by measured elapsed time. Raw Transformers speed fields can use planned max_steps after early stopping and are not adopted as observed throughput.",
        "source_snapshot_sha256": contracts[0]["code_sha256"]})
    write_csv(output / "standalone_endpoints.csv", rows)
    write_csv(output / "strategy_aggregates.csv", aggregates)
    write_csv(output / "paired_seed_differences.csv", paired)
    plt = _plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), layout="constrained")
    styles = {"uniform": ("#2563a6", "o"), "dynamic": ("#c07810", "s"), "static": ("#657d39", "^")}
    for strategy, items in groups.items():
        color, marker = styles[strategy]
        for row in items:
            axes[0].scatter(row["grpo_wall_seconds"] / 60, row["accuracy"] * 100, color=color, marker=marker, s=55)
            axes[0].annotate(f"{strategy[0].upper()}{row['seed']}", (row["grpo_wall_seconds"] / 60, row["accuracy"] * 100),
                             xytext=(4, 6), textcoords="offset points", fontsize=8)
        axes[0].scatter([], [], color=color, marker=marker, label=strategy)
    axes[0].axhline(100 * baselines["B"]["accuracy"], color="#475569", linestyle="--", label="shared SFT B")
    axes[0].set(title="Common standalone reload endpoints", xlabel="Actual GRPO time incl. probes/validation (min)",
                ylabel="Development accuracy (%)", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    positions = list(range(len(rows)))
    axes[1].bar(positions, [r["grpo_wall_seconds"] - r["probe_wall_seconds"] for r in rows], label="policy + in-training validation", color="#94a3b8")
    axes[1].bar(positions, [r["probe_wall_seconds"] for r in rows], bottom=[r["grpo_wall_seconds"] - r["probe_wall_seconds"] for r in rows],
                label="difficulty probes", color="#c07810", hatch="//")
    axes[1].set_xticks(positions, [f"{r['strategy'][0].upper()}{r['seed']}" for r in rows])
    axes[1].set(title="Observed budget allocation", ylabel="Wall time (seconds)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"RTX 3060 Laptop | {len(reference_ids)} fixed development questions | no final-test claim", fontsize=11)
    fig.savefig(output / "standalone_comparison.png", dpi=160)
    fig.savefig(output / "standalone_comparison.svg")
    plt.close(fig)
    lines = ["# 本机时间预算预实验", "", "由完成的训练和共同独立重载记录自动生成。仅开发验证，未启用最终测试。", "",
             "## A/B", "", "|组|正确|平均输出token|截断率|格式错误率|", "|---|---:|---:|---:|---:|"]
    for arm, m in baselines.items():
        lines.append(f"|{arm}|{m['correct']}/{m['n']}|{m['mean_generated_tokens']:.2f}|{m['truncation_rate']:.1%}|{m['format_error_rate']:.1%}|")
    lines += ["", "## 逐种子终点与实际成本", "", "|策略/seed|正确|步数|GRPO秒数|探测秒数|另计重载评测秒数|奖励均值|零方差率|LoRA delta|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['strategy']}/{r['seed']}|{r['correct']}/{r['n']}|{r['steps']}|{r['grpo_wall_seconds']:.1f}|{r['probe_wall_seconds']:.1f}|{r['reload_eval_wall_seconds']:.1f}|{r['mean_reward']:.4f}|{r['zero_variance_fraction']:.1%}|{r['adapter_delta_l2']:.6f}|")
    lines += ["", "## 重复与解释", ""]
    for a in aggregates:
        std = f"{100*a['accuracy_sample_std']:.2f}个百分点" if a["accuracy_sample_std"] is not None else "单次运行，未估计种子标准差"
        lines.append(f"- {a['strategy']}：实际{a['repeats']}次，seed={a['seeds']}，正确率均值{a['accuracy_mean']:.2%}；{std}。")
    lines += ["", "共同时间上限不能代替实际计算成本相等；验证预留、自然停止边界、未完成探测和模型输出长度均会影响端点。探测和训练内验证已包含在GRPO总时间，不重复相加。训练模型加载、独立重载评测的加载与生成时间另列；所列阶段之和不包含未计量的进程启动、Trainer准备和检查点序列化，共同SFT成本也单列。", "",
              "种子间标准差不代表题目泛化置信区间。同一批验证题已用于开发；正式结论需要冻结方案后独立测试。当前没有完成正式测试，不能把开发差异写成泛化提升。", "",
              "所有原始预测、选择、探测、梯度和源码快照均保留；更多输出长度/提取/格式/截断、token与显存字段见standalone_endpoints.csv。静态消融只按实际重复数报告。", ""]
    (output / "findings.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(output / "findings.md")


if __name__ == "__main__":
    main()
