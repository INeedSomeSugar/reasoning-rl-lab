"""Generate comparison tables/figures only from saved real runs."""
import csv
import json
from pathlib import Path
from .runtime import ROOT, json_write, jsonl_read


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 150,
                         "svg.fonttype": "none"})
    return plt


def compare_runs(names, name):
    directories = [ROOT / "runs" / n for n in names]
    contracts = [read_json(p / "comparison_contract.json") for p in directories]
    canonical = dict(contracts[0]["protocol"])
    canonical.pop("seed", None)
    for contract in contracts:
        protocol = dict(contract["protocol"])
        protocol.pop("seed", None)
        if protocol != canonical:
            raise ValueError("Comparison protocols differ beyond the declared GRPO seed")
        for field in ("sft_checkpoint_sha256", "data_manifest_sha256", "pool_ids", "code_sha256"):
            if contract[field] != contracts[0][field]:
                raise ValueError(f"Unfair comparison: {field} differs")
    environments = [read_json(p / "environment.json") for p in directories]
    hardware_keys = [(e["gpus"][0]["name"] if e["gpus"] else "CPU", e["torch_cuda"], e["torch_hip"], e["versions"]["torch"])
                     for e in environments]
    if len(set(hardware_keys)) != 1:
        raise ValueError("Different hardware/build: split timing comparisons into separate reports")
    output = ROOT / "reports/comparisons" / name
    output.mkdir(parents=True, exist_ok=True)
    summary, points = [], []
    exposure_rows = []
    plt = _plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    styles = {"uniform": ("#2563a6", "o"), "dynamic": ("#c07810", "s"), "static": ("#657d39", "^")}
    for directory, contract in zip(directories, contracts, strict=True):
        result = read_json(directory / "training_result.json")
        ledger = jsonl_read(directory / "costs.jsonl")
        total = next(r for r in ledger if r["phase"] == "grpo_total")
        progress = {r["optimizer_step"]: r for r in ledger if r["phase"] == "budget_progress"}
        strategy, seed = contract["strategy"], contract["protocol"]["seed"]
        evaluations = []
        for path in sorted((directory / "validation").glob("*/metrics.json")):
            metrics = read_json(path)
            step = metrics["optimizer_step"]
            if step not in progress:
                continue
            validation_cost = next((r.get("wall_seconds", 0) for r in ledger if r["phase"] == "validation" and r.get("optimizer_step") == step), 0)
            point = {"run": directory.name, "strategy": strategy, "seed": seed, "optimizer_step": step,
                     "evaluation_context": "in_training_diagnostic",
                     "cumulative_wall_seconds": progress[step]["cumulative_training_wall_seconds"]+validation_cost,
                     "accuracy": metrics["accuracy"], "n": metrics["n"],
                     "wilson95_low": metrics["accuracy_wilson95"][0], "wilson95_high": metrics["accuracy_wilson95"][1]}
            points.append(point)
            evaluations.append(point)
        row = {"run": directory.name, "strategy": strategy, "seed": seed,
               "steps": result["completed_optimizer_steps"], "wall_seconds": total["wall_seconds"],
               "device_elapsed_seconds": total["device_elapsed_seconds"],
               "peak_allocated_gib": total["peak_allocated_bytes"]/(1024**3) if total["peak_allocated_bytes"] else None,
               "policy_loss_input_tokens": result["processed_tokens"],
               "policy_generated_tokens": result["generated_tokens"],
               "probe_generated_tokens": result["probe_generated_tokens"],
               "probe_prompt_tokens": result["probe_prompt_tokens"],
               "probe_calls": result["probe_calls"], "mean_reward": result["mean_reward"],
               "zero_variance_fraction": result["zero_variance_group_fraction"],
               "adapter_delta_l2": result["adapter_delta_l2"],
               "validation_accuracy": evaluations[-1]["accuracy"] if evaluations else None,
               "validation_n": evaluations[-1]["n"] if evaluations else None}
        summary.append(row)
        from .data import load_split
        train_by_id = {r["id"]: r for r in load_split("train", "develop")}
        pool_rows = [train_by_id[i] for i in contract["pool_ids"]]
        selections = jsonl_read(directory / "selections.jsonl") if (directory / "selections.jsonl").exists() else []
        draws = [i for batch in selections for i in batch["ids"]]
        median_length = sorted(len(r["question"].split()) for r in pool_rows)[len(pool_rows)//2]
        tags = {
            "solution_contains_division": lambda r: "/" in r["solution"],
            "solution_contains_multiplication": lambda r: "*" in r["solution"],
            "question_above_pool_median_length": lambda r: len(r["question"].split()) > median_length,
        }
        for tag, predicate in tags.items():
            eligible = {r["id"] for r in pool_rows if predicate(r)}
            pool_share = len(eligible)/len(pool_rows)
            exposure_share = sum(i in eligible for i in draws)/len(draws) if draws else None
            exposure_rows.append({"run": directory.name, "strategy": strategy, "seed": seed, "feature": tag,
                "pool_count": len(eligible), "pool_n": len(pool_rows), "pool_share": pool_share,
                "selected_count": sum(i in eligible for i in draws), "draws": len(draws),
                "exposure_share": exposure_share,
                "relative_exposure": exposure_share/pool_share if exposure_share is not None and pool_share else None,
                "interpretation": "Overlapping heuristic features, not human topic labels; short runs are diagnostic only"})
        color, marker = styles[strategy]
        label = f"{strategy}, seed {seed}"
        if evaluations:
            axes[0].errorbar([p["cumulative_wall_seconds"]/60 for p in evaluations],
                [p["accuracy"]*100 for p in evaluations],
                yerr=[[100*(p["accuracy"]-p["wilson95_low"]) for p in evaluations],
                      [100*(p["wilson95_high"]-p["accuracy"]) for p in evaluations]],
                marker=marker, color=color, linestyle="-" if len(evaluations) > 1 else "none", label=label, capsize=3)
        logs = [r for r in ledger if r["phase"] == "grpo_log" and "reward" in r]
        if logs:
            axes[1].plot([r["optimizer_step"] for r in logs], [r["reward"] for r in logs],
                         marker=marker, color=color, label=label)
        axes[2].bar(label, result["probe_generated_tokens"], color=color, hatch="//" if strategy == "static" else None)
    axes[0].set(title="In-training diagnostic accuracy", xlabel="End-to-end GRPO time (min)", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    axes[1].set(title="Training correctness reward", xlabel="Optimizer step", ylabel="Mean 0/1 reward", ylim=(-0.03, 1.03))
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    axes[2].set(title="Difficulty-probe overhead", ylabel="Generated probe tokens", ylim=(0, None))
    axes[2].tick_params(axis="x", labelrotation=20)
    stage = canonical["stage"]
    fig.suptitle(f"{stage} | {hardware_keys[0][0]} | validation only; no final-test claim", fontsize=12)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "accuracy_cost_points.csv", points)
    write_csv(output / "sampling_exposure.csv", exposure_rows)
    seeds = {s: sorted(r["seed"] for r in summary if r["strategy"] == s) for s in styles}
    paired = sorted(set(seeds["uniform"]) & set(seeds["dynamic"]))
    formal = len(paired) >= 3 and not ("smoke" in stage or "provisional" in stage)
    json_write(output / "comparison.json", {"summary": summary, "points": points, "paired_seeds": paired,
        "formal_comparison_completed": False,
        "three_seed_validation_comparison": formal,
        "scope": "In-training diagnostic validation only; compare endpoints using common standalone reloads. Formal claim requires frozen holdout evaluation.",
        "sft_checkpoint_sha256": contracts[0]["sft_checkpoint_sha256"], "hardware": hardware_keys[0],
        "cost_fairness": "Smoke endpoints need not use equal compute; plots expose actual probe/time overhead."})
    return output


def build_report(names=None):
    directories = [ROOT/"runs"/n for n in names] if names else sorted(p for p in (ROOT/"runs").glob("*") if p.is_dir())
    rows, runs, incomplete = [], [], []
    for directory in directories:
        if (directory / "invalidated.json").exists():
            continue
        for path in directory.rglob("metrics.json"):
            if "trainer" in path.parts:
                continue
            metric = read_json(path)
            if "accuracy" in metric:
                context = "in_training_diagnostic" if (directory/"comparison_contract.json").exists() else "standalone_reload"
                rows.append({"run": directory.name, "path": str(path.relative_to(ROOT)), "evaluation_context": context, **metric})
        result = directory / "training_result.json"
        if result.exists():
            runs.append({"run": directory.name, **read_json(result)})
        elif (directory / "provenance.json").exists() and not (directory / "capability.json").exists() and not any(r["run"] == directory.name for r in rows):
            costs = jsonl_read(directory / "costs.jsonl") if (directory / "costs.jsonl").exists() else []
            failed = [c for c in costs if c.get("success") is False]
            interruption = read_json(directory / "interrupted.json") if (directory / "interrupted.json").exists() else {}
            incomplete.append({"run": directory.name, "status": "no_completed_training_or_evaluation_record",
                               "failed_phases": [c["phase"] for c in failed],
                               "reason": interruption.get("reason") or next((c.get("device_timing_error") for c in failed if c.get("device_timing_error")), "See saved logs; no completed result"),
                               "recovery_checkpoint_receipts": len(list((directory / "recovery_checkpoints").glob("*.json")))})
    output = ROOT/"reports"
    output.mkdir(exist_ok=True)
    json_write(output/"all_metrics.json", {"evaluations": rows, "training": runs, "incomplete_runs": incomplete})
    write_csv(output/"evaluation_summary.csv", [{k: r.get(k) for k in ("run", "phase", "evaluation_context", "optimizer_step", "n", "correct", "accuracy", "mean_generated_tokens", "truncation_rate", "format_error_rate", "parse_error_rate")} for r in rows])
    text = ["# 实验报告：实际运行快照", "",
            "本报告由保存的原始指标自动生成。smoke和少量验证题仅用于检查链路；尚未完成正式多种子独立测试的部分不作提升结论。", "",
            "本机首轮研究发现与统一重载终点见 [local_pilot_findings.md](local_pilot_findings.md)，[终点和探测开销图](local_pilot_endpoints.png)。实际库精度变化见 [trl_precision_audit.json](trl_precision_audit.json)。", "",
            "## 实际评测", "", "|运行|划分/阶段|评测上下文|题数|正确|正确率|平均生成token|截断率|格式错误率|",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"|{r['run']}|{r.get('phase')}|{r['evaluation_context']}|{r['n']}|{r['correct']}|{r['accuracy']:.1%}|{r['mean_generated_tokens']:.1f}|{r['truncation_rate']:.1%}|{r['format_error_rate']:.1%}|")
    if not rows:
        text.append("|尚无完成的模型评测|—|—|—|—|—|—|—|—|")
    text += ["", "## 实际训练", "", "|运行|阶段/步数|LoRA delta L2|更新生效|检查点|", "|---|---|---:|---|---|"]
    for r in runs:
        steps = r.get("completed_optimizer_steps", "SFT")
        text.append(f"|{r['run']}|{steps}|{r.get('adapter_delta_l2', 0):.6g}|{r.get('adapter_updated')}|`{r.get('checkpoint')}`|")
    comparisons = sorted((output / "comparisons").glob("*/standalone_comparison.json"))
    if comparisons:
        text += ["", "## 共同独立重载后的开发对照", ""]
        for path in comparisons:
            record = read_json(path)
            text.append(f"- [{path.parent.name}](comparisons/{path.parent.name}/findings.md)：{len(record['endpoints'])}个完成终点；开发验证，正式独立测试状态另列。")
    if incomplete:
        text += ["", "## 未形成完成结果的运行", "", "|运行|失败阶段|恢复检查点记录数|说明|", "|---|---|---:|---|"]
        for r in incomplete:
            reason = str(r["reason"]).splitlines()[0].replace("|", "/")
            text.append(f"|{r['run']}|{','.join(r['failed_phases']) or '未完成'}|{r['recovery_checkpoint_receipts']}|{reason}|")
    text += ["", "## 解释边界与错误分析", "",
             "当前数据划分、评分与训练配置见 README、PROTOCOL.md。逐题原始输出、自动诊断与待人工审阅队列位于每个评测目录。只有实际审阅的题才可归因为算术错误或条件遗漏。", "",
             "训练loss下降、少量训练奖励变动及不同计算成本的smoke结果，都不等同于泛化提升。动态探测的全部开销保留在costs.jsonl和comparison图中；grpo_total包含子阶段，不能重复相加。", "",
             "官方GSM8K测试与SVAMP保留集是否已启用，以experiments/final_claims和对应原始预测为准；缺少正式评测时不填猜测指标。预训练污染未知。", "",
             "人工错误分析及失败归因补充在 [error_analysis.md](error_analysis.md)，当前完成边界见 [HANDOFF.md](../HANDOFF.md)。", ""]
    (output/"experiment_report.md").write_text("\n".join(text), encoding="utf-8")
    return output/"experiment_report.md"
