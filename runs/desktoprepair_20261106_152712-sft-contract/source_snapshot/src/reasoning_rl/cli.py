"""Run with `python -m reasoning_rl` or the project-local `rrl` entry point."""
import argparse
import json
from pathlib import Path
from .runtime import ROOT, bootstrap, json_write, seed_everything


def read_config(path, args):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if getattr(args, "model", None):
        config["model"] = args.model
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed
    if getattr(args, "hardware", None) == "rocm":
        # No CUDA-specific bitsandbytes, flash-attn or vLLM requirement on AMD.
        precision = config.get("dtype", "float16") if config.get("quantization") == "none" else "float16"
        config["quantization"] = "none"
        # Preserve an explicitly recorded non-quantized fallback after FP16 diagnostics.
        config["dtype"] = precision
        if config["dtype"] not in {"float16", "bfloat16", "float32"}:
            raise ValueError("ROCm needs an explicit supported floating-point precision")
        config["hardware_profile"] = "rocm_unverified"
    return config


def main():
    bootstrap()
    parser = argparse.ArgumentParser(description="ReasoningRL Lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data")
    p = sub.add_parser("hardware")
    p.add_argument("--output", default=str(ROOT / "environment/hardware.json"))
    p = sub.add_parser("download-model")
    p.add_argument("--model", choices=["qwen3_17b", "qwen3_06b"], default="qwen3_17b")
    for command in ("check-model", "evaluate", "sft", "grpo", "freeze"):
        p = sub.add_parser(command)
        p.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
        p.add_argument("--model", choices=["qwen3_17b", "qwen3_06b"])
        p.add_argument("--seed", type=int)
        p.add_argument("--hardware", choices=["cuda", "rocm"], default="cuda")
        if command != "freeze":
            p.add_argument("--name", required=True)
        if command == "evaluate":
            p.add_argument("--adapter")
            p.add_argument("--split", choices=["validation", "test", "external"], default="validation")
            p.add_argument("--frozen-plan")
        if command == "grpo":
            p.add_argument("--sft-checkpoint", required=True)
            p.add_argument("--strategy", choices=["uniform", "dynamic", "static"], required=True)
        if command == "freeze":
            p.add_argument("--checkpoints", nargs="+", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "prepare-data":
        from .data import prepare
        manifest = prepare()
        print(json.dumps({s: {"count": x["count"], "sha256": x["sha256"]} for s, x in manifest["splits"].items()}, indent=2))
    elif args.command == "hardware":
        from .runtime import hardware_info
        info = hardware_info()
        output = Path(args.output).resolve()
        if not output.is_relative_to(ROOT.resolve()):
            raise ValueError("Hardware evidence must remain inside this project")
        json_write(output, info)
        print(json.dumps(info, indent=2))
    elif args.command == "download-model":
        from .modeling import download_model
        print(download_model(args.model))
    elif args.command in ("check-model", "sft", "grpo"):
        from . import training
        config = read_config(args.config, args)
        if args.command == "check-model":
            result = training.check_model(config, args.name)
        elif args.command == "sft":
            result = training.run_sft(config, args.name)
        else:
            result = training.run_grpo(config, args.name, args.sft_checkpoint, args.strategy)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .runtime import create_run
        from .modeling import load_model, precision_snapshot
        from .data import load_split
        from .evaluation import authorize_final, evaluate_rows, validation_rows
        config = read_config(args.config, args)
        seed_everything(config["seed"])
        if args.split != "validation":
            if not args.frozen_plan:
                parser.error("Holdout evaluation requires --frozen-plan")
            authorize_final(args.frozen_plan, config, args.adapter)
            claim = ROOT / "experiments/final_claims" / (Path(args.frozen_plan).stem+"_"+args.split+"_"+(Path(args.adapter).name if args.adapter else "public")+".json")
            if claim.exists():
                raise ValueError("Final evaluation already claimed; preserve results and investigate manually")
            # A claim is recorded BEFORE generation, even if interrupted. No repeated favorable draws.
            json_write(claim, {"run": args.name, "adapter": args.adapter, "frozen_plan": args.frozen_plan})
        directory, ledger = create_run(args.name, config)
        from .runtime import checkpoint_hash, sha256_file
        json_write(directory / "evaluation_protocol.json", {"adapter": args.adapter,
            "adapter_sha256": checkpoint_hash(args.adapter) if args.adapter else None,
            "split": args.split, "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
            "frozen_plan": args.frozen_plan, "thinking": False, "decoding": "greedy"})
        with ledger.measure("load_model"):
            model, tokenizer = load_model(config, adapter=args.adapter)
        json_write(directory / "model_precision.json", {"standalone_loader": precision_snapshot(model)})
        key = "validation_size" if args.split == "validation" else args.split+"_size"
        rows = validation_rows(config) if args.split == "validation" else load_split(
            args.split, "final", limit=config["evaluation"][key], seed=config["data_seed"])
        result = evaluate_rows(model, tokenizer, rows, config, directory / args.split, ledger, phase=args.split)
        print(json.dumps(result, indent=2))
    elif args.command == "freeze":
        from .evaluation import freeze_plan
        print(freeze_plan(read_config(args.config, args), args.checkpoints))
    elif args.command == "compare":
        from .reporting import compare_runs
        print(compare_runs(args.runs, args.name))
    elif args.command == "report":
        from .reporting import build_report
        print(build_report(args.runs))
