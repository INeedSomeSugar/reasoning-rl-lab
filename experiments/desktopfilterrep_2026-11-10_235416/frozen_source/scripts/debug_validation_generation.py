"""Single validation-item CUDA diagnosis, never a held-out evaluation or a speed benchmark."""
import argparse
import json
import os
import time

import yaml

from reasoning_rl.runtime import ROOT, bootstrap, json_write, checkpoint_hash, sha256_file


def main():
    bootstrap()
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="configs/local_pilot.yaml")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    from reasoning_rl.modeling import load_model, precision_snapshot
    from reasoning_rl.data import load_split
    from reasoning_rl.evaluation import generate_one
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    row = next(r for r in load_split("validation", "evaluate_validation") if r["id"] == args.id)
    path = ROOT / "reports/diagnostics" / (args.name + ".json")
    if path.exists():
        raise FileExistsError("Preserve previous diagnostic; use a new name")
    output = {"name": args.name, "time_unix": time.time(), "item": args.id,
              "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
              "scope": "Validation-only diagnosis, not a complete baseline or throughput claim",
              "config": config, "config_sha256": sha256_file(config_path), "adapter": args.adapter,
              "adapter_sha256": checkpoint_hash(args.adapter) if args.adapter else None,
              "success": False}
    json_write(path, output)
    try:
        model, tokenizer = load_model(config, adapter=args.adapter)
        output["model_precision"] = precision_snapshot(model)
        results, counts = generate_one(model, tokenizer, row, config, sample=False)
        output.update(predictions=results, counts=counts, success=True)
    except BaseException as error:
        output.update(error_type=type(error).__name__, error=str(error))
        json_write(path, output)
        raise
    json_write(path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
