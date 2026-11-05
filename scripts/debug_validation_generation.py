"""Single validation-item CUDA diagnosis, never a held-out evaluation or a speed benchmark."""
import argparse
import json
import os
import time

import yaml

from reasoning_rl.runtime import ROOT, bootstrap, json_write, jsonl_read


def main():
    bootstrap()
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    from reasoning_rl.modeling import load_model
    from reasoning_rl.data import load_split
    from reasoning_rl.evaluation import generate_one
    config = yaml.safe_load((ROOT / "configs/local_pilot.yaml").read_text(encoding="utf-8"))
    row = next(r for r in load_split("validation", "evaluate_validation") if r["id"] == args.id)
    model, tokenizer = load_model(config)
    results, counts = generate_one(model, tokenizer, row, config, sample=False)
    output = {"name": args.name, "time_unix": time.time(), "item": args.id,
              "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
              "scope": "Validation-only diagnosis, not a complete baseline or throughput claim",
              "predictions": results, "counts": counts}
    json_write(ROOT / "reports/diagnostics" / (args.name + ".json"), output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
