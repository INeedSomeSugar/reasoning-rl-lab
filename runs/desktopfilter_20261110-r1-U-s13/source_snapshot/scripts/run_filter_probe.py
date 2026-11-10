"""Paid fixed-S2 K4 probe; write every raw output before constructing a subset."""
import argparse
from collections import Counter
import json
import time
import yaml
from reasoning_rl.runtime import ROOT, bootstrap, create_run, checkpoint_hash, seed_everything, json_write, jsonl_append
from reasoning_rl.data import load_split
from reasoning_rl.modeling import load_model, precision_snapshot
from reasoning_rl.evaluation import generate_one
from reasoning_rl.prefiltering import raw_success_filter
from filter_common import EXP, FAMILY, START, START_HASH, read


def main(calibration=False):
    bootstrap()
    reg = read(EXP / "calibration_preregistration.json")
    ids = reg["calibration_probe_ids"] if calibration else read(EXP / "candidate_pool.json")["ids"]
    config = yaml.safe_load((ROOT / "configs" / f"{FAMILY}_calibration.yaml").read_text())
    label = "calibration-probe" if calibration else "probe"
    directory, ledger = create_run(FAMILY + "-" + label, config)
    assert checkpoint_hash(ROOT / START) == START_HASH
    seed_everything(reg["probe_seed"])
    by_id = {r["id"]: r for r in load_split("train", "select", seed=config["data_seed"])}
    predictions = []
    with ledger.measure("fixed_s2_probe_total"):
        with ledger.measure("load_fixed_s2"):
            model, tokenizer = load_model(config, adapter=ROOT / START)
        json_write(directory / "model_precision.json", precision_snapshot(model))
        with ledger.measure("fixed_s2_k4_generation") as counts:
            for i, identifier in enumerate(ids):
                rows, cost = generate_one(model, tokenizer, by_id[identifier], config, sample=True, count=reg["probe_k"])
                for j, r in enumerate(rows):
                    r.update(sample_index=j, probe_seed=reg["probe_seed"])
                predictions.extend(rows)
                jsonl_append(directory / "predictions.jsonl", rows)
                counts.update(completed_items=i+1, generated_tokens=sum(r["generated_tokens"] for r in predictions),
                    prompt_tokens=sum(r["prompt_tokens"] for r in predictions))
                print(f"probe {i+1}/{len(ids)} {identifier} successes={sum(r['correct'] for r in rows)}/4", flush=True)
        retained, successes = raw_success_filter(ids, predictions, reg["probe_k"])
        result = {"candidate_ids": ids, "retained_ids": retained, "raw_success_counts": successes,
            "raw_success_histogram": dict(sorted(Counter(successes.values()).items())),
            "probe_seed": reg["probe_seed"], "k": 4, "s2_checkpoint_sha256": START_HASH,
            "generated_tokens": sum(r["generated_tokens"] for r in predictions),
            "prompt_tokens": sum(r["prompt_tokens"] for r in predictions),
            "parse_failures": sum(not r["parse_ok"] for r in predictions),
            "truncations": sum(r["truncated"] for r in predictions),
            "data_scope": "train only; these stochastic rates are distinct from development greedy accuracy"}
        json_write(directory / "probe_result.json", result)
    print(json.dumps({"items": len(ids), "retained": len(retained), "histogram": result["raw_success_histogram"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", action="store_true")
    main(parser.parse_args().calibration)
