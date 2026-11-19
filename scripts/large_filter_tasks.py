"""One GPU task per independent process: old-data diagnostic or fixed-S2 K4 probe."""
import argparse
from collections import Counter
import gc
import json
import yaml
from reasoning_rl.runtime import (ROOT, bootstrap, create_run, seed_everything, checkpoint_hash,
    sha256_file, json_write, jsonl_append)
from reasoning_rl.data import load_split
from reasoning_rl.modeling import load_model, precision_snapshot, encode_sft, CompletionCollator, autocast, save_adapter
from reasoning_rl.evaluation import generate_one, validation_rows
from reasoning_rl.prefiltering import raw_success_filter
from large_filter_common import START, START_HASH, read


def preflight(family):
    import torch
    from reasoning_rl.training import _snapshot_trainable, _update_evidence, _gradient_evidence
    exp = ROOT / "experiments" / family
    config = yaml.safe_load((ROOT / "configs" / f"{family}_calibration.yaml").read_text())
    directory, ledger = create_run(family+"-preflight-a0", config)
    seed_everything(13)
    from reasoning_rl.runtime import hardware_info
    hardware = hardware_info()
    assert len(hardware["gpus"]) == 1 and "9070 XT" in hardware["gpus"][0]["name"]
    model_files = []
    for item in read(ROOT / "experiments/desktoprepair_20261106_152712/public_model_verification.json")["files"]:
        path = ROOT / "models/qwen3_17b" / item["name"]
        assert sha256_file(path) == item["sha256"]
        model_files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": item["sha256"]})
    assert checkpoint_hash(ROOT / START) == START_HASH
    with ledger.measure("load_fixed_s2"):
        model, tok = load_model(config, adapter=ROOT / START, trainable=True)
    precision = precision_snapshot(model)
    candidates = [(r, encode_sft(tok, r, config["sft"]["max_length"], config["generation"]["max_prompt_length"]))
        for r in validation_rows(config)]
    row, encoded = min(((r, e) for r, e in candidates if e is not None), key=lambda x: len(x[1]["input_ids"]))
    batch = {k: v.to(model.device) for k, v in CompletionCollator(tok.pad_token_id)([encoded]).items()}
    initial = _snapshot_trainable(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6, weight_decay=0)
    with ledger.measure("old_development_throwaway_backward"):
        model.train()
        with autocast(config):
            loss = model(**batch).loss
        loss.backward()
        grad = _gradient_evidence(model)
        assert grad["nonzero_gradient_elements"] > 0
        optimizer.step()
        after_signal = _snapshot_trainable(model)
        optimizer.zero_grad(set_to_none=False)
        zero_grad = _gradient_evidence(model)
        assert zero_grad["nonzero_gradient_elements"] == 0
        optimizer.step()
        zero_update = _update_evidence(model, after_signal)
        total_update = _update_evidence(model, initial)
        assert zero_update["adapter_updated"]
    model.eval()
    with torch.inference_mode(), autocast(config):
        expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits.float().cpu()
    path = ROOT / "checkpoints" / (family+"-preflight-a0")
    save_adapter(model, tok, path, config, "diagnostic_only_old_development_never_main_start")
    loss_value = float(loss)
    del model, optimizer, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    with ledger.measure("reload_throwaway_diagnostic"):
        model, tok = load_model(config, adapter=path)
        batch = {k: v.to(model.device) for k, v in CompletionCollator(tok.pad_token_id)([encoded]).items()}
        model.eval()
        with torch.inference_mode(), autocast(config):
            actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits.float().cpu()
    assert torch.equal(expected, actual)
    assert checkpoint_hash(ROOT / START) == START_HASH
    result = {"hardware": hardware, "precision": precision, "public_model_files": model_files,
        "fixed_s2_sha256": START_HASH, "fixed_s2_unchanged": True, "loss": loss_value,
        "example_id": row["id"], "source": "already-used historical development; disposable diagnostic weights",
        "nonzero_backward": grad, "zero_grad": zero_grad, "zero_gradient_adam_update": zero_update,
        "total_diagnostic_update": total_update, "reload_logits_max_absolute_difference": float((expected-actual).abs().max()),
        "exact_sampler_resume_supported": False, "fresh_development_accessed": False}
    json_write(directory / "preflight.json", result)
    json_write(exp / "preflight.json", result)
    print(json.dumps({"preflight": "passed", "reload_max_difference": 0, "old_development_id": row["id"]}))


def probe(family, seed, attempt):
    exp = ROOT / "experiments" / family
    config = yaml.safe_load((ROOT / "configs" / f"{family}_calibration.yaml").read_text())
    ids = read(exp / "candidate_pool.json")["ids"]
    directory, ledger = create_run(f"{family}-probe-j{seed}-a{attempt}", config)
    assert checkpoint_hash(ROOT / START) == START_HASH
    seed_everything(seed)
    by_id = {r["id"]: r for r in load_split("train", "select", seed=config["data_seed"])}
    predictions = []
    with ledger.measure("fixed_s2_probe_total"):
        with ledger.measure("load_fixed_s2"):
            model, tokenizer = load_model(config, adapter=ROOT / START)
        json_write(directory / "model_precision.json", precision_snapshot(model))
        with ledger.measure("fixed_s2_k4_generation") as counts:
            for i, identifier in enumerate(ids):
                rs, _ = generate_one(model, tokenizer, by_id[identifier], config, sample=True, count=4)
                for idx, row in enumerate(rs):
                    row.update(sample_index=idx, probe_seed=seed)
                predictions.extend(rs)
                jsonl_append(directory / "predictions.jsonl", rs)
                counts.update(completed_items=i+1, generated_tokens=sum(r["generated_tokens"] for r in predictions),
                    prompt_tokens=sum(r["prompt_tokens"] for r in predictions))
                print(f"probe {seed} {i+1}/{len(ids)} {identifier} successes={sum(r['correct'] for r in rs)}/4", flush=True)
        retained, successes = raw_success_filter(ids, predictions, 4)
        json_write(directory / "probe_result.json", {"candidate_ids": ids, "retained_ids": retained,
            "raw_success_counts": successes, "raw_success_histogram": dict(sorted(Counter(successes.values()).items())),
            "probe_seed": seed, "k": 4, "s2_checkpoint_sha256": START_HASH,
            "generated_tokens": sum(r["generated_tokens"] for r in predictions), "prompt_tokens": sum(r["prompt_tokens"] for r in predictions),
            "parse_failures": sum(not r["parse_ok"] for r in predictions), "truncations": sum(r["truncated"] for r in predictions)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--task", choices=["preflight", "probe"], required=True)
    parser.add_argument("--probe-seed", type=int)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    bootstrap()
    if args.task == "preflight":
        preflight(args.family)
    else:
        probe(args.family, args.probe_seed, args.attempt)
