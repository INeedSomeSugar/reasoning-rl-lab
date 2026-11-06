"""Comparable decoding, raw predictions and independent final-test authorization."""
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from .runtime import ROOT, json_write, jsonl_write, jsonl_append, jsonl_read, sha256_file
from .modeling import render_prompt, autocast
from .scoring import score_answer, SCORER_VERSION


def validation_rows(config):
    """Resolve a preregistered development subset without permitting split leakage."""
    from .data import load_split
    rows = load_split("validation", "evaluate_validation", seed=config["data_seed"])
    options = config["evaluation"]
    if not options.get("validation_ids_path"):
        return rows[:options["validation_size"]]
    path = (ROOT / options["validation_ids_path"]).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Development ID manifest must be project-local")
    selected = json.loads(path.read_text(encoding="utf-8"))
    ids = selected[options.get("validation_subset", "comparison")]
    by_id = {r["id"]: r for r in rows}
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError("Development IDs must be unique members of validation")
    if len(ids) != options["validation_size"]:
        raise ValueError("Development size differs from preregistered IDs")
    return [by_id[i] for i in ids]


def diagnostic_tags(text, score, truncated):
    """Automatic surface diagnostics; arithmetic/condition errors require human review."""
    tags = []
    if truncated:
        tags.append("truncated")
    if not score.parse_ok:
        tags.append("answer_extraction_failure")
    if not score.format_ok:
        tags.append("format_error")
    grams = [tuple(text.split()[i:i+8]) for i in range(max(0, len(text.split())-7))]
    if grams and max(Counter(grams).values()) >= 3:
        tags.append("repetition_suspected")
    if not score.correct and score.parse_ok:
        tags.append("wrong_numeric_answer_requires_review")
    return tags


def generate_one(model, tokenizer, row, config, sample=False, count=1):
    import torch
    prompt = render_prompt(tokenizer, row["question"])
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens > config["generation"]["max_prompt_length"]:
        raise ValueError(f"Prompt exceeds frozen budget: {row['id']}")
    options = config["generation"]
    kwargs = {"max_new_tokens": options["max_new_tokens"], "do_sample": sample,
              "num_return_sequences": count, "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": tokenizer.eos_token_id, "use_cache": True}
    if sample:
        kwargs.update(temperature=options["temperature"], top_p=options["top_p"], top_k=options["top_k"])
    # Explicitly replace the model's inherited sampling defaults.
    from transformers import GenerationConfig
    generation_config = GenerationConfig(**kwargs)
    training = model.training
    model.eval()
    start = time.perf_counter()
    try:
        with torch.inference_mode(), autocast(config):
            generated = model.generate(**inputs, generation_config=generation_config,
                                       use_model_defaults=False, do_sample=sample)
    finally:
        model.train(training)
    results = []
    for tokens in generated[:, prompt_tokens:]:
        ids = tokens.tolist()
        eos = tokenizer.eos_token_id in ids
        if eos:
            ids = ids[:ids.index(tokenizer.eos_token_id)+1]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        score = score_answer(text, row["answer"])
        truncated = not eos and len(ids) >= options["max_new_tokens"]
        results.append({"id": row["id"], "split": row["split"], "prediction": text,
                        "gold_answer": row["answer"], "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(ids), "truncated": truncated, **score.as_dict(),
                        "diagnostic_tags": diagnostic_tags(text, score, truncated)})
    return results, {"prompt_tokens": prompt_tokens*count, "generated_tokens": sum(r["generated_tokens"] for r in results),
                     "wall_seconds": time.perf_counter()-start}


def aggregate(predictions):
    n = len(predictions)
    if not n:
        raise ValueError("No predictions")
    correct = sum(r["correct"] for r in predictions)
    p = correct/n
    z = 1.959963984540054
    middle = (p+z*z/(2*n))/(1+z*z/n)
    half = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/(1+z*z/n)
    lengths = sorted(r["generated_tokens"] for r in predictions)
    return {"n": n, "correct": correct, "accuracy": p, "accuracy_wilson95": [middle-half, middle+half],
            "format_error_rate": sum(not r["format_ok"] for r in predictions)/n,
            "parse_error_rate": sum(not r["parse_ok"] for r in predictions)/n,
            "truncation_rate": sum(r["truncated"] for r in predictions)/n,
            "mean_generated_tokens": sum(lengths)/n, "p50_generated_tokens": lengths[n//2],
            "generated_tokens": sum(lengths), "prompt_tokens": sum(r["prompt_tokens"] for r in predictions),
            "automatic_error_tags": dict(Counter(t for r in predictions for t in r["diagnostic_tags"]))}


def evaluate_rows(model, tokenizer, rows, config, directory, ledger, phase="validation", step=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "predictions.jsonl").exists():
        raise ValueError("Evaluation output already exists; preserve it and use a new run name")
    predictions = []
    with ledger.measure(phase, optimizer_step=step) as cost:
        for i, row in enumerate(rows):
            result, _ = generate_one(model, tokenizer, row, config, sample=False)
            predictions.extend(result)
            # Incremental evidence survives interruption.
            jsonl_append(directory / "predictions.jsonl", result)
            cost["prompt_tokens"] = cost.get("prompt_tokens", 0) + sum(r["prompt_tokens"] for r in result)
            cost["generated_tokens"] = cost.get("generated_tokens", 0) + sum(r["generated_tokens"] for r in result)
            cost["completed_predictions"] = len(predictions)
            print(f"{phase} {i+1}/{len(rows)} {row['id']} correct={result[0]['correct']} tokens={result[0]['generated_tokens']}", flush=True)
        metrics = aggregate(predictions)
        cost.update(prompt_tokens=metrics["prompt_tokens"], generated_tokens=metrics["generated_tokens"])
    metrics.update(phase=phase, optimizer_step=step, decoding="greedy", thinking=False, scorer_version=SCORER_VERSION)
    json_write(directory / "metrics.json", metrics)
    reviews = [{"id": r["id"], "prediction": r["prediction"], "gold_answer": r["gold_answer"],
                "automatic_tags": r["diagnostic_tags"], "human_reviewed": False,
                "arithmetic_error": None, "condition_omission": None,
                "other_error": None, "review_notes": ""} for r in predictions if not r["correct"]]
    jsonl_write(directory / "error_review.jsonl", reviews)
    return metrics


def freeze_plan(config, checkpoint_paths):
    from .runtime import checkpoint_hash, code_hash
    from .modeling import model_spec, PROMPT_VERSION
    if not checkpoint_paths:
        raise ValueError("Freeze concrete checkpoints, including public baseline via an empty adapter only in config")
    digest = sha256_file(ROOT / "data/manifest.json")
    if any((ROOT / "experiments/final_claims").glob("*.json")):
        raise ValueError("Holdout already opened: do not register a new favorable plan against the same test set")
    payload = {"config": config, "data_manifest_sha256": digest,
               "code_sha256": code_hash(), "model_spec": model_spec(config["model"]),
               "prompt_version": PROMPT_VERSION, "scorer_version": SCORER_VERSION,
               "checkpoints": {str(Path(p).resolve()): checkpoint_hash(p) for p in checkpoint_paths},
               "decoding": "greedy", "thinking": False,
               "external": "svamp", "selection_rule": "declared fixed endpoints; final test is single-use"}
    identifier = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = ROOT / "experiments/frozen" / (identifier + ".json")
    if not path.exists():
        json_write(path, payload)
    return path


def authorize_final(frozen_path, config, adapter):
    from .runtime import checkpoint_hash, code_hash
    from .modeling import model_spec, PROMPT_VERSION
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    if frozen["config"] != config or frozen["data_manifest_sha256"] != sha256_file(ROOT / "data/manifest.json"):
        raise ValueError("Frozen plan changed")
    if any((frozen["code_sha256"] != code_hash(), frozen["model_spec"] != model_spec(config["model"]),
            frozen["prompt_version"] != PROMPT_VERSION, frozen["scorer_version"] != SCORER_VERSION)):
        raise ValueError("Code/model/prompt/scorer changed after final freeze")
    if adapter is not None:
        path = str(Path(adapter).resolve())
        if frozen["checkpoints"].get(path) != checkpoint_hash(adapter):
            raise ValueError("Checkpoint not in frozen final plan")
    return frozen
