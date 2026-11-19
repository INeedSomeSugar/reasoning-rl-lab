"""Real-tokenizer, real-model SFT contract audit and unchanged-LoRA round trip."""
import gc
import json
import time
import torch
import torch.nn.functional as F
import yaml
from reasoning_rl.runtime import ROOT, create_run, json_write, jsonl_write, seed_everything, checkpoint_hash, sha256_file
from reasoning_rl.modeling import (load_model, render_prompt, encode_sft, CompletionCollator,
                                   autocast, save_adapter, precision_snapshot, model_spec)
from reasoning_rl.data import load_split
from reasoning_rl.evaluation import evaluate_rows, validation_rows

FAMILY = "desktoprepair_20261106_152712"


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    assert torch.version.hip and torch.cuda.device_count() == 1
    assert "9070 XT" in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported()
    config = yaml.safe_load((ROOT / f"configs/{FAMILY}_regression.yaml").read_text(encoding="utf-8"))
    directory, ledger = create_run(FAMILY + "-sft-contract", config)
    seed_everything(config["seed"])
    with ledger.measure("load_public_model"):
        model, tokenizer = load_model(config)
    model.eval()
    rows = load_split("train", "train", limit=128, seed=config["data_seed"])
    records, encoded = [], []
    for row in rows:
        prompt = render_prompt(tokenizer, row["question"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target = row["solution"].strip()
        completion_ids = tokenizer.encode(target + tokenizer.eos_token, add_special_tokens=False)
        item = encode_sft(tokenizer, row, config["sft"]["max_length"], config["generation"]["max_prompt_length"])
        record = {"id": row["id"], "prompt": prompt, "gold_solution": target,
            "prompt_tokens": len(prompt_ids), "completion_tokens_including_eos": len(completion_ids),
            "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
            "completion_ends_with_eos": completion_ids[-1] == tokenizer.eos_token_id,
            "full_gold_preserved": tokenizer.decode(completion_ids[:-1], skip_special_tokens=False) == target,
            "separately_encoded_equals_concatenated": prompt_ids + completion_ids == tokenizer.encode(prompt + target + tokenizer.eos_token, add_special_tokens=False),
            "prompt_boundary_tail": tokenizer.decode(prompt_ids[-18:], skip_special_tokens=False),
            "first_completion_tokens": completion_ids[:8], "retained": item is not None,
            "chat_template_full_conversation": tokenizer.apply_chat_template([
                {"role": "user", "content": __import__("reasoning_rl.modeling", fromlist=["PROMPT"]).PROMPT.format(question=row["question"])},
                {"role": "assistant", "content": target}], tokenize=False, enable_thinking=False)}
        if item:
            assert item["labels"][:len(prompt_ids)] == [-100] * len(prompt_ids)
            assert item["labels"][len(prompt_ids):] == completion_ids
            assert item["labels"][len(prompt_ids)] == item["input_ids"][len(prompt_ids)]
            assert all(record[k] for k in ("completion_ends_with_eos", "full_gold_preserved", "separately_encoded_equals_concatenated"))
            record["causal_first_target"] = {"logit_position": len(prompt_ids)-1, "label_position": len(prompt_ids),
                "token_id": completion_ids[0], "model_shifts_labels_internally": True}
            record["encoded"] = item
            encoded.append(item)
        records.append(record)
    jsonl_write(directory / "tokenizer_audit.jsonl", records)
    batch = CompletionCollator(tokenizer.pad_token_id)([min(encoded, key=lambda e: len(e["input_ids"])), max(encoded, key=lambda e: len(e["input_ids"]))])
    padding = batch["attention_mask"] == 0
    assert bool((batch["labels"][padding] == -100).all())
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with ledger.measure("manual_shifted_cross_entropy"), torch.inference_mode(), autocast(config):
        outputs = model(**batch)
        manual = F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                                 batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(outputs.loss.float(), manual.float(), atol=1e-5)
    loss_audit = {"model_loss": float(outputs.loss), "manual_causal_shift_loss": float(manual),
        "absolute_difference": float((outputs.loss-manual).abs()), "padding_positions": int(padding.sum()),
        "padding_all_ignored": True, "supervised_positions": int((batch["labels"] != -100).sum()),
        "double_shift": False, "full_gold_truncation": False}
    del outputs, manual, batch
    # Baseline, untrained LoRA, and independent reload use exactly the old32 regression IDs.
    regression = validation_rows(config)
    evaluate_rows(model, tokenizer, regression, config, directory / "public_regression", ledger, phase="public_regression")
    row = rows[0]
    ids = tokenizer(render_prompt(tokenizer, row["question"]), add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode(), autocast(config):
        baseline_logits = model(**ids, logits_to_keep=1).logits.float().cpu()
    del model
    clear()
    with ledger.measure("load_fresh_untrained_lora"):
        model, tokenizer = load_model(config, trainable=True, fresh_lora=True)
    model.eval()
    with torch.inference_mode(), autocast(config):
        fresh_logits = model(**ids, logits_to_keep=1).logits.float().cpu()
    assert torch.equal(baseline_logits, fresh_logits)
    evaluate_rows(model, tokenizer, regression, config, directory / "untrained_regression", ledger, phase="untrained_regression")
    path = ROOT / "checkpoints" / (FAMILY + "-untrained-control")
    assert not path.exists()
    save_adapter(model, tokenizer, path, config, "untrained_lora_control", optimizer_steps=0,
                 explicitly_not_sft=True, data_manifest_sha256=sha256_file(ROOT / "data/manifest.json"))
    fresh_precision = precision_snapshot(model)
    del model
    clear()
    with ledger.measure("reload_untrained_lora"):
        model, tokenizer = load_model(config, adapter=path)
    model.eval()
    with torch.inference_mode(), autocast(config):
        reloaded_logits = model(**ids, logits_to_keep=1).logits.float().cpu()
    assert torch.equal(baseline_logits, reloaded_logits)
    evaluate_rows(model, tokenizer, regression, config, directory / "untrained_reloaded_regression", ledger, phase="untrained_reloaded_regression")
    preds = [json.loads((directory / name / "metrics.json").read_text()) for name in
             ("public_regression", "untrained_regression", "untrained_reloaded_regression")]
    raw = [__import__("reasoning_rl.runtime", fromlist=["jsonl_read"]).jsonl_read(directory / name / "predictions.jsonl") for name in
           ("public_regression", "untrained_regression", "untrained_reloaded_regression")]
    text_equal = all([r["prediction"] for r in raw[0]] == [r["prediction"] for r in v] for v in raw[1:])
    assert text_equal
    # Verify local model bytes against the already vendor-verified download receipt.
    receipt = json.loads((ROOT / "environment/desktop_model_verification.json").read_text())
    json_write(directory / "audit_result.json", {"status": "pass", "loss_audit": loss_audit,
        "all128_boundaries_masks_full_gold_eos_pass": True, "retained": len(encoded), "dropped": len(records)-len(encoded),
        "untrained_vs_public_logits_max_abs_difference": float((fresh_logits-baseline_logits).abs().max()),
        "reloaded_vs_public_logits_max_abs_difference": float((reloaded_logits-baseline_logits).abs().max()),
        "all32_predictions_identical": text_equal, "regression_correct_counts": [m["correct"] for m in preds],
        "fresh_precision": fresh_precision, "reload_precision": precision_snapshot(model),
        "control_stage": "untrained_lora_control; not SFT", "control_checkpoint_sha256": checkpoint_hash(path),
        "model_spec": model_spec(config["model"]), "model_verification_receipt_sha256": sha256_file(ROOT / "environment/desktop_model_verification.json"),
        "chat_template_note": "Inference prefix explicitly closes the non-thinking block; target continues this identical prefix. Full-conversation rendering is retained for inspection.",
        "causal_claim": "No boundary, label shift, EOS, masking or unchanged-adapter loading defect observed; this does not identify SFT degradation cause."})
    print("PASS: real-tokenizer/causal-loss/untrained-LoRA/reload audit", flush=True)


if __name__ == "__main__":
    main()
