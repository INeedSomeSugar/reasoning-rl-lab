"""Verify fixed S2, actual GPU backward, Adam zero-gradient drift and reload."""
import gc
import json
import time
import yaml
from reasoning_rl.runtime import (ROOT, bootstrap, hardware_info, json_write, checkpoint_hash,
    sha256_file, create_run, seed_everything)
from reasoning_rl.modeling import load_model, encode_sft, CompletionCollator, autocast, save_adapter, precision_snapshot
from reasoning_rl.data import load_split
from reasoning_rl.training import _snapshot_trainable, _update_evidence, _gradient_evidence
from filter_common import EXP, FAMILY, OLD, START, START_HASH


def main():
    bootstrap()
    import torch
    config = yaml.safe_load((ROOT / "configs/desktoprepair_20261106_152712_grpo_s13.yaml").read_text())
    directory, ledger = create_run(FAMILY + "-preflight", config)
    seed_everything(13)
    hw = hardware_info()
    assert len(hw["gpus"]) == 1 and "9070 XT" in hw["gpus"][0]["name"]
    files = []
    for r in json.loads((OLD / "public_model_verification.json").read_text())["files"]:
        actual = sha256_file(ROOT / "models/qwen3_17b" / r["name"])
        assert actual == r["sha256"], r["name"]
        files.append({"name": r["name"], "sha256": actual})
    assert checkpoint_hash(ROOT / START) == START_HASH
    with ledger.measure("load_fixed_s2"):
        model, tok = load_model(config, adapter=ROOT / START, trainable=True)
    precision = precision_snapshot(model)
    rs = load_split("train", "train", limit=32, seed=config["data_seed"])
    items = [(r, encode_sft(tok, r, config["sft"]["max_length"], config["generation"]["max_prompt_length"])) for r in rs]
    row, item = min(((r, e) for r, e in items if e is not None), key=lambda x: len(x[1]["input_ids"]))
    batch = {k: v.to(model.device) for k, v in CompletionCollator(tok.pad_token_id)([item]).items()}
    before = _snapshot_trainable(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["grpo"]["learning_rate"], weight_decay=0.0)
    with ledger.measure("fixed_s2_backward_and_adam_diagnostic"):
        model.train()
        with autocast(config):
            loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        grad = _gradient_evidence(model)
        assert grad["nonzero_gradient_elements"] > 0
        optimizer.step()
        after_signal = _snapshot_trainable(model)
        optimizer.zero_grad(set_to_none=False)  # actual zero tensor, retaining Adam state
        zero_grad = _gradient_evidence(model)
        assert zero_grad["nonzero_gradient_elements"] == 0
        optimizer.step()
        zero_update = _update_evidence(model, after_signal)
        assert zero_update["adapter_updated"]
        total_update = _update_evidence(model, before)
    model.eval()
    with torch.inference_mode(), autocast(config):
        expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits.float().cpu()
    path = ROOT / "checkpoints" / (FAMILY + "-preflight")
    with ledger.measure("diagnostic_save"):
        save_adapter(model, tok, path, config, "s2_diagnostic_only_not_experiment_start")
    loss_value = float(loss)
    del model, optimizer, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    with ledger.measure("independent_reload"):
        model, tok = load_model(config, adapter=path)
        batch = {k: v.to(model.device) for k, v in CompletionCollator(tok.pad_token_id)([item]).items()}
        model.eval()
        with torch.inference_mode(), autocast(config):
            actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits.float().cpu()
    difference = float((expected - actual).abs().max())
    assert torch.equal(expected, actual)
    assert checkpoint_hash(ROOT / START) == START_HASH
    result = {"hardware": hw, "precision": precision, "fixed_s2_sha256": START_HASH,
        "verified_model_files": files, "loss": loss_value, "example_id": row["id"],
        "nonzero_backward": grad, "zero_gradient_diagnostic": zero_grad,
        "actual_weight_delta_after_zero_gradient_adam_step": zero_update,
        "total_diagnostic_weight_update": total_update,
        "saved_checkpoint_sha256": checkpoint_hash(path), "reload_logits_max_absolute_difference": difference,
        "fixed_s2_unchanged": True, "diagnostic_only": True, "no_new_sft_start": True}
    json_write(directory / "preflight.json", result)
    json_write(EXP / "preflight.json", result)
    print(json.dumps({"passed": True, "reload_difference": difference,
        "zero_gradient_adam_weight_delta_l2": zero_update["adapter_delta_l2"]}))


if __name__ == "__main__":
    main()
