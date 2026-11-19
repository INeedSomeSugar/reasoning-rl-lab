"""Verify public bytes; select REAL SFT or explicitly initialize from posttrained A."""
import json
import torch
import yaml
from reasoning_rl.runtime import ROOT, json_write, sha256_file, checkpoint_hash, seed_everything
from reasoning_rl.modeling import load_model, save_adapter, model_spec, autocast, render_prompt
from reasoning_rl.data import load_split
from repair_common import FAMILY, EXP, read


def main():
    decision = read(EXP / "sft_decision.json")
    receipt = read(ROOT / "environment/desktop_model_verification.json")
    model_dir = ROOT / "models/qwen3_17b"
    checks = []
    for row in receipt["files"]:
        p = model_dir / row["name"]
        actual = sha256_file(p)
        checks.append({"name": row["name"], "bytes": p.stat().st_size, "sha256": actual,
            "matches_previously_officially_verified_receipt": actual == row["sha256"] and p.stat().st_size == row["bytes"],
            "original_verification": row["verification"]})
    assert all(r["matches_previously_officially_verified_receipt"] for r in checks)
    assert receipt["model_spec"] == model_spec("qwen3_17b")
    current = EXP / "public_model_verification.json"
    json_write(current, {"status": "pass", "model_spec": receipt["model_spec"], "files": checks,
        "method": "Rehashed current local bytes against the prior official-LFS/Git verified receipt; offline reuse, no new download",
        "upstream_receipt_sha256": sha256_file(ROOT / "environment/desktop_model_verification.json")})
    config = yaml.safe_load((ROOT / f"configs/{FAMILY}_eval.yaml").read_text(encoding="utf-8"))
    if decision["route"] == "real_sft":
        path = ROOT / decision["checkpoint"]
        meta = read(path / "rrl_checkpoint.json")
        assert meta["stage"] == "sft"
        result = {"kind": "sft", "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_hash(path), "sft_candidate": decision["selected_candidate"],
            "public_model_verification_sha256": sha256_file(current)}
    else:
        path = ROOT / "checkpoints" / (FAMILY + "-public-A-initialization")
        if path.exists():
            raise FileExistsError(path)
        seed_everything(13)
        model, tokenizer = load_model(config)
        row = load_split("train", "train", limit=1, seed=config["data_seed"])[0]
        inputs = tokenizer(render_prompt(tokenizer, row["question"]), add_special_tokens=False, return_tensors="pt").to(model.device)
        model.eval()
        with torch.inference_mode(), autocast(config):
            expected = model(**inputs, logits_to_keep=1).logits.float().cpu()
        from peft import LoraConfig, get_peft_model
        lora = config["lora"]
        model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
            lora_dropout=0.0, target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
        model.eval()
        with torch.inference_mode(), autocast(config):
            actual = model(**inputs, logits_to_keep=1).logits.float().cpu()
        assert torch.equal(expected, actual)
        assert all(torch.count_nonzero(p).item() == 0 for n, p in model.named_parameters() if "lora_B" in n)
        save_adapter(model, tokenizer, path, config, "public_posttrained_initialization", explicitly_not_sft=True,
            optimizer_steps=0, training_ids=[], data_manifest_sha256=sha256_file(ROOT / "data/manifest.json"),
            source_verification_receipt=str(current.relative_to(ROOT)), source_verification_receipt_sha256=sha256_file(current),
            source_logits_max_abs_difference=0.0, initialization_seed=13,
            basis="Pinned public Qwen3-1.7B posttrained model; zero-output LoRA, no SFT")
        result = {"kind": "public_posttrained_initialization", "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_hash(path), "explicitly_not_sft": True,
            "public_model_verification_sha256": sha256_file(current)}
    json_write(EXP / "common_start.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
