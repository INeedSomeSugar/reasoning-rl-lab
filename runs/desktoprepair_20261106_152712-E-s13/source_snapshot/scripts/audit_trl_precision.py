"""Measure the pinned TRL preparation cast on the actual local model, without training."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import yaml

from reasoning_rl.runtime import ROOT, bootstrap, json_write, sha256_file


def snapshot(model):
    return {
        "input_embedding": str(model.get_input_embeddings().weight.dtype),
        "output_head": str(model.get_output_embeddings().weight.dtype),
        "lora_dtypes": sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}),
        "norm_dtypes": sorted({str(p.dtype) for n, p in model.named_parameters() if "norm" in n}),
    }


def main():
    bootstrap()
    from reasoning_rl.modeling import load_model
    from trl.models.utils import prepare_peft_model

    config = yaml.safe_load((ROOT / "configs/signal_pilot.yaml").read_text(encoding="utf-8"))
    model, _ = load_model(config, adapter=ROOT / "checkpoints/local17v2_20261105_smoke-B", trainable=True)
    before = snapshot(model)
    args = SimpleNamespace(bf16=True, gradient_checkpointing=True,
                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_peft_model(model, None, args)
    after = snapshot(model)
    source = Path(inspect.getsourcefile(prepare_peft_model))
    evidence = {
        "status": "measured", "training_performed": False, "holdout_accessed": False,
        "library": "trl==0.23.1", "function": "trl.models.utils.prepare_peft_model",
        "library_source_sha256": sha256_file(source), "before": before, "after": after,
        "interpretation": "The GRPO constructor casts FP32 embedding/head to BF16 for this 4-bit BF16 model. "
                          "Independent endpoint evaluation uses the shared standalone loader; in-training validation is diagnostic.",
    }
    json_write(ROOT / "reports/trl_precision_audit.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
