"""One model loader and one prompt protocol shared by all experimental arms."""
import json
from pathlib import Path
from .runtime import ROOT, json_write

PROMPT_VERSION = "math_numeric_v2"
PROMPT = "Solve the math problem. Show concise calculations. Write your final numerical answer on a separate last line beginning with ####. Put the computed number after ####.\n\nProblem: {question}"


def model_spec(key):
    return json.loads((ROOT / "configs/models.lock.json").read_text(encoding="utf-8"))[key]


def download_model(key):
    from huggingface_hub import snapshot_download
    spec = model_spec(key)
    directory = ROOT / "models" / key
    snapshot_download(spec["repo"], revision=spec["revision"], local_dir=directory,
                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model", "README.md", "LICENSE"],
                      max_workers=2)
    json_write(directory / "rrl_model_source.json", spec)
    return directory


def load_model(config, adapter=None, trainable=False, fresh_lora=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    key = config["model"]
    directory = ROOT / "models" / key
    if not (directory / "config.json").exists():
        raise FileNotFoundError(f"Download pinned model first: rrl download-model --model {key}")
    source = json.loads((directory / "rrl_model_source.json").read_text(encoding="utf-8"))
    if source != model_spec(key):
        raise ValueError("Local model revision does not match models.lock.json")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[config["dtype"]]
    quantized = config.get("quantization") == "nf4"
    kwargs = {"torch_dtype": dtype, "attn_implementation": "sdpa", "local_files_only": True,
              "trust_remote_code": False, "device_map": {"": 0} if torch.cuda.is_available() else {"": "cpu"}}
    if quantized:
        if not torch.cuda.is_available() or torch.version.hip:
            raise ValueError("This pinned NF4 path is validated for NVIDIA CUDA only. Use quantization: none on ROCm.")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(directory, **kwargs)
    model.config.use_cache = not trainable
    if quantized:
        # Preserve the same non-quantized-layer precision during training and reload/evaluation.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=trainable,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter:
        adapter = Path(adapter)
        metadata = json.loads((adapter / "rrl_checkpoint.json").read_text(encoding="utf-8"))
        for field in ("model", "dtype", "quantization"):
            if metadata["config"][field] != config[field]:
                raise ValueError(f"Checkpoint loading protocol mismatch: {field}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif fresh_lora:
        lora = config["lora"]
        model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
            lora_dropout=0.0, target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def render_prompt(tokenizer, question):
    return tokenizer.apply_chat_template([{"role": "user", "content": PROMPT.format(question=question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_sft(tokenizer, row, max_length, max_prompt_length):
    prompt_ids = tokenizer.encode(render_prompt(tokenizer, row["question"]), add_special_tokens=False)
    completion_ids = tokenizer.encode(row["solution"].strip() + tokenizer.eos_token, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_length or len(prompt_ids) + len(completion_ids) > max_length:
        return None  # Drop entire example; never train on silently truncated gold reasoning or answer.
    return {"input_ids": prompt_ids + completion_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
            "labels": [-100] * len(prompt_ids) + completion_ids}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        values = {}
        for key in ("input_ids", "attention_mask", "labels"):
            pad = self.pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
            values[key] = torch.tensor([r[key] + [pad] * (length-len(r[key])) for r in rows], dtype=torch.long)
        return values


def save_adapter(model, tokenizer, path, config, stage, **extra):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    json_write(path / "rrl_checkpoint.json", {"stage": stage, "config": config,
        "model_spec": model_spec(config["model"]), "prompt_version": PROMPT_VERSION, "thinking": False, **extra})


def autocast(config):
    import contextlib
    import torch
    if torch.cuda.is_available() and config["dtype"] != "float32":
        return torch.autocast("cuda", dtype=getattr(torch, config["dtype"]))
    return contextlib.nullcontext()
