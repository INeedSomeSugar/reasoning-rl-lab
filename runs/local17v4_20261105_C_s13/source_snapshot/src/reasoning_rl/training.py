"""Transformers SFT and TRL GRPO, with explicit masks and instrumented selection."""
import gc
import json
import math
import time
from pathlib import Path
from .runtime import ROOT, create_run, json_write, jsonl_write, jsonl_append, sha256_file, checkpoint_hash, code_hash, seed_everything
from .modeling import load_model, encode_sft, CompletionCollator, save_adapter, render_prompt, autocast
from .data import load_split
from .evaluation import evaluate_rows, generate_one
from .scoring import score_answer, group_advantages
from .sampling import DifficultySampler


def _snapshot_trainable(model):
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def _update_evidence(model, initial):
    import torch
    delta2 = sum(float(((p.detach().float().cpu()-initial[n])**2).sum())
                 for n, p in model.named_parameters() if n in initial)
    return {"adapter_delta_l2": delta2**0.5, "adapter_updated": delta2 > 0,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def _gradient_evidence(model):
    import torch
    squared, nonzero, finite, observed = 0.0, 0, True, 0
    frozen_with_grad = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            frozen_with_grad += int(parameter.grad is not None)
        elif parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            observed += 1
            finite = finite and bool(torch.isfinite(gradient).all().item())
            nonzero += int(torch.count_nonzero(gradient).item())
            squared += float((gradient*gradient).sum().item()) if finite else 0.0
    if not finite or frozen_with_grad:
        raise FloatingPointError("Non-finite LoRA gradients or gradients on frozen backbone")
    return {"finite_gradients": finite, "gradient_tensors": observed,
            "nonzero_gradient_elements": nonzero, "clipped_gradient_l2": squared**0.5,
            "frozen_parameter_gradients": frozen_with_grad}


def check_model(config, name):
    import torch
    directory, ledger = create_run(name, config)
    seed_everything(config["seed"])
    with ledger.measure("load_model"):
        model, tokenizer = load_model(config, trainable=True, fresh_lora=True)
    candidates = load_split("train", "train", limit=32, seed=config["data_seed"])
    encoded = [(r, encode_sft(tokenizer, r, config["sft"]["max_length"], config["generation"]["max_prompt_length"])) for r in candidates]
    encoded = [(r, e) for r, e in encoded if e is not None]
    row, item = min(encoded, key=lambda x: len(x[1]["input_ids"]))
    batch = {k: v.to(model.device) for k, v in CompletionCollator(tokenizer.pad_token_id)([item]).items()}
    before = _snapshot_trainable(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with ledger.measure("forward_backward_lora_update") as counts:
        model.train()
        with autocast(config):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite forward loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        finite_gradients = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
        nonzero_gradients = sum(int(torch.count_nonzero(g).item()) for g in gradients)
        frozen_gradients = sum(p.grad is not None for p in model.parameters() if not p.requires_grad)
        if not finite_gradients or not nonzero_gradients or frozen_gradients:
            raise RuntimeError("LoRA gradient isolation or finiteness failed")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        counts.update(processed_tokens=int(batch["attention_mask"].sum()), generated_tokens=0)
    result = {"loss": loss.item(), "finite_gradients": finite_gradients,
              "nonzero_gradient_elements": nonzero_gradients, "frozen_parameter_gradients": frozen_gradients,
              "example_id": row["id"], **_update_evidence(model, before)}
    model.eval()
    with torch.inference_mode(), autocast(config):
        expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits.float().cpu()
    path = ROOT / "checkpoints" / name
    save_adapter(model, tokenizer, path, config, "capability_smoke_only")
    result["checkpoint"] = str(path)
    result["checkpoint_sha256"] = checkpoint_hash(path)
    # Release all references before loading a second full copy on the 6GB GPU.
    del optimizer, gradients, model, batch, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with ledger.measure("checkpoint_reload"):
        model, tokenizer = load_model(config, adapter=path, trainable=False)
        batch = {k: v.to(model.device) for k, v in CompletionCollator(tokenizer.pad_token_id)([item]).items()}
        model.eval()
        with torch.inference_mode(), autocast(config):
            actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits.float().cpu()
        result["reload_logits_max_absolute_difference"] = (expected-actual).abs().max().item()
        result["reload_logits_match"] = torch.allclose(expected, actual, atol=0.02, rtol=0.002)
    if not result["adapter_updated"] or not result["reload_logits_match"]:
        json_write(directory / "capability.json", result)
        raise RuntimeError("Adapter update or checkpoint round trip failed")
    json_write(directory / "capability.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_sft(config, name):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, TrainerCallback
    directory, ledger = create_run(name, config)
    seed_everything(config["seed"])
    model, tokenizer = load_model(config, trainable=True, fresh_lora=True)
    source = load_split("train", "train", limit=config["sft"]["pool_size"], seed=config["data_seed"])
    prepared, retained, dropped = [], [], []
    for row in source:
        encoded = encode_sft(tokenizer, row, config["sft"]["max_length"], config["generation"]["max_prompt_length"])
        if encoded:
            prepared.append(encoded)
            retained.append(row["id"])
        else:
            dropped.append(row["id"])
    if not prepared:
        raise ValueError("All SFT examples exceeded length budget")
    json_write(directory / "data_usage.json", {"retained": retained, "dropped_length": dropped,
                                               "manifest_sha256": sha256_file(ROOT / "data/manifest.json")})
    initial = _snapshot_trainable(model)
    counts = {"processed_tokens": 0, "supervised_tokens": 0}

    class MaskedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            counts["processed_tokens"] += int(inputs["attention_mask"].sum().item())
            counts["supervised_tokens"] += int((inputs["labels"] != -100).sum().item())
            # CausalLM shifts labels internally. Prompt/pad positions already carry -100.
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    class LogCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            ledger.emit(phase="gradient_diagnostic", optimizer_step=state.global_step+1, **_gradient_evidence(model))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                ledger.emit(phase="sft_log", optimizer_step=state.global_step, **logs)

    opts = config["sft"]
    args = TrainingArguments(output_dir=str(directory / "trainer"), max_steps=opts["steps"],
        per_device_train_batch_size=1, gradient_accumulation_steps=opts["gradient_accumulation_steps"],
        learning_rate=opts["learning_rate"], lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch", bf16=config["dtype"] == "bfloat16",
        fp16=config["dtype"] == "float16", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], logging_steps=1,
        save_strategy="no", eval_strategy="no", seed=config["seed"], data_seed=config["data_seed"],
        dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
        disable_tqdm=True, use_cpu=not torch.cuda.is_available())
    trainer = MaskedTrainer(model=model, args=args, train_dataset=Dataset.from_list(prepared),
        data_collator=CompletionCollator(tokenizer.pad_token_id), callbacks=[LogCallback()])
    with ledger.measure("sft_training") as cost:
        output = trainer.train()
        cost.update(**counts, generated_tokens=0)
    path = ROOT / "checkpoints" / name
    save_adapter(model, tokenizer, path, config, "sft", training_ids=retained,
                 data_manifest_sha256=sha256_file(ROOT / "data/manifest.json"))
    trainer.save_state()
    result = {**output.metrics, **_update_evidence(model, initial), **counts,
              "retained_examples": len(retained), "dropped_length": len(dropped),
              "checkpoint": str(path), "checkpoint_sha256": checkpoint_hash(path)}
    json_write(directory / "training_result.json", result)
    if not result["adapter_updated"]:
        raise RuntimeError("SFT did not update the adapter")
    return result


def run_grpo(config, name, sft_checkpoint, strategy):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    if strategy not in ("uniform", "dynamic", "static"):
        raise ValueError(strategy)
    source_meta = json.loads((Path(sft_checkpoint) / "rrl_checkpoint.json").read_text(encoding="utf-8"))
    if source_meta["stage"] != "sft":
        raise ValueError("C/D/static must start from a real SFT checkpoint")
    if source_meta["data_manifest_sha256"] != sha256_file(ROOT / "data/manifest.json"):
        raise ValueError("SFT and GRPO data manifests differ")
    directory, ledger = create_run(name, {**config, "strategy": strategy})
    seed_everything(config["seed"])
    opts = config["grpo"]
    if opts["beta"] != 0:
        raise ValueError("v0.1 fixes beta=0; nonzero KL needs an explicit frozen SFT reference adapter")
    model, tokenizer = load_model(config, adapter=sft_checkpoint, trainable=True)
    pool = load_split("train", "select", limit=opts["pool_size"], seed=config["data_seed"])
    # Reject overlong prompts before any arm starts; use precisely the same IDs in all arms.
    pool = [r for r in pool if len(tokenizer.encode(render_prompt(tokenizer, r["question"]), add_special_tokens=False))
            <= config["generation"]["max_prompt_length"]]
    if not pool:
        raise ValueError("Empty GRPO pool")
    by_id = {r["id"]: {"id": r["id"], "answer": r["answer"], "prompt": render_prompt(tokenizer, r["question"])} for r in pool}
    selector = DifficultySampler([r["id"] for r in pool], strategy, config["seed"],
                                 opts["uniform_mix"], opts["prior"])
    initial = _snapshot_trainable(model)
    initial_hash = checkpoint_hash(sft_checkpoint)
    json_write(directory / "comparison_contract.json", {
        "sft_checkpoint_sha256": initial_hash, "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
        "pool_ids": list(by_id), "protocol": config, "strategy": strategy,
        "code_sha256": code_hash(),
        "probe_cost_included": True, "distributed_supported": False})
    statistics = {"generation_batches": 0, "processed_tokens": 0, "padded_processed_tokens": 0,
                  "generation_prompt_tokens": 0, "logprob_forward_input_tokens": 0,
                  "logprob_forward_padded_tokens": 0, "generated_tokens": 0,
                  "probe_generated_tokens": 0, "probe_prompt_tokens": 0, "probe_calls": 0,
                  "zero_variance_groups": 0, "groups": 0, "reward_sum": 0.0, "reward_count": 0,
                  "nonzero_advantage_completions": 0, "last_refresh_step": None,
                  "sampler_weight_change_events": 0, "format_errors": 0, "parse_errors": 0,
                  "truncated_completions": 0}
    budget_started = time.perf_counter()
    training_started = None

    class BudgetReached(RuntimeError):
        pass

    def accounted_tokens():
        return sum(statistics[k] for k in ("generated_tokens", "generation_prompt_tokens", "probe_generated_tokens", "processed_tokens", "probe_prompt_tokens"))

    def budget_exceeded():
        return time.perf_counter()-training_started >= opts["max_wall_seconds"] or accounted_tokens() >= opts["max_accounted_tokens"]

    class InstrumentedGRPO(GRPOTrainer):
        """Single-device prompt selection at the actual generation boundary avoids dataloader lookahead."""
        def _generate_and_score_completions(self, inputs):
            step = self.state.global_step
            refresh = strategy != "uniform" and (selector.updates == 0 or
                (strategy == "dynamic" and step > 0 and step % opts["refresh_interval"] == 0
                 and statistics["last_refresh_step"] != step))
            if refresh:
                observations, predictions = {}, []
                probe_complete = True
                # Isolate probe RNG from training generation RNG. Probes remain real paid compute in the ledger.
                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), ledger.measure("difficulty_probe", optimizer_step=step) as cost:
                    torch.manual_seed(config["seed"]+100000+selector.updates)
                    for row in pool:
                        if budget_exceeded():
                            probe_complete = False
                            break
                        result, token_counts = generate_one(self.accelerator.unwrap_model(self.model), tokenizer,
                            row, config, sample=True, count=opts["probe_samples"])
                        predictions.extend(result)
                        observations[row["id"]] = (sum(r["correct"] for r in result), len(result))
                        statistics["probe_generated_tokens"] += token_counts["generated_tokens"]
                        statistics["probe_prompt_tokens"] += token_counts["prompt_tokens"]
                    cost.update(generated_tokens=sum(r["generated_tokens"] for r in predictions),
                                prompt_tokens=sum(r["prompt_tokens"] for r in predictions),
                                probe_items=len(observations), probe_complete=probe_complete)
                if not probe_complete:
                    jsonl_write(directory / f"probes/step_{step:05d}_incomplete.jsonl", predictions)
                    raise BudgetReached("Budget reached inside difficulty probe; partial probe not used for selection")
                old_weights = selector.weights() if selector.updates else None
                selector.refresh(observations)
                if old_weights is not None:
                    statistics["sampler_weight_change_events"] += int(any(abs(a-b)>1e-12 for a,b in zip(old_weights, selector.weights(), strict=True)))
                statistics["probe_calls"] += 1
                statistics["last_refresh_step"] = step
                jsonl_write(directory / f"probes/step_{step:05d}.jsonl", predictions)
                json_write(directory / f"probes/step_{step:05d}_state.json", selector.state_dict())
            if budget_exceeded():
                raise BudgetReached("Budget reached before next rollout group")
            if len(inputs) % self.num_generations:
                raise ValueError("Generation batch must contain complete GRPO groups")
            chosen = selector.draw(len(inputs)//self.num_generations)
            actual = [dict(by_id[i]) for i in chosen for _ in range(self.num_generations)]
            jsonl_append(directory / "selections.jsonl", [{"optimizer_step": step, "generation_batch": statistics["generation_batches"],
                               "ids": chosen, "sampler_updates": selector.updates,
                               "weights": selector.state_dict()["weights"]}])
            statistics["generation_batches"] += 1
            # TRL unwraps the model for generation; re-enter AMP explicitly for FP32 frozen
            # embedding/head layers alongside BF16 k-bit compute on the native Windows path.
            with autocast(config):
                generated = super()._generate_and_score_completions(actual)
            # TRL 0.23.1's unwrap helper re-enables checkpointing with default kwargs.
            # Restore the explicit non-reentrant policy for the subsequent backward pass.
            if self.args.gradient_checkpointing:
                self.accelerator.unwrap_model(self.model).gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            statistics["generation_prompt_tokens"] += int(generated["prompt_mask"].sum().item())
            return generated

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, *args, **kwargs):
            statistics["logprob_forward_input_tokens"] += int(attention_mask.sum().item())
            statistics["logprob_forward_padded_tokens"] += int(input_ids.numel())
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, *args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Completion loss mask hides truncated rows; count generated positions independently of that mask.
            statistics["processed_tokens"] += int(inputs["prompt_mask"].sum().item())
            ids = inputs["completion_ids"]
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
            first_eos = torch.where(ids == tokenizer.eos_token_id, positions, ids.shape[1]).amin(dim=1)
            lengths = (first_eos+1).clamp(max=ids.shape[1])
            statistics["processed_tokens"] += int(lengths.sum().item())
            statistics["padded_processed_tokens"] += int(inputs["prompt_ids"].numel()+ids.numel())
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def reward(completions, answer, id, completion_ids=None, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        scores = [score_answer(t, a) for t, a in zip(texts, answer, strict=True)]
        rewards = [float(s.correct) for s in scores]
        statistics["reward_sum"] += sum(rewards)
        statistics["reward_count"] += len(rewards)
        statistics["format_errors"] += sum(not s.format_ok for s in scores)
        statistics["parse_errors"] += sum(not s.parse_ok for s in scores)
        if completion_ids is not None:
            statistics["generated_tokens"] += sum(len(ids) for ids in completion_ids)
        group_size = opts["num_generations"]
        for begin in range(0, len(rewards), group_size):
            group = rewards[begin:begin+group_size]
            if len(set(id[begin:begin+group_size])) != 1:
                raise ValueError("GRPO group mixes different questions")
            statistics["groups"] += 1
            statistics["zero_variance_groups"] += int(len(set(group)) == 1)
            statistics["nonzero_advantage_completions"] += sum(a != 0 for a in group_advantages(group, opts["scale_rewards"]))
        reward_log = []
        for i, (text, score, r) in enumerate(zip(texts, scores, rewards, strict=True)):
            ids = completion_ids[i] if completion_ids is not None else []
            truncated = tokenizer.eos_token_id not in ids if ids else None
            statistics["truncated_completions"] += int(truncated is True)
            reward_log.append({"id": id[i], "generation_batch": statistics["generation_batches"]-1,
                               "group_within_batch": i//group_size,
                               "prediction": text, "gold_answer": answer[i], "reward": r,
                               "generated_tokens": len(ids), "truncated": truncated, **score.as_dict()})
        jsonl_append(directory / "rollouts.jsonl", reward_log)
        return rewards

    class EvidenceCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            ledger.emit(phase="gradient_diagnostic", optimizer_step=state.global_step+1, **_gradient_evidence(model))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                ledger.emit(phase="grpo_log", optimizer_step=state.global_step, **logs)

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.perf_counter()-training_started
            total_tokens = accounted_tokens()
            ledger.emit(phase="budget_progress", optimizer_step=state.global_step,
                        cumulative_training_wall_seconds=elapsed, generated_tokens=statistics["generated_tokens"],
                        probe_generated_tokens=statistics["probe_generated_tokens"],
                        policy_loss_input_tokens=statistics["processed_tokens"],
                        probe_prompt_tokens=statistics["probe_prompt_tokens"],
                        zero_variance_group_fraction=statistics["zero_variance_groups"]/max(statistics["groups"], 1))
            if elapsed >= opts["max_wall_seconds"] or total_tokens >= opts["max_accounted_tokens"]:
                control.should_training_stop = True
            if state.global_step % opts["validation_interval"] == 0 or control.should_training_stop:
                evaluate_rows(trainer.accelerator.unwrap_model(trainer.model), tokenizer,
                    load_split("validation", "evaluate_validation", limit=config["evaluation"]["validation_size"], seed=config["data_seed"]),
                    config, directory / f"validation/step_{state.global_step:05d}", ledger,
                    phase="validation", step=state.global_step)
            # Validation time counts against the run's wall budget, too.
            if time.perf_counter()-training_started >= opts["max_wall_seconds"]:
                control.should_training_stop = True

    generation = config["generation"]
    args = GRPOConfig(output_dir=str(directory / "trainer"), max_steps=opts["steps"],
        per_device_train_batch_size=1, gradient_accumulation_steps=opts["gradient_accumulation_steps"],
        num_generations=opts["num_generations"], steps_per_generation=opts["gradient_accumulation_steps"],
        learning_rate=opts["learning_rate"], lr_scheduler_type="constant", warmup_steps=0,
        optim="adamw_torch", max_grad_norm=1.0, weight_decay=0.0,
        bf16=config["dtype"] == "bfloat16", fp16=config["dtype"] == "float16",
        max_prompt_length=generation["max_prompt_length"], max_completion_length=generation["max_new_tokens"],
        temperature=generation["temperature"], top_p=generation["top_p"], top_k=generation["top_k"],
        generation_kwargs={"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                           "pad_token_id": tokenizer.pad_token_id, "use_cache": True},
        repetition_penalty=1.0, beta=opts["beta"], epsilon=opts["epsilon"],
        num_iterations=1, importance_sampling_level="token", scale_rewards=opts["scale_rewards"],
        loss_type=opts["loss_type"], mask_truncated_completions=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False, use_liger_loss=False, report_to=[], logging_steps=1,
        save_strategy="no", eval_strategy="no", seed=config["seed"], data_seed=config["data_seed"],
        dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
        disable_tqdm=True, use_cpu=not torch.cuda.is_available())
    trainer = InstrumentedGRPO(model=model, args=args, reward_funcs=reward,
        train_dataset=Dataset.from_list(list(by_id.values())), processing_class=tokenizer, callbacks=[EvidenceCallback()])
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Prompt selector is single-device only; multi-GPU is deliberately unsupported")
    json_write(directory / "resolved_training_args.json", args.to_dict())
    json_write(directory / "resolved_generation_config.json", trainer.generation_config.to_dict())
    training_started = time.perf_counter()
    with ledger.measure("grpo_total") as cost:
        try:
            output = trainer.train()
            output_metrics = output.metrics
        except BudgetReached as error:
            output_metrics = {"budget_stop_reason": str(error)}
        final_step = trainer.state.global_step
        if not (directory / f"validation/step_{final_step:05d}/metrics.json").exists():
            ledger.emit(phase="budget_progress", optimizer_step=final_step,
                        cumulative_training_wall_seconds=time.perf_counter()-training_started,
                        generated_tokens=statistics["generated_tokens"],
                        probe_generated_tokens=statistics["probe_generated_tokens"],
                        policy_loss_input_tokens=statistics["processed_tokens"],
                        probe_prompt_tokens=statistics["probe_prompt_tokens"], final_endpoint=True)
            evaluate_rows(trainer.accelerator.unwrap_model(trainer.model), tokenizer,
                load_split("validation", "evaluate_validation", limit=config["evaluation"]["validation_size"], seed=config["data_seed"]),
                config, directory / f"validation/step_{final_step:05d}", ledger, phase="validation", step=final_step)
        cost.update(policy_loss_input_tokens=statistics["processed_tokens"],
                    policy_loss_padded_input_tokens=statistics["padded_processed_tokens"],
                    generation_prompt_tokens=statistics["generation_prompt_tokens"],
                    logprob_forward_input_tokens=statistics["logprob_forward_input_tokens"],
                    logprob_forward_padded_tokens=statistics["logprob_forward_padded_tokens"],
                    generated_tokens=statistics["generated_tokens"],
                    probe_prompt_tokens=statistics["probe_prompt_tokens"],
                    probe_generated_tokens=statistics["probe_generated_tokens"])
    path = ROOT / "checkpoints" / name
    save_adapter(model, tokenizer, path, config, "grpo", strategy=strategy,
                 sft_checkpoint_sha256=initial_hash, data_manifest_sha256=sha256_file(ROOT / "data/manifest.json"))
    trainer.save_state()
    result = {**output_metrics, **statistics, **_update_evidence(model, initial),
              "mean_reward": statistics["reward_sum"]/max(1, statistics["reward_count"]),
              "zero_variance_group_fraction": statistics["zero_variance_groups"]/max(1, statistics["groups"]),
              "training_format_error_rate": statistics["format_errors"]/max(1, statistics["reward_count"]),
              "training_parse_error_rate": statistics["parse_errors"]/max(1, statistics["reward_count"]),
              "training_truncation_rate": statistics["truncated_completions"]/max(1, statistics["reward_count"]),
              "checkpoint": str(path), "checkpoint_sha256": checkpoint_hash(path),
              "sft_checkpoint_sha256": initial_hash, "completed_optimizer_steps": trainer.state.global_step,
              "wall_budget_overshoot_seconds": max(0, time.perf_counter()-training_started-opts["max_wall_seconds"])}
    if strategy == "uniform" or selector.updates:
        json_write(directory / "sampler_final.json", selector.state_dict())
    else:
        json_write(directory / "sampler_final.json", {"strategy": strategy, "updates": 0, "incomplete_initial_probe": True})
    json_write(directory / "training_result.json", result)
    return result
