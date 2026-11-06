"""Independent file/reward/weight/sampler replay audit. Never deserialize holdouts."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from safetensors.torch import load_file
from reasoning_rl.runtime import ROOT,json_write,jsonl_read,sha256_file,checkpoint_hash
from reasoning_rl.data import load_split
from reasoning_rl.scoring import score_answer,group_advantages
from reasoning_rl.sampling import DifficultySampler,RolloutCacheSampler

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"reports/comparisons"/FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot_hash(root):
    digest=hashlib.sha256()
    for folder in ("src","configs"):
        for path in sorted((root/folder).rglob("*")):
            if path.is_file() and path.suffix in {".py",".yaml",".json"}:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def main(partial=False):
    OUT.mkdir(parents=True,exist_ok=True)
    plan=read(EXP/"sampling_frozen.json")
    train_ids={r["id"] for r in load_split("train","develop")}
    sets=read(EXP/"development_ids.json")
    assert not set(sets["comparison"])&set(sets["confirmation"])
    assert not train_ids & (set(sets["comparison"])|set(sets["confirmation"]))
    source=ROOT/plan["common_start"]["checkpoint"]
    initial=load_file(source/"adapter_model.safetensors",device="cpu")
    control=ROOT/"checkpoints"/(FAMILY+"-untrained-control")
    contract_audit=read(ROOT/"runs"/(FAMILY+"-sft-contract")/"audit_result.json")
    assert checkpoint_hash(control)==contract_audit["control_checkpoint_sha256"]
    assert read(control/"rrl_checkpoint.json")["stage"]=="untrained_lora_control"
    control_weights=load_file(control/"adapter_model.safetensors",device="cpu")
    assert all(torch.count_nonzero(p).item()==0 for key,p in control_weights.items() if "lora_B" in key)
    records=[]
    for directory in sorted((ROOT/"runs").glob(FAMILY+"*")):
        if not directory.is_dir() or not (directory/"provenance.json").exists():
            continue
        if not ((directory/"training_result.json").exists() or (directory/"audit_result.json").exists()
                or any(directory.glob("**/metrics.json"))):
            continue
        record={"run":directory.name,"source_snapshot_hash_matches":snapshot_hash(directory/"source_snapshot")==read(directory/"provenance.json")["code_sha256"]}
        environment=read(directory/"environment.json")
        record["single_RX9070XT"]=(len(environment["gpus"])==1 and "9070 XT" in environment["gpus"][0]["name"]
                                     and environment["diagnostic_flags"]["HIP_VISIBLE_DEVICES"]=="0")
        for path in directory.glob("**/predictions.jsonl"):
            rows=jsonl_read(path)
            if not rows:
                continue
            assert all(r["split"]=="validation" and r["id"] not in train_ids for r in rows)
            assert len({r["id"] for r in rows})==len(rows)
            assert all(score_answer(r["prediction"],r["gold_answer"]).correct==r["correct"] for r in rows)
            metrics_path=path.with_name("metrics.json")
            if metrics_path.exists():
                metrics=read(metrics_path)
                assert len(rows)==metrics["n"] and sum(r["correct"] for r in rows)==metrics["correct"]
                assert math.isclose(sum(r["truncated"] for r in rows)/len(rows),metrics["truncation_rate"])
                assert math.isclose(sum(not r["parse_ok"] for r in rows)/len(rows),metrics["parse_error_rate"])
        result_path=directory/"training_result.json"
        if result_path.exists():
            result=read(result_path)
            checkpoint=ROOT/"checkpoints"/directory.name
            record["checkpoint_hash_matches"]=checkpoint_hash(checkpoint)==result["checkpoint_sha256"]
            if (directory/"supervision_batches.jsonl").exists():
                batches=jsonl_read(directory/"supervision_batches.jsonl")
                assert sum(r["supervised_tokens"] for r in batches)==result["supervised_tokens"]
                assert sum(r["processed_tokens"] for r in batches)==result["processed_tokens"]
                assert all(r["supervised_tokens"]==r["causally_shifted_supervised_tokens"] for r in batches)
                record["supervision_reconciles"]=True
            if (directory/"comparison_contract.json").exists():
                contract=read(directory/"comparison_contract.json")
                ids=contract["pool_ids"]
                assert set(ids)<=train_ids
                assert contract["initial_checkpoint_sha256"]==plan["common_start"]["checkpoint_sha256"]
                endpoint=load_file(checkpoint/"adapter_model.safetensors",device="cpu")
                delta=math.sqrt(sum(float(((endpoint[k].float()-initial[k].float())**2).sum()) for k in initial))
                assert math.isclose(delta,result["adapter_delta_l2"],rel_tol=1e-7,abs_tol=1e-9)
                record["independent_saved_weight_delta_l2"]=delta
                rollouts=jsonl_read(directory/"rollouts.jsonl")
                assert all(r["id"] in ids and float(score_answer(r["prediction"],r["gold_answer"]).correct)==r["reward"] for r in rollouts)
                assert sum(r["generated_tokens"] for r in rollouts)==result["generated_tokens"]
                assert sum(r["reward"] for r in rollouts)==result["reward_sum"]
                groups=defaultdict(list)
                for r in rollouts:
                    groups[(r["generation_batch"],r["group_within_batch"])].append(r)
                assert all(len({r["id"] for r in group})==1 for group in groups.values())
                assert len(groups)==result["groups"]
                assert sum(len({r["reward"] for r in group})==1 for group in groups.values())==result["zero_variance_groups"]
                strategy=contract["strategy"]
                opts=contract["protocol"]["grpo"]
                cls=DifficultySampler if strategy=="uniform" else RolloutCacheSampler
                kwargs={} if strategy=="uniform" else {"max_trials":opts["cache_max_trials"],"ttl_steps":opts["cache_ttl_steps"]}
                sampler=cls(ids,strategy,contract["protocol"]["seed"],opts["uniform_mix"],opts["prior"],**kwargs)
                if strategy!="uniform":
                    probe=jsonl_read(directory/"probes/step_00000.jsonl")
                    assert {r["id"] for r in probe}==set(ids)
                    sampler.initialize_ordered({i:[float(r["correct"]) for r in probe if r["id"]==i] for i in ids})
                events=jsonl_read(directory/"selections.jsonl")
                for event in events:
                    step=event["optimizer_step"]
                    if strategy!="uniform":
                        sampler.advance(step)
                    actual=dict(zip(ids,sampler.weights(),strict=True))
                    assert actual==event["weights"]
                    assert sampler.draw(len(event["ids"]))==event["ids"]
                    if strategy=="dynamic":
                        for key in sorted(k for k in groups if k[0]==event["generation_batch"]):
                            group=groups[key]
                            sampler.observe_rollouts(group[0]["id"],[r["reward"] for r in group],step)
                final=read(directory/"sampler_final.json")
                if strategy!="uniform":
                    sampler.advance(final["optimizer_step"])
                expected=sampler.state_dict()
                assert expected==final
                record["exact_sampler_replay_matches"]=True
                record["reward_token_group_counts_reconcile"]=True
                gradients=[r for r in jsonl_read(directory/"costs.jsonl") if r["phase"]=="gradient_diagnostic"]
                assert len(gradients)==result["completed_optimizer_steps"]
                assert all(r["finite_gradients"] and not r["frozen_parameter_gradients"] for r in gradients)
                record["finite_gradients"]=True
                record["nonzero_gradient_steps"]=sum(r["nonzero_gradient_elements"]>0 for r in gradients)
        assert all(v for k,v in record.items() if isinstance(v,bool)),record
        records.append(record)
    all_nine=[]
    for seed in (13,29,47):
        pair=[]
        for arm in ("C","D","E"):
            directory=ROOT/"runs"/f"{FAMILY}-{arm}-s{seed}"
            if partial and not (ROOT/"runs"/(directory.name+"-comparison")/"validation/metrics.json").exists():
                continue
            assert read(directory/"provenance.json")["code_sha256"]==plan["code_sha256"]
            assert read(directory/"comparison_contract.json")["pool_ids"]==plan["pool_ids"]
            predictions=jsonl_read(ROOT/"runs"/(directory.name+"-comparison")/"validation/predictions.jsonl")
            assert [r["id"] for r in predictions]==sets["comparison"]
            all_nine.append(directory.name)
        dirs=[ROOT/"runs"/f"{FAMILY}-{arm}-s{seed}" for arm in ("D","E")]
        if partial and not all((p/"probes/step_00000_state.json").exists() for p in dirs):
            continue
        ds,es=[read(p/"probes/step_00000_state.json") for p in dirs]
        dp,ep=[jsonl_read(p/"probes/step_00000.jsonl") for p in dirs]
        assert ds["weights"]==es["weights"]
        assert [(r["id"],r["prediction"]) for r in dp]==[(r["id"],r["prediction"]) for r in ep]
    confirmation_status=ROOT/"environment"/(FAMILY+"-confirmation-status.json")
    confirmed=confirmation_status.exists() and read(confirmation_status)["status"]=="completed_one_stage_confirmation"
    if confirmed:
        claim=read(EXP/"confirmation_opened.json")
        for endpoint in claim["endpoints"]:
            rows=jsonl_read(ROOT/"runs"/f"{FAMILY}-{endpoint['label']}-confirmation"/"validation/predictions.jsonl")
            assert [r["id"] for r in rows]==sets["confirmation"]
            if endpoint["checkpoint"]:
                assert checkpoint_hash(ROOT/endpoint["checkpoint"])==endpoint["checkpoint_sha256"]
    assert not any((ROOT/"experiments/final_claims").glob("*.json"))
    json_write(OUT/("evidence_audit_partial.json" if partial else "evidence_audit.json"),{"status":"partial_pass" if partial else "pass","runs_checked":len(records),"runs":records,
        "complete_nine_endpoints":all_nine,"one_stage_confirmation_complete":confirmed,
        "test_SVAMP_content_deserialized":False,"formal_final_claims_present":False,
        "checks":"Source/config snapshots, one9070XT, raw scores, full supervision, actual saved parameter deltas, exact seed/weight/cache sampling replay and paired initial probes"})
    print(json.dumps({"status":"partial_pass" if partial else "pass","runs_checked":len(records),"confirmation_complete":confirmed},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true")
    main(parser.parse_args().partial)
