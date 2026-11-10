"""Run all three paired seeds, then a single prespecified stage confirmation."""
import argparse
import time
from reasoning_rl.runtime import ROOT,json_write,jsonl_read,sha256_file,checkpoint_hash,code_hash
from repair_common import FAMILY,EXP,read,run_phase,remaining


def verify(plan):
    assert code_hash()==plan["code_sha256"], "Source/config changed after freeze"
    assert sha256_file(EXP/"development_ids.json")==plan["development_ids_sha256"]
    assert checkpoint_hash(ROOT/plan["common_start"]["checkpoint"])==plan["common_start"]["checkpoint_sha256"]
    for name,digest in read(EXP/"frozen_source_manifest.json").items():
        assert sha256_file(ROOT/name)==digest, f"Frozen executable changed: {name}"


def comparison(plan):
    status_path=ROOT/"environment"/(FAMILY+"-controlled-status.json")
    if status_path.exists():
        raise FileExistsError(status_path)
    state={"status":"running","started_unix":time.time(),"completed":[],"holdout_opened":False}
    json_write(status_path,state)
    try:
        source=plan["common_start"]
        source_flag="--sft-checkpoint" if source["kind"]=="sft" else "--public-initialization"
        for seed,arm in plan["order"]:
            verify(plan)
            name=f"{FAMILY}-{arm}-s{seed}"
            strategy={"C":"uniform","D":"dynamic","E":"static"}[arm]
            config=plan["configs"][str(seed)]["path"]
            run_phase(f"{arm}-s{seed}",["-m","reasoning_rl","grpo","--name",name,"--hardware","rocm",
                "--config",config,"--strategy",strategy,source_flag,source["checkpoint"]],state,status_path)
            contract=read(ROOT/"runs"/name/"comparison_contract.json")
            assert contract["pool_ids"]==plan["pool_ids"]
            assert contract["initial_checkpoint_sha256"]==source["checkpoint_sha256"]
            run_phase(f"{arm}-s{seed}-comparison",["-m","reasoning_rl","evaluate","--name",name+"-comparison",
                "--hardware","rocm","--config",config,"--adapter","checkpoints/"+name],state,status_path,
                timeout=plan["cost_based_capacity"]["evaluation_reserve_seconds"]+30)
        pairs=[]
        for seed in plan["seeds"]:
            paths=[ROOT/"runs"/f"{FAMILY}-{arm}-s{seed}" for arm in ("D","E")]
            ds,es=[read(p/"probes/step_00000_state.json") for p in paths]
            dp,ep=[jsonl_read(p/"probes/step_00000.jsonl") for p in paths]
            assert ds["weights"]==es["weights"]
            assert [(r["id"],r["prediction"]) for r in dp]==[(r["id"],r["prediction"]) for r in ep]
            pairs.append({"seed":seed,"initial_predictions_equal":True,"initial_weights_equal":True})
        json_write(EXP/"paired_initialization_checks.json",pairs)
        state.update(status="completed_balanced_development_comparison",finished_unix=time.time())
        json_write(status_path,state)
    except BaseException as error:
        state.update(status="failed_or_stopped",error=repr(error),finished_unix=time.time())
        json_write(status_path,state)
        raise


def confirmation(plan):
    status=read(ROOT/"environment"/(FAMILY+"-controlled-status.json"))
    assert status["status"]=="completed_balanced_development_comparison"
    verify(plan)
    claim=EXP/"confirmation_opened.json"
    if claim.exists():
        raise FileExistsError("One stage confirmation already opened; preserve it")
    endpoints=[("A",None)]
    decision=read(EXP/"sft_decision.json")
    if decision["route"]=="real_sft":
        endpoints.append((decision["selected_candidate"],decision["checkpoint"]))
    endpoints += [(f"{arm}-s{seed}",f"checkpoints/{FAMILY}-{arm}-s{seed}") for seed in plan["seeds"] for arm in ("C","D","E")]
    reserve=plan["cost_based_capacity"]["evaluation_reserve_seconds"]+30
    if remaining() < len(endpoints)*reserve+900:
        json_write(EXP/"confirmation_not_opened.json",{"reason":"Insufficient conservative time reserve for every fixed endpoint",
            "remaining_seconds":remaining(),"needed_seconds":len(endpoints)*reserve+900})
        return
    # All checkpoint identities are fixed BEFORE the first confirmation output.
    json_write(claim,{"opened_unix":time.time(),"independent_test":False,
        "scope":"disjoint validation128, one stage opening, no tuning afterwards",
        "development_ids_sha256":plan["development_ids_sha256"],"code_sha256":plan["code_sha256"],
        "endpoints":[{"label":label,"checkpoint":adapter,"checkpoint_sha256":checkpoint_hash(ROOT/adapter) if adapter else None}
                     for label,adapter in endpoints]})
    status_path=ROOT/"environment"/(FAMILY+"-confirmation-status.json")
    state={"status":"running","started_unix":time.time(),"completed":[],"holdout_opened":False}
    json_write(status_path,state)
    try:
        for label,adapter in endpoints:
            verify(plan)
            args=["-m","reasoning_rl","evaluate","--name",f"{FAMILY}-{label}-confirmation","--hardware","rocm",
                  "--config",f"configs/{FAMILY}_confirm.yaml"]
            if adapter:
                args += ["--adapter",adapter]
            run_phase(label+"-confirmation",args,state,status_path,timeout=reserve)
        state.update(status="completed_one_stage_confirmation",finished_unix=time.time())
        json_write(status_path,state)
    except BaseException as error:
        state.update(status="incomplete_confirmation",error=repr(error),finished_unix=time.time())
        json_write(status_path,state)
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["comparison","confirmation","all"],default="all")
    args=parser.parse_args()
    plan=read(EXP/"sampling_frozen.json")
    if args.stage in {"comparison","all"}:
        comparison(plan)
    if args.stage in {"confirmation","all"}:
        confirmation(plan)
