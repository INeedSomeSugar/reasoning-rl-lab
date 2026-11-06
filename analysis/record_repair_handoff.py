"""Record source integrity and the complete local pending-file inventory, without Git writes."""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from reasoning_rl.runtime import ROOT,json_write,jsonl_read,sha256_file,code_hash

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"reports/comparisons"/FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources():
    verification=read(EXP/"public_model_verification.json")
    files=[]
    for item in verification["files"]:
        path=ROOT/"models/qwen3_17b"/item["name"]
        digest=sha256_file(path)
        assert digest==item["sha256"]
        files.append({"name":item["name"],"sha256":digest,"matches_initial_verified_receipt":True})
    json_write(EXP/"source_reverification_at_handoff.json",{"time_unix":time.time(),"files":files,
        "public_model_unchanged":True,"method":"Local bytes rehashed; no vendor download or new external assertion"})
    mapping={}
    for folder in ("src","configs","scripts","tests","analysis","requirements"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file() and path.suffix in {".py",".ps1",".yaml",".json",".txt"}:
                mapping[path.relative_to(ROOT).as_posix()]=sha256_file(path)
    mapping["pyproject.toml"]=sha256_file(ROOT/"pyproject.toml")
    json_write(EXP/"final_source_manifest.json",{"files":mapping,"frozen_execution_code_sha256":code_hash(),
        "analysis_source_added_after_execution_freeze":True,
        "scope":"Executable source/config/tests and reproducible post-hoc analysis; raw-run code snapshots retain earlier SFT-development variants"})
    print(json.dumps({"public_source_reverified":len(files),"source_files_hashed":len(mapping),"code_sha256":code_hash()},indent=2))


def pending():
    modified=subprocess.check_output(["git","diff","--name-only","-z"],cwd=ROOT).split(b"\0")
    new=subprocess.check_output(["git","ls-files","--others","--exclude-standard","-z"],cwd=ROOT).split(b"\0")
    modified=sorted(x.decode("utf-8") for x in modified if x)
    new=sorted(x.decode("utf-8") for x in new if x)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/"environment"/(FAMILY+"-local-archives.json"))
    adapters=read(ROOT/"migration"/(FAMILY+"_CHECKPOINTS_MANIFEST.json"))
    payload={"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"local_head":head,
        "verified_remote_main_at_recovery":"4e938eb696b1cbb4290c14515d30069ff6e50113",
        "committed_or_pushed_by_this_study":False,"modified_tracked_paths":modified,"new_git_visible_paths":new,
        "counts":{"modified_tracked":len(modified),"new_git_visible":len(new)},
        "git_ignored_checkpoints_for_separate_backup":[r["checkpoint"] for r in adapters["checkpoints"]],
        "local_archives":receipt["archives"],"full_model_environment_caches_not_for_commit":True,
        "inventory_note":"This inventory and final receipts are generated after the source ZIP snapshot; self-reference and later receipt paths are intentionally not recursively hashed."}
    json_write(OUT/"pending_files.json",payload)
    lines=["# 本轮待提交与备份清单","",f"家族：`{FAMILY}`。本轮没有执行Git提交或推送，当前HEAD仍为 `{head}`。历史工件保留。","",
        f"Git中已跟踪文件修改 {len(modified)} 项，新增可见文件 {len(new)} 项（逐项路径见 [完整JSON清单](pending_files.json)，不在正文展开每份run的源码快照）。","",
        "|工件组|位置 / 处理|","|---|---|",
        "|核心代码|`src/reasoning_rl/{training,sampling,evaluation,cli,runtime}.py`：缓存采样、显式起点、验证ID与监督/成本证据|",
        f"|配置与工程检查|`configs/main.yaml`、`configs/{FAMILY}_*.yaml`、三个新增测试文件；47项测试通过|",
        "|执行与复算|新增` scripts/*repair*`及`analysis/`脚本，复现文档与固定命令收据|",
        f"|实验登记|`experiments/{FAMILY}/`：ID、曝光、四候选、定标、冻结协议、确认开启清单及源码摘要|",
        f"|原始证据|49份`runs/{FAMILY}-*`、对应`logs/`与`environment/`；监督、预测、rollout、探测、缓存和成本|",
        f"|中文报告|`reports/comparisons/{FAMILY}/`：报告、CSV、逐题预测、图、归因与证据审计|",
        "|项目交接|README、PROJECT_CONTEXT、HANDOFF、PROTOCOL、台式机记录与新复现说明|",
        f"|真实适配器（Git忽略）|{receipt['checkpoint_count']}份；SFT4、GRPO12（3定标+9主终点）、未训练控制1，不能都称正式对照。逐文件摘要见`migration/{FAMILY}_CHECKPOINTS_MANIFEST.json`|","",
        "## 已生成的本地归档","",
        "以下只在本机生成，未上传。公开基础模型、环境、缓存与完整优化器状态不在包内。归档收据在源ZIP生成之后单独提供，避免摘要自引用。","",
        "|归档|字节数|SHA256|","|---|---:|---|"]
    for item in receipt["archives"]:
        path="../../../"+item["path"]
        lines.append(f"|[{Path(item['path']).name}]({path})|{item['bytes']}|`{item['sha256']}`|")
    lines += ["",f"[归档收据](../../../environment/{FAMILY}-local-archives.json) · [SHA256SUMS](../../../dist/{FAMILY}/SHA256SUMS.txt)。源ZIP是生成时的完整源码/证据快照，不包含随后生成的收据或本清单最终版本。", "",
        "本轮不上传模型/缓存，不把本地备份写成已发布Release。提交时应将小型源码、配置、文档、原始证据纳入Git；LoRA ZIP按单独的大工件备份处理。"]
    (OUT/"pending_artifacts.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(payload["counts"],indent=2))


def close():
    state=read(EXP/"session.json")
    start=dt.datetime.fromisoformat(state["started_utc"]).timestamp()
    ended=time.time()
    assert read(ROOT/"environment"/(FAMILY+"-confirmation-status.json"))["status"]=="completed_one_stage_confirmation"
    assert read(OUT/"evidence_audit.json")["status"]=="pass"
    assert code_hash()==read(EXP/"sampling_frozen.json")["code_sha256"]
    eval_count=sum(len(jsonl_read(p)) for p in (ROOT/"runs").glob(FAMILY+"*/**/predictions.jsonl"))
    rollout_count=sum(len(jsonl_read(p)) for p in (ROOT/"runs").glob(FAMILY+"*/rollouts.jsonl"))
    probe_count=sum(len(jsonl_read(p)) for p in (ROOT/"runs").glob(FAMILY+"*/probes/step_*.jsonl"))
    result={"status":"completed_repair_development_study","family":FAMILY,"started_utc":state["started_utc"],
        "ended_utc":dt.datetime.fromtimestamp(ended,dt.timezone.utc).isoformat(),"ended_beijing":dt.datetime.fromtimestamp(ended,dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "total_wall_seconds_including_handoff":ended-start,"six_hour_cap_seconds":21600,"within_cap":ended-start<=21600,
        "completed_scope":{"SFT_candidates":4,"main_CDE_seeds":[13,29,47],"main_endpoints":9,
            "comparison_questions_per_endpoint":128,"one_stage_confirmation_questions_per_endpoint":128,
            "confirmed_fixed_endpoints":11,"new_runs_evidence_audited":read(OUT/"evidence_audit.json")["runs_checked"],
            "saved_evaluation_predictions":eval_count,"saved_training_rollouts":rollout_count,"saved_probe_completions":probe_count},
        "model_tasks_finished_unix":read(ROOT/"environment"/(FAMILY+"-confirmation-status.json"))["finished_unix"],
        "holdout_test_SVAMP_opened":False,"cloud_GPU_or_paid_API_used":False,"committed_or_pushed":False,
        "source_integrity_code_sha256":code_hash(),"local_archives_receipt":f"environment/{FAMILY}-local-archives.json",
        "note":"Timestamp closes local artifact/handoff work; subsequent chat delivery adds only its own display time. No model training or evaluation remains active."}
    json_write(EXP/"session_final.json",result)
    json_write(ROOT/"environment"/(FAMILY+"-status.json"),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["sources","pending","close"])
    args=parser.parse_args()
    {"sources":sources,"pending":pending,"close":close}[args.phase]()
