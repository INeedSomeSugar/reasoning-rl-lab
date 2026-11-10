"""Hash the requested handoff materials and close the session without publication."""
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from reasoning_rl.runtime import ROOT,json_write,sha256_file,checkpoint_hash
sys.path.insert(0,str(ROOT/"scripts"))
from filter_common import EXP,FAMILY,START,START_HASH,read
from run_filter_controlled import verify


def main():
    status=read(ROOT/"environment"/(FAMILY+"-status.json"))
    assert status["status"] == "completed_three_paired_seed_development"
    verify(read(EXP/"protocol_frozen.json"))
    assert read(EXP/"report_written.json")
    assert read(ROOT/"reports/comparisons"/FAMILY/"evidence_audit.json")["all_checks_passed"]
    assert checkpoint_hash(ROOT/START) == START_HASH
    if (EXP/"artifact_manifest.json").exists():
        raise FileExistsError("Preserve an existing closing manifest")
    report=ROOT/"reports/comparisons"/FAMILY/"findings.md"
    md=report.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)",md):
        if "://" in target or target.startswith("#"):
            continue
        assert (report.parent/target.split("#")[0]).resolve().exists(),target
    pending=ROOT/"reports/comparisons"/FAMILY/"pending_artifacts.md"
    git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)
    pending.write_text("# 本地交接与提交状态\n\n全部模型任务、离线复算和交接文档已完成。本轮未提交、未推送、未发布Release；旧提交与负面工件保留。\n\n"
        "主要新代码为固定原始K4筛选、完整探测扣费、实际loss-mask/advantage对账和逐优化步参数改变量记录。"
        "原中断U/13有136步且无终点，主结果来自r1新前缀的六个完成终点；原失败和恢复过程均保留。\n\n"
        "正式GSM8K test/SVAMP仍封存。不要把优化器诊断状态称为已支持精确断点续训。\n\n"
        "下列是收尾时Git状态（文件内容摘要见artifact_manifest.json）：\n\n```text\n"+git_status+"```\n",encoding="utf-8",newline="\n")
    scopes=[EXP,ROOT/"reports/comparisons"/FAMILY,ROOT/"src",ROOT/"scripts",ROOT/"tests",ROOT/"analysis",ROOT/"requirements"]
    scopes+=list((ROOT/"runs").glob(FAMILY+"-*"))
    scopes+=list((ROOT/"checkpoints").glob(FAMILY+"-*"))
    scopes+=[ROOT/START]
    files=set()
    for scope in scopes:
        for path in scope.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                assert path.resolve().is_relative_to(ROOT.resolve())
                files.add(path)
    for pattern in ("configs/*",f"logs/{FAMILY}*.log",f"environment/{FAMILY}*","environment/requirements-desktop-rocm10.lock.txt"):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for relative in ("README.md","HANDOFF.md","PROJECT_CONTEXT.md","AGENTS.md","pyproject.toml",
        "docs/REPRODUCING_FILTER_20261110.md","data/manifest.json","data/processed/train.jsonl",
        "data/processed/validation.jsonl","data/audit/near_pairs.jsonl","data/audit/summary.json",
        "data/raw/sources.lock.json","data/raw/gsm_license.txt","data/raw/svamp_license.txt"):
        path=ROOT/relative
        if path.exists():
            files.add(path)
    rows=[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sorted(files)]
    json_write(EXP/"artifact_manifest.json",{"family":FAMILY,"created_unix":time.time(),"files":rows,
        "excludes":"Public base weights/environment caches/git metadata/formal holdout contents; model hashes are in preflight.json. Manifest and later session_final.json are not self-hashed.",
        "publication":"Local workspace only; no commit/push/release created"})
    now=time.time()
    session=read(EXP/"session.json")
    zone=dt.timezone(dt.timedelta(hours=8))
    json_write(EXP/"session_final.json",{"family":FAMILY,"status":"completed_three_paired_seed_development_and_handoff",
        "model_finished_unix":status["finished_unix"],"model_finished_beijing":dt.datetime.fromtimestamp(status["finished_unix"],zone).isoformat(),
        "handoff_finished_unix":now,"handoff_finished_beijing":dt.datetime.fromtimestamp(now,zone).isoformat(),
        "wall_through_models_seconds":status["finished_unix"]-session["started_unix"],
        "wall_through_handoff_seconds":now-session["started_unix"],"max_session_seconds":session["max_wall_seconds"],
        "model_tasks_within_original_deadline":status["finished_unix"]<=session["deadline_unix"],
        "handoff_manifest_sha256":sha256_file(EXP/"artifact_manifest.json"),"manifest_file_count":len(rows),
        "common_start_sha256":START_HASH,"holdout_opened":False,"cloud_gpu_used":False,"paid_api_used":False,
        "resume_note":"Original interrupted136-step attempt preserved, six completed endpoints use r1 names with original frozen parameters and budgets",
        "git_publication":"Not performed; local changes and artifacts are reviewable"})
    print(json.dumps({"closed":True,"manifest_files":len(rows),"model_finished_beijing":dt.datetime.fromtimestamp(status["finished_unix"],zone).isoformat(),
        "handoff_finished_beijing":dt.datetime.fromtimestamp(now,zone).isoformat()}))


if __name__=="__main__":
    main()
