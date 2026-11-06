"""Local handoff archives only: no commits, uploads, full model or environment."""
import datetime as dt
import json
import subprocess
import zipfile
from pathlib import Path
from reasoning_rl.runtime import ROOT,json_write,sha256_file,checkpoint_hash

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"dist"/FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    state=read(ROOT/"environment"/(FAMILY+"-confirmation-status.json"))
    assert state["status"]=="completed_one_stage_confirmation"
    assert read(ROOT/"reports/comparisons"/FAMILY/"evidence_audit.json")["status"]=="pass"
    if OUT.exists():
        raise FileExistsError("Preserve existing local archives")
    OUT.mkdir(parents=True)
    checkpoints=sorted(p for p in (ROOT/"checkpoints").glob(FAMILY+"*") if (p/"adapter_model.safetensors").exists())
    allowed={"adapter_model.safetensors","adapter_config.json","rrl_checkpoint.json"}
    records=[]
    for path in checkpoints:
        meta=read(path/"rrl_checkpoint.json")
        records.append({"checkpoint":path.relative_to(ROOT).as_posix(),"stage":meta["stage"],
            "role":"untrained_control_not_SFT" if meta["stage"]=="untrained_lora_control" else
                   "cost_learning_signal_calibration" if "-calibration-" in path.name else "trained_endpoint",
            "checkpoint_sha256":checkpoint_hash(path),
            "files":{p.name:{"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sorted(path.iterdir()) if p.name in allowed}})
    manifest=ROOT/"migration"/(FAMILY+"_CHECKPOINTS_MANIFEST.json")
    counts={stage:sum(r["stage"]==stage for r in records) for stage in sorted({r["stage"] for r in records})}
    json_write(manifest,{"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"family":FAMILY,"checkpoints":records,
        "count_by_stage":counts,"checkpoint_count":len(records),
        "qualification":"Untrained control and calibration endpoints are explicitly distinguished; not all are scientific training comparisons",
        "excluded":["public model weights","tokenizer copies (use pinned base)","environment","caches","complete optimizer state"],
        "common_start":read(EXP/"common_start.json")})
    weights=OUT/(FAMILY+"-lora.zip")
    with zipfile.ZipFile(weights,"x",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for checkpoint in checkpoints:
            for path in sorted(checkpoint.iterdir()):
                if path.name in allowed:
                    archive.write(path,"reasoning-rl-lab/"+path.relative_to(ROOT).as_posix())
        archive.write(manifest,"reasoning-rl-lab/"+manifest.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(weights) as archive:
        assert archive.testzip() is None
    # Use Git's ignore rules and an explicit project-content allowlist.
    raw=subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard"],cwd=ROOT)
    names=sorted(set(n.decode("utf-8") for n in raw.split(b"\0") if n))
    folders={"src","scripts","tests","configs","requirements","docs","reports","runs","logs","experiments","environment","migration","analysis","data"}
    root_names={".gitignore",".gitattributes","AGENTS.md","README.md","HANDOFF.md","PROJECT_CONTEXT.md","pyproject.toml"}
    files=[]
    for name in names:
        relative=Path(name)
        if (relative.parts[0] in folders or name in root_names) and (ROOT/relative).is_file():
            assert (ROOT/relative).resolve().is_relative_to(ROOT.resolve())
            if relative.suffix.lower() not in {".safetensors",".pt",".bin",".zip",".exe",".dll"}:
                files.append(ROOT/relative)
    source=OUT/(FAMILY+"-source-and-evidence.zip")
    with zipfile.ZipFile(source,"x",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path,"reasoning-rl-lab/"+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
    receipt={"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"publication_status":"local_only_not_committed_or_pushed",
        "archives":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in (source,weights)],
        "checkpoint_count":len(records),"count_by_stage":counts,"source_and_evidence_file_count":len(files),
        "manifest":manifest.relative_to(ROOT).as_posix(),"manifest_sha256":sha256_file(manifest),
        "public_model_or_environment_included":False,"optimizer_exact_resume_supported":False,
        "note":"Archive receipt is generated after the source snapshot and is supplied beside the ZIPs, avoiding a self-hash cycle. Source evidence includes preserved historical records; holdout files are only archived as opaque bytes, not deserialized or evaluated."}
    json_write(ROOT/"environment"/(FAMILY+"-local-archives.json"),receipt)
    json_write(OUT/"archive_receipt.json",receipt)
    (OUT/"SHA256SUMS.txt").write_text("\n".join(f"{r['sha256']}  {Path(r['path']).name}" for r in receipt["archives"])+"\n",encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
