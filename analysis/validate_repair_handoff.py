"""Check final artifact bindings, arithmetic, links and explicit development scope."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from PIL import Image
from reasoning_rl.runtime import ROOT,json_write,code_hash,sha256_file

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
OUT=ROOT/"reports/comparisons"/FAMILY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(prearchive=False):
    summary=read(OUT/"summary.json")
    plan=read(EXP/"sampling_frozen.json")
    assert summary["all_nine_comparison_complete"] and summary["all_nine_confirmation_complete"]
    assert code_hash()==plan["code_sha256"]
    with (OUT/"standalone_endpoints.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==9 and sum(int(r["steps"]) for r in rows)==979
    assert sum(int(r["nonzero_gradient_steps"]) for r in rows)==244
    for r in rows:
        total=float(r["training_process_seconds"])+float(r["independent_comparison_process_seconds"])
        assert math.isclose(total,float(r["actual_end_to_end_seconds"]),abs_tol=1e-6)
        assert math.isclose(float(r["common_end_to_end_cap_seconds"])-total,float(r["end_to_end_unused_seconds"]),abs_tol=1e-6)
        assert r["common_start_sha256"]==plan["common_start"]["checkpoint_sha256"]
        assert r["comparison_n"]==r["confirmation_n"]=="128"
    md=(OUT/"findings.md").read_text(encoding="utf-8")
    assert not any(p in md for p in ("确认正在执行","阶段确认尚在写入","本节将在完整结束后","最终交接将补充"))
    assert "正式独立测试" in md and "不声称" in md
    docs=[ROOT/"README.md",ROOT/"PROJECT_CONTEXT.md",ROOT/"HANDOFF.md",ROOT/"docs/PROTOCOL.md",
          ROOT/"docs/DESKTOP_SETUP_20261105.md",ROOT/"docs/REPRODUCING_REPAIR_20261106.md",OUT/"findings.md"]
    if (OUT/"pending_artifacts.md").exists():
        docs.append(OUT/"pending_artifacts.md")
    checked,missing=[],[]
    allowed_future={"session_final.json","pending_artifacts.md"}
    for path in docs:
        text=path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
            if target.startswith(("https:","http:","#","codex:")):
                continue
            name=target.split("#",1)[0].strip("<>")
            resolved=(path.parent/name).resolve()
            if not resolved.exists():
                if prearchive and resolved.name in allowed_future:
                    continue
                missing.append({"source":path.relative_to(ROOT).as_posix(),"target":target})
            checked.append({"source":path.relative_to(ROOT).as_posix(),"target":target})
    assert not missing,missing
    images=[]
    for name in ("standalone_comparison.png","cost_ledger.png","effective_pool_observed.png","sft_strength.png"):
        p=OUT/name
        with Image.open(p) as im:
            assert im.width>=1000 and im.height>=600
            images.append({"file":name,"width":im.width,"height":im.height,"sha256":sha256_file(p)})
    assert read(OUT/"evidence_audit.json")["status"]=="pass"
    visual=read(OUT/"visual_qa.json")
    assert all(visual["figure_sha256"][r["file"]]==r["sha256"] for r in images)
    assert not any((ROOT/"experiments/final_claims").glob("*.json"))
    result={"status":"prearchive_pass" if prearchive else "pass","local_links_checked":len(checked),"missing_links":missing,
        "figures":images,"csv_arithmetic_reconciles":True,"all_nine_endpoint_sets_complete":True,
        "source_execution_hash_matches_freeze":True,"formal_holdout_not_opened":True,
        "visual_review":"Agent inspection is separately recorded in visual_qa.json and bound to exact image hashes; this script checks bindings and dimensions, not visual acceptance"}
    json_write(OUT/("prearchive_validation.json" if prearchive else "handoff_validation.json"),result)
    print(json.dumps({"status":result["status"],"local_links_checked":len(checked),"images":len(images)},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prearchive",action="store_true")
    main(parser.parse_args().prearchive)
