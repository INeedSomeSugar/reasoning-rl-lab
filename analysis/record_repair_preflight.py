"""Current local runtime evidence; no system configuration changes."""
import importlib.metadata
import json
import time
import torch
from reasoning_rl.runtime import ROOT,json_write,jsonl_append,sha256_file,hardware_info

FAMILY="desktoprepair_20261106_152712"
EXP=ROOT/"experiments"/FAMILY
assert torch.version.hip and torch.cuda.device_count()==1
assert "9070 XT" in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported()
packages={p.metadata["Name"]:p.version for p in importlib.metadata.distributions()}
files=["configs/models.lock.json","configs/data.sources.lock.json","data/manifest.json",
       "environment/requirements-desktop-rocm10.lock.txt","environment/desktop_amd_wheels.json",
       "environment/desktop_model_verification.json"]
json_write(ROOT/"environment"/(FAMILY+"-preflight.json"),{"recorded_unix":time.time(),"runtime":hardware_info(),
    "all_packages":dict(sorted(packages.items())),"existing_receipts_sha256":{p:sha256_file(ROOT/p) for p in files},
    "remote_main_verified_via_local_authenticated_git_ls_remote":"4e938eb696b1cbb4290c14515d30069ff6e50113",
    "single_gpu_assertion":"This process exposes exactly RX9070XT as device0; HIP_VISIBLE_DEVICES=0; recorded per-run precision also checks parameter devices",
    "scope":"Existing project-local environment reused. No driver, OS, other project, cloud GPU or paid API changed/used."})
jsonl_append(EXP/"recovery_incidents.jsonl",[
    {"stage":"recovery","observation":"Sandbox CIM process enumeration denied","resolution":"Did not change system permissions; used project process/runtime evidence and PyTorch device isolation assertions"},
    {"stage":"remote_verification","observation":"Sandbox git DNS failure; connected GitHub API returned404 for the private repository","resolution":"Read-only authenticated local git ls-remote outside sandbox succeeded; local and remote main match4e938eb"},
    {"stage":"dependency_check","observation":"python -m pip check unavailable because project uv environment has no pip module","resolution":"Project-local uv pip check verified69 installed packages compatible; no reinstall needed"}])
print("Current single RX9070XT runtime and package evidence saved")
