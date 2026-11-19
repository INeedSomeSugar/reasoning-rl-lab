"""Bounded CPU-only check that hidden execution survives the launching shell."""
import argparse
import os
import time
from reasoning_rl.runtime import ROOT, json_write

p = argparse.ArgumentParser()
p.add_argument("--family", required=True)
args = p.parse_args()
target = ROOT / "experiments" / args.family / "persistent_launch_check.json"
if target.exists():
    raise FileExistsError(target)
started = time.time()
for i in range(30):
    json_write(target, {"pid": os.getpid(), "started_unix": started,
        "heartbeat_unix": time.time(), "iteration": i, "status": "running", "model_task": False})
    time.sleep(1)
json_write(target, {"pid": os.getpid(), "started_unix": started, "finished_unix": time.time(),
    "status": "completed", "model_task": False})
