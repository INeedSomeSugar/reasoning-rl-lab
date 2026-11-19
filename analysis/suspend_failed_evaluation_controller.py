"""Protect remaining attempts while diagnosing rapid evaluation failures."""
import json
import time
import psutil
from reasoning_rl.runtime import ROOT, jsonl_append

parent = psutil.Process(34984)
assert "resume_large_filter_evaluation.py" in " ".join(parent.cmdline())
parent.suspend()
children = [{"pid": p.pid, "cmdline": p.cmdline()} for p in parent.children(recursive=True)]
record = {"event": "protective_controller_suspension_after_rapid_evaluation_failures",
    "label": "evaluation-controller-diagnostic-suspension", "runner_pid": parent.pid,
    "time_unix": time.time(), "model_children_not_suspended": True, "children": children}
jsonl_append(ROOT / "experiments/desktopfilterrep_2026-11-10_235416/process_events.jsonl", [record])
print(json.dumps(record, ensure_ascii=False))
