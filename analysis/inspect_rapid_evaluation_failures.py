"""Read-only diagnosis; no model load and no accuracy values displayed."""
import json
from reasoning_rl.runtime import ROOT

exp = ROOT / "experiments/desktopfilterrep_2026-11-10_235416"
units = json.loads((exp / "evaluation_units.json").read_text(encoding="utf-8"))
for key, item in units.items():
    if item["status"] in {"pending", "first_success_fixed"}:
        continue
    print(json.dumps({"key": key, "status": item["status"], "attempts": [
        {k: a.get(k) for k in ("label", "status", "error")} | {"receipt": {
            k: a.get("receipt", {}).get(k) for k in ("pid", "exit_code", "process_wall_seconds", "timeout")}}
        for a in item["attempts"]]}, ensure_ascii=False))
    for attempt in item["attempts"]:
        path = ROOT / "logs" / (attempt["run_name"]+".log")
        print("LOG "+str(path))
        if path.exists():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-45:]))
        else:
            print("No log file")
