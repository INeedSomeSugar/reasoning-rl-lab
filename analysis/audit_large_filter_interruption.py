"""Bound lost process cost from raw timestamps and the next recorded OS boot."""
import argparse
import datetime as dt
import json
import sys

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read


def main(family):
    exp = ROOT / "experiments" / family
    bounds_path = exp / "interruption_cost_bounds.jsonl"
    events_path = exp / "system_restart_events_2026-11-11_1701.json"
    events = json.loads(events_path.read_text(encoding="utf-8-sig"))
    commands = jsonl_read(exp / "commands.jsonl")
    exact_processes = jsonl_read(exp / "process_costs.jsonl")
    audited = []
    for raw in jsonl_read(bounds_path):
        label = raw["label"]
        assert not [p for p in exact_processes if p["label"] == label], "Do not double-count an exact process receipt"
        cmd = [c for c in commands if c["label"] == label]
        assert len(cmd) == 1
        cmd = cmd[0]
        started = cmd["started_unix"]
        outer = cmd.get("outer_budget_start_unix") or started
        cost_path = ROOT / "runs" / (family+"-"+label) / "costs.jsonl"
        activity = jsonl_read(cost_path)
        last = max(r["time_unix"] for r in activity)
        boots = [(dt.datetime.fromisoformat(e["TimeCreated"]).timestamp(), e) for e in events
            if e["ProviderName"] == "Microsoft-Windows-Kernel-General" and e["Id"] == 12]
        boots = [(timestamp, e) for timestamp, e in boots if timestamp > last]
        assert boots, "A later OS boot is required for this refined upper bound"
        upper_time, boot = min(boots, key=lambda t: t[0])
        lower = max(0, last-started)
        upper = min(raw["process_seconds_upper_bound"], upper_time-started)
        assert 0 <= lower <= upper and outer <= started
        audited.append({"label": label, "status": "confirmed_system_restart_interruption",
            "started_unix": started, "outer_budget_start_unix": outer, "last_raw_activity_unix": last,
            "next_OS_boot_event_unix": upper_time, "next_OS_boot_event": boot,
            "process_seconds_lower_bound": lower, "process_seconds_upper_bound": upper,
            "failed_endpoint_path_seconds_lower_bound": last-outer,
            "failed_endpoint_path_seconds_upper_bound": upper_time-outer,
            "known_parent_preparation_seconds": started-outer,
            "exact_exit_code": None, "checkpoint_selected": False, "exact_sampler_optimizer_resume": False,
            "raw_costs_sha256": sha256_file(cost_path), "source_interruption_receipt": raw,
            "interpretation": "Lower bound: last timestamp written by the live child. Upper bound: the old process cannot survive the first subsequent OS boot. Downtime is separately in registered session wall time."})
    target = exp / "interruption_cost_audit.json"
    receipt = {"audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "family": family,
        "source_bounds_sha256": sha256_file(bounds_path), "system_restart_events_sha256": sha256_file(events_path),
        "interrupted_processes": audited, "fresh_predictions_read": False,
        "all_raw_and_loose_original_bounds_preserved": True}
    if target.exists():
        existing = read(target)
        assert existing["interrupted_processes"] == audited and existing["source_bounds_sha256"] == receipt["source_bounds_sha256"]
    else:
        json_write(target, receipt)
    print({"interrupted_attempts": len(audited), "bounded_process_seconds": [
        (r["label"], r["process_seconds_lower_bound"], r["process_seconds_upper_bound"]) for r in audited],
        "exact_exit_codes_unknown": True, "no_new_model_or_fresh_inference": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
