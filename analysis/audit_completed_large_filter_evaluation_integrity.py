"""CPU-only integrity checks for saved evaluations, without displaying outcomes."""
import argparse
import datetime as dt
import json
import statistics
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read
from large_filter_status import status


def main(family):
    exp = ROOT / "experiments" / family
    plan = read(exp / "protocol_frozen.json")
    units = read(exp / "evaluation_units.json")
    lock = read(exp / "all_training_endpoints_fixed.json")
    fresh = read(ROOT / plan["fresh_manifest_path"])
    training = read(exp / "training_units.json")
    records, durations = [], []
    for key in plan["evaluation_order"]:
        item = units[key]
        if item["status"] != "first_success_fixed":
            continue
        directory = ROOT / "runs" / item["run_name"]
        path = directory / "fresh_development/predictions.jsonl"
        assert sha256_file(path) == item["predictions_sha256"]
        with path.open(encoding="utf-8") as stream:
            actual_ids = [json.loads(line)["id"] for line in stream if line.strip()]
        assert actual_ids == fresh["ids"] and len(actual_ids) == plan["development_n"]
        assert read(directory / "resolved_evaluation_ids.json")["ids"] == actual_ids
        assert read(directory / "evaluation_protocol.json")["adapter_sha256"] == lock["objects"][key]["checkpoint_sha256"]
        metrics = read(directory / "fresh_development/metrics.json")
        assert metrics["n"] == plan["development_n"] and metrics["decoding"] == "greedy"
        records.append({"endpoint": key, "n": len(actual_ids), "predictions_sha256": item["predictions_sha256"],
            "actual_prediction_ID_order_matches_fresh_manifest": True, "adapter_identity_matches_weight_lock": True})
        if key in training:
            durations.append(item["evaluation_process_seconds"])
    remaining = sum(item["status"] == "running_attempt" or (item["status"] in ("pending", "pending_retry")
        and len(item.get("attempts", [])) < 2) for item in units.values())
    exhausted = sum(item["status"] == "exhausted_failure" for item in units.values())
    result = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "all_completed_evaluation_integrity_passed", "completed_evaluations": len(records),
        "records": records, "no_accuracy_values_or_response_text_displayed": True, "main_statistics_not_run": True,
        "remaining_eligible_objects_for_time_estimate": remaining, "exhausted_missing_objects_excluded_from_ETA": exhausted,
        "completed_trained_endpoint_process_seconds_mean": statistics.mean(durations),
        "remaining_model_hours_at_observed_mean": remaining*statistics.mean(durations)/3600}
    path = exp / ("evaluation_integrity_through_"+str(len(records))+".json")
    if path.exists():
        previous = read(path)
        assert previous["records"] == records
    else:
        json_write(path, result)
    current = status(family)
    print(json.dumps({k: current.get(k) for k in ("status", "phase", "runner_pid", "successful_training",
        "successful_evaluation", "current_evaluation_predictions", "heartbeat_age_seconds", "error")}, ensure_ascii=False))
    print(json.dumps({"integrity_verified": len(records), "artifact": str(path),
        "remaining_model_hours_at_observed_mean": result["remaining_model_hours_at_observed_mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
