"""Choose capacity strictly from measured elapsed costs, before the full probe."""
import json
import math
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file
from filter_common import EXP, FAMILY, OLD, read, remaining, utc_now


def main():
    assert read(EXP / "calibration_complete.json")["completed"]
    if (EXP / "capacity_frozen.json").exists():
        raise FileExistsError("Capacity already fixed")
    reg = read(EXP / "calibration_preregistration.json")
    costs = {r["phase"]: r for r in jsonl_read(EXP / "process_costs.jsonl")}
    eval_seconds = costs["S2-development"]["process_wall_seconds"]
    eval_reserve = math.ceil(eval_seconds / 30)*30 + 90
    probe_ledger = jsonl_read(ROOT / "runs" / (FAMILY+"-calibration-probe") / "costs.jsonl")
    probe_generation = next(r["wall_seconds"] for r in probe_ledger if r["phase"] == "fixed_s2_k4_generation")
    probe_process = costs["calibration-probe"]["process_wall_seconds"]
    setup = max(0, probe_process - probe_generation)
    historical = read(OLD / "sampling_frozen.json")["cost_based_capacity"]["actual60_item_probe_seconds"] / 60
    seconds_per_item = max(1.25*probe_generation/16, historical)
    choices = []
    left = remaining()
    selected = None
    for allocation in reg["training_allocation_choices_descending_seconds"]:
        possible = [n for n in reg["pool_capacity_choices_descending"]
            if n*seconds_per_item + setup <= reg["maximum_probe_fraction_of_training_allocation"]*allocation]
        if not possible:
            continue
        n = possible[0]
        needed = 6*(allocation+eval_reserve+60) + n*seconds_per_item + setup + 600
        choice = {"train_allocation_seconds": allocation, "candidate_pool_size": n,
            "estimated_probe_process_seconds": n*seconds_per_item+setup,
            "conservative_whole_session_seconds_needed": needed, "fits_remaining": needed <= left}
        choices.append(choice)
        if choice["fits_remaining"] and selected is None:
            selected = choice
    if selected is None:
        raise RuntimeError("No preregistered six-endpoint capacity fits remaining session time")
    prereg = read(EXP / "candidate_preregistration.json")
    ids = prereg["ids"][:selected["candidate_pool_size"]]
    json_write(EXP / "candidate_pool.json", {"ids": ids,
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
        "candidate_preregistration_sha256": sha256_file(EXP / "candidate_preregistration.json"),
        "selection": "cost-selected prefix; no model scores or development correctness consulted"})
    result = {"frozen_utc": utc_now(), "remaining_session_seconds_at_decision": left,
        "calibration_eval_process_seconds": eval_seconds, "evaluation_reserve_seconds": eval_reserve,
        "probe_calibration_process_seconds": probe_process, "probe_calibration_generation_seconds": probe_generation,
        "estimated_probe_seconds_per_item": seconds_per_item, "estimated_probe_setup_seconds": setup,
        "evaluated_choices": choices, "selected": selected,
        "end_to_end_common_cap_seconds": selected["train_allocation_seconds"]+eval_reserve+60,
        "candidate_pool_sha256": sha256_file(EXP / "candidate_pool.json"),
        "basis": "Timing only. Baseline development accuracy and calibration reward histogram do not select capacity."}
    json_write(EXP / "capacity_frozen.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
