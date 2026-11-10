"""Reuse immutable controller with new artifact names after an external chat interruption.

No training/data/budget parameter changes. The original partial run is never overwritten.
"""
import datetime as dt
import json
import shutil
import sys
import time
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, checkpoint_hash
sys.path.insert(0, str(ROOT/"scripts"))
import filter_common as common
import run_filter_controlled as controlled


def main():
    family, exp = common.FAMILY, common.EXP
    receipt_path=exp/"interruption_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Recovery already registered; do not duplicate it")
    original=ROOT/"environment"/(family+"-status.json")
    backup=ROOT/"environment"/(family+"-interrupted-status.json")
    shutil.copyfile(original,backup)
    commands=jsonl_read(exp/"commands.jsonl")
    start=next(r["started_unix"] for r in commands if r["phase"] == "U-s13")
    rows=jsonl_read(ROOT/"runs"/(family+"-U-s13")/"costs.jsonl")
    last=max(r["time_unix"] for r in rows)
    confirmed_absent=dt.datetime(2026,11,10,10,6,7,tzinfo=dt.timezone.utc).timestamp()
    assert checkpoint_hash(ROOT/common.START) == common.START_HASH
    json_write(receipt_path,{"registered_unix":time.time(),"event":"Interactive turn was aborted; first owned controller/session no longer exists and no Python process remained at verification",
        "original_partial_run":family+"-U-s13","original_log":f"logs/{family}-U-s13.log",
        "completed_optimizer_steps_observed":136,"saved_endpoint_checkpoint":False,
        "original_process_start_unix":start,"last_observed_event_unix":last,"first_confirmed_absent_unix":confirmed_absent,
        "failed_attempt_observed_minimum_seconds":last-start,
        "failed_attempt_conservative_upper_bound_seconds":confirmed_absent-start,
        "exit_time_and_exact_process_duration_known":False,
        "cost_policy":"Retain failed attempt and its observed/upper-bound interval in campaign cost; do not call its partial weights a valid endpoint. All completed U/F endpoints retain the same frozen allocation. Report operational campaign cost separately from matched successful-endpoint cost.",
        "recovery":"Same frozen controller, same configs/data/subset/S2/budgets/order, new artifact prefix r1; no completed main score was observed or selected",
        "whole_session_deadline_changed":False,"frozen_source_changed":False})
    controlled.verify(common.read(exp/"protocol_frozen.json"))
    controlled.FAMILY=family+"-r1"
    recovery_status=ROOT/"environment"/(controlled.FAMILY+"-status.json")

    def sync_status():
        if recovery_status.exists():
            state=common.read(recovery_status)
            state.update(run_prefix=controlled.FAMILY, recovery_after_external_interruption=True,
                original_interruption_receipt=str(receipt_path.relative_to(ROOT)))
            json_write(original,state)

    def recovered_phase(label,args,**kwargs):
        sync_status()
        try:
            return common.run_phase("r1-"+label,args,**kwargs)
        finally:
            sync_status()

    controlled.run_phase=recovered_phase
    try:
        controlled.main()
    finally:
        sync_status()


if __name__=="__main__":
    main()
