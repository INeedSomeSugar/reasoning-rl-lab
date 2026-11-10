"""Raw K-shot filtering and explicit standalone versus amortized cost accounting."""
from collections import Counter
import json
import math
from pathlib import Path
from .runtime import ROOT, jsonl_read, sha256_file


def local_file(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen study files must stay project-local")
    return path


def raw_success_filter(candidate_ids, predictions, k=4):
    """Jeffreys smoothing is deliberately irrelevant to eligibility."""
    if k != 4 or len(set(candidate_ids)) != len(candidate_ids) or not candidate_ids:
        raise ValueError("This controlled filter requires nonempty unique candidates and K4")
    observed, success = Counter(), Counter()
    ids = set(candidate_ids)
    for r in predictions:
        if r["id"] not in ids or type(r["correct"]) is not bool:
            raise ValueError("Probe records must be binary-scored members of the candidate pool")
        observed[r["id"]] += 1
        success[r["id"]] += int(r["correct"])
    if any(observed[i] != k for i in candidate_ids):
        raise ValueError("Require exactly K raw observations per candidate; no partial cache")
    return [i for i in candidate_ids if success[i] in (1, 2, 3)], {i: success[i] for i in candidate_ids}


def charge_probe(physical_seconds, endpoint_count=3):
    if not math.isfinite(physical_seconds) or physical_seconds <= 0 or endpoint_count <= 0:
        raise ValueError("Probe cost must be a measured positive finite duration")
    return {"physical_one_time_seconds": physical_seconds,
        "primary_standalone_debit_per_F_endpoint_seconds": physical_seconds,
        "secondary_amortized_debit_per_F_endpoint_seconds": physical_seconds / endpoint_count,
        "F_endpoint_count": endpoint_count, "never_free": True}


def remaining_training_seconds(common_seconds, elapsed_seconds, debit_seconds=0.0):
    if any(not math.isfinite(x) or x < 0 for x in (common_seconds, elapsed_seconds, debit_seconds)):
        raise ValueError("Budgets and elapsed costs must be nonnegative finite seconds")
    return max(0.0, common_seconds - elapsed_seconds - debit_seconds)


def resolve_frozen_pool(options, train, start_hash):
    """Validate paid probe bytes, raw membership and fixed S2 before uniform sampling."""
    path = local_file(options["candidate_pool_path"])
    candidate = json.loads(path.read_text(encoding="utf-8"))
    ids = candidate["ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Candidate pool IDs must be nonempty and unique")
    if candidate["data_manifest_sha256"] != sha256_file(ROOT / "data/manifest.json"):
        raise ValueError("Candidate data changed")
    by_id = {r["id"]: r for r in train}
    if not set(ids) <= set(by_id) or any(by_id[i]["split"] != "train" for i in ids):
        raise ValueError("Candidate pool must be train-only")
    chosen, debit, evidence = ids, 0.0, None
    if options.get("prefilter_manifest_path"):
        manifest = json.loads(local_file(options["prefilter_manifest_path"]).read_text(encoding="utf-8"))
        if manifest["candidate_pool_sha256"] != sha256_file(path) or manifest["s2_checkpoint_sha256"] != start_hash:
            raise ValueError("Probe candidates or fixed S2 differ")
        raw_path = local_file(manifest["predictions_path"])
        if sha256_file(raw_path) != manifest["predictions_sha256"]:
            raise ValueError("Paid probe output changed")
        chosen, successes = raw_success_filter(ids, jsonl_read(raw_path), manifest["k"])
        if chosen != manifest["retained_ids"] or successes != manifest["raw_success_counts"]:
            raise ValueError("Frozen raw-count filter membership changed")
        if not chosen:
            raise ValueError("Empty filtered pool; do not silently change the probe seed")
        evidence = charge_probe(manifest["probe_process_wall_seconds"])
        debit = evidence["primary_standalone_debit_per_F_endpoint_seconds"]
    return [by_id[i] for i in chosen], ids, debit, evidence
