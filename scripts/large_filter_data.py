"""Old-data calibration configuration and pre-probe candidate review/fresh-development freeze."""
import argparse
import hashlib
import json
import random
import yaml

from reasoning_rl.runtime import ROOT, json_write, jsonl_read, jsonl_write, sha256_file
from reasoning_rl.data import load_split
from reasoning_rl.fresh_development import KIND, REGISTRY
from large_filter_common import read, write_new, utc_now, matrix_order


def calibrate_setup(family):
    exp = ROOT / "experiments" / family
    cfg = yaml.safe_load((ROOT / "configs/desktopfilter_20261110_U_s13.yaml").read_text())
    old_dev = read(ROOT / "experiments/desktoprepair_20261106_152712/development_ids.json")["comparison"][:64]
    write_new(exp / "calibration_development_ids.json", {"comparison": old_dev, "scope": "already-used historical development64, timing only"})
    cfg["evaluation"].update(validation_size=64, validation_ids_path=f"experiments/{family}/calibration_development_ids.json")
    cfg["grpo"].pop("candidate_pool_path")
    cfg["grpo"].update(max_wall_seconds=3600, excluded_train_ids=[], experiment_arm="old_data_calibration")
    cfg["stage"] = "large_fixed_prefilter_cost_calibration_only"
    path = ROOT / "configs" / f"{family}_calibration.yaml"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    old_ids = read(ROOT / "experiments/desktopfilter_20261110/candidate_pool.json")["ids"]
    flags = set(read(exp / "preregistration.json")["known_questionable_ids"])
    train_ids = [i for i in old_ids if i not in flags][:16]
    write_new(exp / "calibration_training_pool.json", {"ids": train_ids,
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"), "scope": "historical training items, not new candidates"})
    cfg["grpo"].update(candidate_pool_path=f"experiments/{family}/calibration_training_pool.json", steps=8,
        max_wall_seconds=300, pool_size=16, standalone_endpoints_only=True)
    (ROOT / "configs" / f"{family}_calibration_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    write_new(exp / "registered_matrix_order.json", {"registered_utc": utc_now(), "order": matrix_order(),
        "reason": "Interleave F/R and rotate U position within each training-seed block. Fixed before probes or new evaluations.",
        "evaluation_order": ["S2", "A"] + [x["key"] for x in reversed(matrix_order())]})


def review_pack(family):
    from transformers import AutoTokenizer
    from reasoning_rl.modeling import render_prompt
    exp = ROOT / "experiments" / family
    by_id = {r["id"]: r for r in load_split("train", "select")}
    tok = AutoTokenizer.from_pretrained(ROOT / "models/qwen3_17b", local_files_only=True)
    order = read(exp / "candidate_order.json")["ordered_representative_ids"]
    review, overlong = [], []
    for index, identifier in enumerate(order):
        row = by_id[identifier]
        length = len(tok.encode(render_prompt(tok, row["question"]), add_special_tokens=False))
        if length > 512:
            overlong.append({"id": identifier, "prompt_tokens": length, "candidate_order_index": index})
            continue
        review.append({**row, "candidate_order_index": index, "prompt_tokens": length,
            "reviewer": "Codex root Agent", "human_reviewed": False, "review_status": "pending"})
        if len(review) == 320:
            break
    if (exp / "candidate_review_pack.jsonl").exists():
        raise FileExistsError("Review pack exists")
    jsonl_write(exp / "candidate_review_pack.jsonl", review)
    write_new(exp / "candidate_length_exclusions.json", {"rule": "max rendered prompt512 before probing", "items": overlong})
    print(json.dumps({"review_pack_size": len(review), "overlong": len(overlong)}))


def freeze(family):
    from transformers import AutoTokenizer
    from reasoning_rl.modeling import render_prompt
    exp = ROOT / "experiments" / family
    if (exp / "data_frozen.json").exists():
        raise FileExistsError("Data already frozen")
    review = jsonl_read(exp / "candidate_review_completed.jsonl")
    assert all(r["review_status"] in {"accept", "exclude"} and r["human_reviewed"] is False for r in review)
    pool = [r for r in review if r["review_status"] == "accept"]
    if len(pool) != 256:
        raise ValueError("Expected the first256 accepted candidates or explicitly document a data-shortfall protocol before inference")
    packed = jsonl_read(exp / "candidate_review_pack.jsonl")
    assert [r["id"] for r in review] == [r["id"] for r in packed[:len(review)]]
    assert pool[-1]["id"] == review[-1]["id"]
    all_clusters = jsonl_read(exp / "clusters.jsonl")
    by_cluster = {c["cluster_id"]: c for c in all_clusters}
    id_cluster = {i: c["cluster_id"] for c in all_clusters for i in c["ids"]}
    pool_ids = [r["id"] for r in pool]
    # Also keep the entire pre-extracted review/backfill pack out of development.
    blocked_clusters = {id_cluster[r["id"]] for r in packed}
    blocked_clusters.update(id_cluster[r["id"]] for r in read(exp / "candidate_length_exclusions.json")["items"])
    clusters = [c for c in all_clusters if c["eligible"] and c["cluster_id"] not in blocked_clusters]
    random.Random(20261115).shuffle(clusters)
    tok = AutoTokenizer.from_pretrained(ROOT / "models/qwen3_17b", local_files_only=True)
    by_id = {r["id"]: r for r in load_split("train", "select")}
    development, selected_clusters, dev_length_excluded = [], [], []
    for c in clusters:
        if any(len(tok.encode(render_prompt(tok, by_id[i]["question"]), add_special_tokens=False)) > 512 for i in c["ids"]):
            dev_length_excluded.append(c["cluster_id"])
            continue
        selected_clusters.append(c["cluster_id"])
        for identifier in c["ids"]:
            development.append({**by_id[identifier], "split": "fresh_development", "original_project_split": "train",
                "cluster_id": c["cluster_id"], "exposure_verification": "project-record unexposed under historical ledger and conservative closure"})
        if len(development) >= 1024:
            break
    if len(development) < 1024:
        raise ValueError("Eligible data shortfall; preserve actual quantities, never use old development/test")
    data_dir = ROOT / "data/fresh_development" / family
    data_dir.mkdir(parents=True, exist_ok=False)
    jsonl_write(data_dir / "rows.jsonl", development)
    dev_ids = [r["id"] for r in development]
    assert not set(dev_ids) & set(pool_ids)
    assert not set(id_cluster[i] for i in dev_ids) & set(id_cluster[i] for i in pool_ids)
    write_new(exp / "candidate_pool.json", {"ids": pool_ids,
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"), "clusters": {i: id_cluster[i] for i in pool_ids},
        "review_sha256": sha256_file(exp / "candidate_review_completed.jsonl"),
        "history_audit_sha256": sha256_file(exp / "history_audit.json"), "project_record_unexposed_only": True})
    manifest_path = data_dir / "manifest.json"
    manifest = {"kind": KIND, "family": family, "frozen_utc": utc_now(), "n": len(development),
        "ids": dev_ids, "id_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
        "rows_path": (data_dir / "rows.jsonl").relative_to(ROOT).as_posix(), "rows_sha256": sha256_file(data_dir / "rows.jsonl"),
        "original_data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
        "original_train_file_sha256": read(ROOT / "data/manifest.json")["splits"]["train"]["sha256"],
        "source": "GSM8K official train; posterior isolated fresh DEVELOPMENT, not formal test or public-pretraining-unseen",
        "clusters": selected_clusters, "cluster_members": {c: by_cluster[c]["ids"] for c in selected_clusters},
        "development_sampling_seed": 20261115, "length_excluded_clusters": dev_length_excluded,
        "history_ledger_sha256": sha256_file(exp / "history_exposure_ledger.jsonl"),
        "candidate_pool_sha256": sha256_file(exp / "candidate_pool.json"),
        "known_questionable_cluster_exclusion": True, "near_duplicate_threshold": 0.90,
        "formal_holdout_used": False, "public_pretraining_exposure": "unknown"}
    write_new(manifest_path, manifest)
    registry_path = ROOT / REGISTRY
    registry = read(registry_path) if registry_path.exists() else {"entries": []}
    registry["entries"].append({"family": family, "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path), "reserved_for_development": True})
    json_write(registry_path, registry)
    write_new(exp / "data_frozen.json", {"frozen_utc": utc_now(), "candidate_n": len(pool), "development_n": len(development),
        "candidate_pool_sha256": sha256_file(exp / "candidate_pool.json"),
        "fresh_manifest_path": manifest_path.relative_to(ROOT).as_posix(), "fresh_manifest_sha256": sha256_file(manifest_path),
        "reviewed_candidates": len(review), "review_exclusions": [r["id"] for r in review if r["review_status"] == "exclude"],
        "original_manifest_unchanged": True, "formal_holdout_opened": False})
    print(json.dumps({"candidate_n": len(pool), "development_n": len(development), "reviewed_candidates": len(review)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--stage", choices=["calibrate_setup", "review_pack", "freeze"], required=True)
    args = p.parse_args()
    globals()[args.stage](args.family)
