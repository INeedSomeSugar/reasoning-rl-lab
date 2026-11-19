"""Register a new study, verify released history, and build a conservative exposure ledger.

No model inference and no formal holdout content is read by this program.
"""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import time
import zipfile

from reasoning_rl.runtime import ROOT, json_write, jsonl_write, jsonl_read, sha256_file
from reasoning_rl.data import normalize_question, question_hash, UnionFind

ID_RE = re.compile(r"gsm8k-train-\d{5}")
FLAGS = ["gsm8k-train-05134", "gsm8k-train-01188", "gsm8k-train-07439",
         "gsm8k-train-02307", "gsm8k-train-06831"]
RELEASES = [
    ("local-pilot-20261105", "dist/reasoning-rl-lab-desktop-transfer-20261105.zip",
     "461f151bd25230f0743171fa58538d0191ba89dc3368e0fb800f5f5c38a4ccf4"),
    ("desktop-rocm10-pilot-20261105", "dist/desktop-rocm10-pilot-20261105/reasoning-rl-lab-desktop-rocm10-20261105.zip",
     "14c1f915823971d8559226d8d9bb63d856d2f5481c4405bdd378a43f751684d5"),
    ("desktop-repair-cache-dev-20261106", "dist/desktoprepair_20261106_152712/desktoprepair_20261106_152712-source-and-evidence.zip",
     "effe8fba0d18d7e6a86ecc1bc93deeadd89a2a152d266b9bc1f20a6d633e36d3"),
    ("fixed-prefilter-dev-20261110", "dist/desktopfilter_20261110-publication/desktopfilter_20261110-source-and-evidence.zip",
     "a01b79c693b14389fb50d5bdaf30588984ebe965b25f00af83e2f0ef58576516"),
]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def register(family):
    exp = ROOT / "experiments" / family
    exp.mkdir(exist_ok=False)
    stamp = time.time()
    json_write(exp / "session.json", {"family": family, "registered_utc": now(),
        "registered_unix": stamp, "earliest_observed_task_clock_utc": "2026-11-10T15:50:31+00:00",
        "initial_orientation_cost": "Before registration; earliest verified clock is a lower bound, not a measured process interval.",
        "initial_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "initial_git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "hard_limit": "Set from measured calibration and full probe durations before main training; no inherited four-hour limit.",
        "formal_holdouts_opened": False})
    json_write(exp / "preregistration.json", {"registered_utc": now(), "family": family,
        "train_sampling_seed": 20261114, "development_sampling_seed": 20261115,
        "training_seeds": [13, 29, 47, 71, 97], "probe_seeds": [20261111, 20261112, 20261113],
        "random_subset_seeds": [3101, 3102, 3103], "candidate_target": 256, "development_minimum_target": 1024,
        "eligibility": "Original official GSM8K train only; preserve original manifest; exclude original validation, nominal SFT128, any historical training/probe/calibration/development/prediction and conservative historical mentions, flags and their entire near/exact components. Prompt <=512. One candidate per qualified cluster; development contains whole remaining clusters.",
        "candidate_order": "Sort eligible clusters by minimum ID, Python random.Random(20261114).shuffle; smallest eligible ID per cluster. Review in that order, exclude questionable clusters and take first256 accepted. No reward or model-performance selection.",
        "development_order": "After candidate/quality exclusions, sort remaining eligible clusters, random.Random(20261115).shuffle; append whole clusters until >=1024. No development inference until all endpoints frozen.",
        "review": "Agent reviews question and original reference solution under a common rule: coherent quantities, unambiguous requested number, reference derivation and final gold consistent. Preserve originals; never edit gold. human_reviewed=false.",
        "known_questionable_ids": FLAGS, "k": 4, "retain_successes": [1, 2, 3],
        "budget_rule": "P_j=full independent probe process including startup/loading/saving; B=max(3600,300*ceil(3*max(P_j)/300)); F/R deduct P_j in full; U does not. Only time sets B.",
        "evaluation_reserve_rule": "Use independent S2 greedy evaluation on64 already-used development questions. E=300*ceil((1.35*calibration_process_seconds*N/64+120)/300). O=300. C=B+E+O; never refill training from evaluation reserve.",
        "session_rule": "Before any main training freeze primary physical allowance (35 training processes after F/R debits +37*E +3*P already measured + calibration/preparation actual), plus one full retry allowance per failed unit (35 training and37 evaluations), plus20% scheduling/audit margin and six hours analysis. Fixed deadline from session registration; never extend based on fresh outcomes.",
        "retry_rule": "At most two full attempts per unit; new attempt prefix, same S2/config/data. First success is used. No exact optimizer/sampler resume claim. All training retries finish or exhaust before fresh evaluation. Evaluation failures can have one complete same-weight retry; no training thereafter.",
        "degenerate_rule": "If any F_j is empty: preserve all probes and stop the main matrix before training, report unavailable comparisons; no seed/pool/K change. If all256 retained, F/R/U still run as registered and report degeneracy.",
        "statistics": {"conditional_cluster_bootstrap_replicates": 20000, "seed": 20261116,
            "conditional_scope": "Jointly resample fresh-development near-duplicate clusters for the same fixed37 models; training/probe seeds fixed. Reuse each of five U endpoints in all three F-U blocks.",
            "crossed_bootstrap_replicates": 20000,
            "crossed_scope": "Resample five training-seed blocks, three paired selection blocks (F_j/R_j) and question clusters; reuse U within each sampled training-seed block. Exploratory with only3 selection blocks and one candidate pool."},
        "decisions": {"engineering_signal_min_relative_improvement": 0.20, "all_three_probe_blocks_signal_direction_positive": True,
            "development_evidence_min_F_minus_U_pp": 1.0, "conditional95_lower_bound_positive": True,
            "interpretation": "Thresholds prospectively guide a second independent-pool study; are neither guaranteed outcomes nor a power claim. Negative/cross-zero results retained; no result-driven extra runs."},
        "blind_review": {"seed": 20261117, "rule": "For each F-U, F-R, F-S2 comparison use seed29 in each selection block; include all discordant question IDs across those pairs. Hide methods and outcome labels, randomize aliases/order. Also deterministic sample16 jointly correct and16 jointly wrong per comparison; preserve common-seed sampling. Agent review, not human. No unreviewed numeric error called arithmetic error."},
        "calibration": "Only historical used questions for timing/backward/save/reload; no new candidate calibration exposure. No new SFT run.",
        "heldout": "GSM8K official test and SVAMP remain sealed; fresh development is retrospectively isolated original-train data, project-record unexposed only; public pretraining exposure unknown."})
    print(json.dumps({"family": family, "registered": True}))


def audit(family):
    exp = ROOT / "experiments" / family
    if (exp / "history_audit.json").exists():
        raise FileExistsError("History audit already completed")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    rows = []
    for split in ("train", "validation"):
        info = manifest["splits"][split]
        assert sha256_file(ROOT / info["path"]) == info["sha256"]
        rows.extend(jsonl_read(ROOT / info["path"]))
    rows.sort(key=lambda r: r["id"])
    by_id = {r["id"]: r for r in rows}
    valid = set(by_id)
    sources, versions, restored, archives = [], set(), [], []

    def accept(relative, content, origin):
        key = (relative, hashlib.sha256(content).hexdigest())
        if key in versions:
            return
        versions.add(key)
        sources.append((relative, content, origin, key[1]))

    eligible_roots = {"runs", "experiments", "configs", "reports", "analysis", "scripts", "logs", "checkpoints"}
    suffixes = {".json", ".jsonl", ".yaml", ".yml", ".md", ".py", ".ps1", ".log", ".txt"}
    # Scan actual evidence, not source snapshots duplicated inside each run.
    for folder in sorted(eligible_roots):
        for p in sorted((ROOT / folder).rglob("*")):
            if not p.is_file() or p.suffix not in suffixes:
                continue
            rel = p.relative_to(ROOT)
            if family in rel.parts or "source_snapshot" in rel.parts or "frozen_source" in rel.parts:
                continue
            if folder == "checkpoints" and p.name != "rrl_checkpoint.json":
                continue
            accept(rel.as_posix(), p.read_bytes(), "checkout")
    for tag, relative, expected in RELEASES:
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Published archive SHA256 differs: {relative}")
        count, changed, missing, run_names = 0, 0, 0, set()
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt ZIP: {relative}")
            for name in archive.namelist():
                parts = PurePosixPath(name).parts
                if ".." in parts:
                    raise ValueError("Unsafe archive path")
                if parts and parts[0] == "reasoning-rl-lab":
                    parts = parts[1:]
                if not parts or parts[0] not in eligible_roots or any(x in parts for x in ("source_snapshot", "frozen_source")):
                    continue
                rel = PurePosixPath(*parts)
                if rel.suffix not in suffixes or name.endswith("/"):
                    continue
                if parts[0] == "checkpoints" and rel.name != "rrl_checkpoint.json":
                    continue
                content = archive.read(name)
                count += 1
                if parts[0] == "runs" and len(parts) > 2:
                    run_names.add(parts[1])
                local = ROOT.joinpath(*parts)
                digest = hashlib.sha256(content).hexdigest()
                same = local.is_file() and sha256_file(local) == digest
                if not same:
                    missing += int(not local.exists())
                    changed += int(local.exists())
                    dest = exp / "history_restored" / tag / Path(*parts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
                    assert sha256_file(dest) == digest
                    restored.append({"archive": relative, "member": name,
                        "restored": dest.relative_to(ROOT).as_posix(), "sha256": digest,
                        "reason": "missing" if not local.exists() else "archived_version_differs_preserved_separately"})
                accept(rel.as_posix(), content, "released:" + tag)
        archives.append({"tag": tag, "path": relative, "sha256": actual, "bytes": path.stat().st_size,
            "verified_against_published_receipt": True, "auditable_members": count,
            "missing_restored": missing, "different_versions_restored": changed, "runs": sorted(run_names)})

    exposures = defaultdict(list)
    source_index = []
    for rel, content, origin, digest in sources:
        text = content.decode("utf-8-sig", errors="replace")
        ids = sorted(set(ID_RE.findall(text)) & valid)
        if not ids and not (rel.startswith("runs/") and rel.endswith("/config.json")):
            continue
        if "predictions" in rel or "/probes/" in rel or rel.endswith("probes.jsonl"):
            kind = "actual_prediction_or_probe"
        elif "supervision" in rel or "rollout" in rel or "selections" in rel:
            kind = "actual_training_or_selection_evidence"
        elif "data_usage" in rel or "comparison_contract" in rel or "rrl_checkpoint" in rel:
            kind = "nominal_or_recorded_training_pool"
        elif "development_ids" in rel or "evaluation_ids" in rel:
            kind = "registered_development"
        else:
            kind = "conservative_historical_reference"
        src = len(source_index)
        source_index.append({"index": src, "path": rel, "origin": origin, "sha256": digest, "kind": kind, "ids": ids})
        for identifier in ids:
            exposures[identifier].append({"source_index": src, "kind": kind})
        # Reconstruct all nominal SFT/GRPO pools for historical run configs, including failed starts.
        if rel.startswith("runs/") and rel.endswith("/config.json"):
            cfg = json.loads(text)
            train = [r for r in rows if r["split"] == "train"]
            random.Random(cfg.get("data_seed", 0)).shuffle(train)
            nominal = [(r["id"], "nominal_sft_config_pool") for r in train[:cfg.get("sft", {}).get("pool_size", 0)]]
            opts = cfg.get("grpo", {})
            if not opts.get("candidate_pool_path"):
                offset = opts.get("pool_offset", 0)
                nominal.extend((r["id"], "nominal_grpo_config_pool") for r in train[offset:offset+opts.get("pool_size", 0)])
            for identifier, label in nominal:
                exposures[identifier].append({"source_index": src, "kind": label})
    # Audit the full original training sequence independently of explicit IDs in raw evidence.
    original_train = [r for r in rows if r["split"] == "train"]
    random.Random(20261105).shuffle(original_train)
    nominal_sft = [r["id"] for r in original_train[:128]]
    blocked = set(exposures) | set(nominal_sft) | set(FLAGS) | {r["id"] for r in rows if r["split"] == "validation"}
    original_pairs = jsonl_read(ROOT / "data/audit/near_pairs.jsonl")
    uf = UnionFind(len(rows))
    positions = {r["id"]: i for i, r in enumerate(rows)}
    for p in original_pairs:
        if p["a"] in positions and p["b"] in positions:
            uf.union(positions[p["a"]], positions[p["b"]])
    # Add a train-only refit as a conservative check without reopening holdout questions.
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, dtype=np.float32).fit_transform(
        [normalize_question(r["question"]) for r in rows])
    distances, indices = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(matrix).radius_neighbors(
        matrix, radius=0.10, return_distance=True)
    new_pairs = []
    for i, (ds, js) in enumerate(zip(distances, indices, strict=True)):
        for d, j in zip(ds, js, strict=True):
            j = int(j)
            if j > i:
                uf.union(i, j)
                new_pairs.append({"a": rows[i]["id"], "b": rows[j]["id"], "similarity": 1-float(d)})
    hashes = defaultdict(list)
    for i, r in enumerate(rows):
        assert r["question_sha256"] == question_hash(r["question"])
        hashes[r["question_sha256"]].append(i)
    exact = []
    for indices in hashes.values():
        for j in indices[1:]:
            uf.union(indices[0], j)
            exact.append([rows[indices[0]]["id"], rows[j]["id"]])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[uf.find(i)].append(r["id"])
    clusters, exclusion = [], []
    for ids in sorted(groups.values(), key=lambda x: min(x)):
        ids = sorted(ids)
        hits = sorted(blocked & set(ids))
        cluster = {"cluster_id": min(ids), "ids": ids, "historical_or_flagged_hits": hits, "eligible": not bool(hits)}
        clusters.append(cluster)
        if hits:
            exclusion.extend({"id": i, "cluster_id": min(ids), "blocked_by": hits} for i in ids)
    ledger = []
    for r in rows:
        ledger.append({"id": r["id"], "question_sha256": r["question_sha256"],
            "source": r["source"], "official_split": r["official_split"], "original_project_split": r["split"],
            "events": exposures.get(r["id"], []), "nominal_sft128": r["id"] in nominal_sft,
            "known_questionable": r["id"] in FLAGS})
    jsonl_write(exp / "history_sources.jsonl", source_index)
    jsonl_write(exp / "history_exposure_ledger.jsonl", ledger)
    jsonl_write(exp / "history_restoration.jsonl", restored)
    jsonl_write(exp / "clusters.jsonl", clusters)
    jsonl_write(exp / "near_pairs_train_only.jsonl", new_pairs)
    jsonl_write(exp / "historical_exclusions.jsonl", exclusion)
    eligible = [c for c in clusters if c["eligible"]]
    random.Random(20261114).shuffle(eligible)
    json_write(exp / "candidate_order.json", {"seed": 20261114,
        "ordered_representative_ids": [c["ids"][0] for c in eligible], "ordered_cluster_ids": [c["cluster_id"] for c in eligible],
        "rule": "First256 passing blind-to-model Agent quality review. Full candidate order registered before any new probe.",
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json")})
    report = {"audited_utc": now(), "archive_checks": archives, "distinct_source_versions": len(sources),
        "historical_sources_with_train_ids": len(source_index), "restored_versions": len(restored),
        "historically_referenced_or_configured_ids": len(exposures), "nominal_sft128_ids": nominal_sft,
        "original_validation_entirely_excluded": 512, "flagged_ids": FLAGS,
        "original_near_pairs_sha256": sha256_file(ROOT / "data/audit/near_pairs.jsonl"),
        "train_only_near_pairs": len(new_pairs), "normalized_exact_pairs": exact,
        "cluster_count": len(clusters), "blocked_ids_including_cluster_closure": len(exclusion),
        "eligible_clusters": len(eligible), "eligible_rows": sum(len(c["ids"]) for c in eligible),
        "scope": "All present historical runs/configs/experiments/reports/logs/scripts and four SHA256-verified released source/evidence archives. Missing/different historical evidence restored separately. Conservative references excluded even if no model exposure is proven. This establishes project-record eligibility, not public-model pretraining absence.",
        "holdout_content_read": False, "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json")}
    json_write(exp / "history_audit.json", report)
    print(json.dumps({k: report[k] for k in ("restored_versions", "historically_referenced_or_configured_ids", "eligible_clusters", "eligible_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", choices=["register", "audit"], required=True)
    args = parser.parse_args()
    {"register": register, "audit": audit}[args.stage](args.family)
