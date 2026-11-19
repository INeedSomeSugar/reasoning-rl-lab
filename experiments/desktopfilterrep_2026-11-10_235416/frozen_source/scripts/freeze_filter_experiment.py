"""Freeze actual retained IDs, paid probe, executable source, six configs and endpoints."""
from collections import Counter
import copy
import json
import shutil
import yaml
from reasoning_rl.runtime import ROOT, json_write, jsonl_read, sha256_file, checkpoint_hash, code_hash
from reasoning_rl.prefiltering import raw_success_filter, charge_probe
from reasoning_rl.modeling import model_spec, PROMPT_VERSION
from reasoning_rl.scoring import SCORER_VERSION
from filter_common import EXP, FAMILY, START, START_HASH, read, utc_now


def main():
    if (EXP / "protocol_frozen.json").exists():
        raise FileExistsError("Never overwrite an experimental freeze")
    cap = read(EXP / "capacity_frozen.json")
    ids = read(EXP / "candidate_pool.json")["ids"]
    assert sha256_file(EXP / "candidate_pool.json") == cap["candidate_pool_sha256"]
    assert checkpoint_hash(ROOT / START) == START_HASH
    raw_path = ROOT / "runs" / (FAMILY+"-probe") / "predictions.jsonl"
    predictions = jsonl_read(raw_path)
    kept, successes = raw_success_filter(ids, predictions)
    if not kept:
        json_write(EXP / "empty_filter_abort.json", {"all_raw_outputs_preserved": True, "no_main_comparison": True})
        raise RuntimeError("Fixed K4 filter retained no items; do not pick another seed")
    cost = next(r for r in jsonl_read(EXP / "process_costs.jsonl") if r["phase"] == "probe")
    assert cost["exit_code"] == 0
    manifest = {"candidate_pool_sha256": sha256_file(EXP / "candidate_pool.json"),
        "s2_checkpoint_sha256": START_HASH, "probe_seed": read(EXP / "calibration_preregistration.json")["probe_seed"],
        "k": 4, "retained_ids": kept, "raw_success_counts": successes,
        "raw_success_histogram": dict(sorted(Counter(successes.values()).items())),
        "rule": "RAW integer success count in {1,2,3}; uniform retained subset, no dynamic filtering",
        "predictions_path": str(raw_path.relative_to(ROOT)), "predictions_sha256": sha256_file(raw_path),
        "probe_process_wall_seconds": cost["process_wall_seconds"],
        "cost_accounting": charge_probe(cost["process_wall_seconds"]),
        "reuse_scope": "Same fixed S2 probe seed and actual subset shared across all three training seeds; full one-time process cost debited to each F in primary comparison"}
    json_write(EXP / "prefilter_manifest.json", manifest)
    base = yaml.safe_load((ROOT / "configs" / f"{FAMILY}_calibration.yaml").read_text())
    inherited = yaml.safe_load((ROOT / "configs/desktoprepair_20261106_152712_grpo_s13.yaml").read_text())
    assert base["generation"] == inherited["generation"]
    for key in ("learning_rate", "num_generations", "gradient_accumulation_steps", "loss_type", "scale_rewards", "beta", "epsilon"):
        assert base["grpo"][key] == inherited["grpo"][key]
    base["stage"] = "prefilter_uniform_frozen_development"
    base["grpo"].update(candidate_pool_path=f"experiments/{FAMILY}/candidate_pool.json",
        pool_size=len(ids), pool_offset=0,
        steps=100000, max_wall_seconds=cap["selected"]["train_allocation_seconds"],
        max_accounted_tokens=100000000, standalone_endpoints_only=True, final_validation_reserve_seconds=0,
        endpoint_eval_reserve_seconds=cap["evaluation_reserve_seconds"],
        common_end_to_end_cap_seconds=cap["end_to_end_common_cap_seconds"],
        sampling_mechanism="uniform_frozen_pool_v1", audit_loss_signal=True, audit_optimizer_deltas=True,
        save_optimizer_state=True)
    configs = {}
    for seed in (13, 29, 47):
        paired = []
        for arm in ("U", "F"):
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["grpo"]["experiment_arm"] = arm
            if arm == "F":
                config["grpo"]["prefilter_manifest_path"] = f"experiments/{FAMILY}/prefilter_manifest.json"
            path = ROOT / "configs" / f"{FAMILY}_{arm}_s{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
            configs[f"{arm}-s{seed}"] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            paired.append(config)
        u, f = copy.deepcopy(paired)
        for config in (u, f):
            config["grpo"].pop("experiment_arm")
            config["grpo"].pop("prefilter_manifest_path", None)
        assert u == f, "U/F may differ only in prefilter membership/debit metadata"
    files = {}
    for folder in ("src", "configs", "scripts", "tests", "requirements"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".json", ".ps1", ".txt"):
                relative = path.relative_to(ROOT)
                files[relative.as_posix()] = sha256_file(path)
                dest = EXP / "frozen_source" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
    files["pyproject.toml"] = sha256_file(ROOT / "pyproject.toml")
    shutil.copyfile(ROOT / "pyproject.toml", EXP / "frozen_source/pyproject.toml")
    json_write(EXP / "frozen_source_manifest.json", files)
    protocol = {"family": FAMILY, "status": "frozen_development_only", "frozen_utc": utc_now(),
        "common_start": START, "common_start_sha256": START_HASH,
        "model_spec": model_spec(base["model"]), "prompt_version": PROMPT_VERSION, "scorer_version": SCORER_VERSION,
        "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
        "candidate_pool_path": f"experiments/{FAMILY}/candidate_pool.json",
        "candidate_pool_sha256": sha256_file(EXP / "candidate_pool.json"), "candidate_pool_ids": ids,
        "prefilter_manifest_sha256": sha256_file(EXP / "prefilter_manifest.json"), "retained_ids": kept,
        "development_ids_sha256": sha256_file(EXP / "development_ids.json"),
        "development_scope": "Previously used confirmation128 reused as DEVELOPMENT; no independent test claim or subsequent tuning",
        "seeds": [13, 29, 47], "order": [[13,"U"],[13,"F"],[29,"F"],[29,"U"],[47,"U"],[47,"F"]],
        "configs": configs, "code_sha256": code_hash(),
        "frozen_source_manifest_sha256": sha256_file(EXP / "frozen_source_manifest.json"),
        "budget": cap, "probe_cost": manifest["cost_accounting"],
        "only_intervention": "Full fixed train candidate pool uniform versus fixed-S2 raw-K4 mixed-success subset uniform. Shared LR/model/reward/optimizer/group size/temperature and budget. No dynamic resampling.",
        "evaluation": "All six checkpoints fixed before any main endpoint score. Independent fresh-process reload, same frozen128 IDs, greedy, max512, no training after evaluation.",
        "primary_measures": ["paired F-U accuracy difference at fixed end-to-end cap", "effective nonzero-advantage unmasked completions / charged end-to-end second", "nonzero-gradient steps / charged end-to-end second"],
        "cost_scope": "Training process includes startup/loading/cache verification/sampling/optimization/saving; independent evaluation process includes its startup/loading/generation/saving. F additionally debited full probe process. Nonoverlapping elapsed totals, tokens separate.",
        "stop_rule": "Complete current atomic rollout/update, save then stop; report natural-boundary overshoot and unused evaluation reserve; outer endpoint common cap enforced by parent.",
        "no_best_seed_selection": True, "no_formal_holdout": True,
        "limitations": ["Single fixed probe seed and candidate pool", "Only three training seeds", "Already-used development128", "Exact bitwise reproducibility across GPU kernels not claimed"]}
    json_write(EXP / "protocol_frozen.json", protocol)
    print(json.dumps({"candidate_pool": len(ids), "retained": len(kept), "probe_seconds": cost["process_wall_seconds"],
        "common_training_budget": base["grpo"]["max_wall_seconds"], "endpoint_cap": cap["end_to_end_common_cap_seconds"], "frozen": True}))


if __name__ == "__main__":
    main()
