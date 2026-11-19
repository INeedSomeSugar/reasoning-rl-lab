"""Strict, project-local access to retrospectively isolated original-train development data."""
import hashlib
import json
from pathlib import Path

from .runtime import ROOT, checkpoint_hash, jsonl_read, sha256_file

REGISTRY = Path("data/fresh_development/registry.json")
KIND = "posthoc_isolated_original_train_development_v1"


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Fresh development evidence must stay within the project")
    return path


def read_manifest(relative, expected_sha256=None):
    path = local_path(relative)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError("Fresh development manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != KIND:
        raise ValueError("Not an isolated original-train development manifest")
    if manifest["original_data_manifest_sha256"] != sha256_file(ROOT / "data/manifest.json"):
        raise ValueError("Original data provenance changed")
    ids = manifest["ids"]
    if not ids or len(ids) != len(set(ids)) or any(not i.startswith("gsm8k-train-") for i in ids):
        raise ValueError("Invalid fresh development membership")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != manifest["id_sha256"]:
        raise ValueError("Fresh development ID digest differs")
    return manifest


def reserved_train_ids():
    """Ordinary train/select/develop loaders cannot return any registered fresh items."""
    path = ROOT / REGISTRY
    if not path.exists():
        return set()
    registry = json.loads(path.read_text(encoding="utf-8"))
    result = set()
    for entry in registry["entries"]:
        manifest = read_manifest(entry["manifest_path"], entry["manifest_sha256"])
        result.update(manifest["ids"])
    return result


def authorize_endpoint(manifest_relative, lock_relative, key, adapter):
    """All35 attempted training units must be terminal and all successful weights immutable."""
    lock_path = local_path(lock_relative)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("stage") != "all_training_endpoints_fixed_before_fresh_evaluation":
        raise ValueError("Fresh development is sealed until all training endpoints are fixed")
    expected_keys = set(lock["registered_training_keys"])
    if len(expected_keys) != 35 or set(lock["training_units"]) != expected_keys:
        raise ValueError("Training matrix is incomplete or altered")
    for unit in lock["training_units"].values():
        if unit["status"] not in {"first_success_fixed", "exhausted_failure"}:
            raise ValueError("Training unit has not reached a terminal status")
        if unit["status"] == "first_success_fixed":
            if checkpoint_hash(local_path(unit["checkpoint"])) != unit["checkpoint_sha256"]:
                raise ValueError("A frozen training endpoint changed")
    if set(lock["objects"]) != expected_keys | {"A", "S2"}:
        raise ValueError("Evaluation objects do not match the registered37")
    manifest_path = local_path(manifest_relative)
    if sha256_file(manifest_path) != lock["fresh_development_manifest_sha256"]:
        raise ValueError("Endpoint lock is for a different fresh development set")
    if key not in lock["objects"] or lock["objects"][key]["status"] != "fixed":
        raise ValueError("Requested endpoint is absent from the frozen successful objects")
    obj = lock["objects"][key]
    actual_adapter = local_path(adapter) if adapter else None
    expected_adapter = local_path(obj["checkpoint"]) if obj.get("checkpoint") else None
    if actual_adapter != expected_adapter:
        raise ValueError("Requested adapter does not match its frozen evaluation identity")
    if actual_adapter and checkpoint_hash(actual_adapter) != obj["checkpoint_sha256"]:
        raise ValueError("Frozen evaluation adapter changed")
    for item in lock["public_model_files"]:
        if sha256_file(local_path(item["path"])) != item["sha256"]:
            raise ValueError("Public model/tokenizer bytes changed")
    return lock


def load_fresh_development(manifest_relative, *, intent, lock_relative=None, endpoint_key=None,
                           adapter=None, expected_sha256=None):
    if intent != "evaluate_fresh_development" or not lock_relative or not endpoint_key:
        raise ValueError("Fresh development requires the dedicated frozen-endpoint evaluation entry")
    manifest = read_manifest(manifest_relative, expected_sha256)
    authorize_endpoint(manifest_relative, lock_relative, endpoint_key, adapter)
    path = local_path(manifest["rows_path"])
    if sha256_file(path) != manifest["rows_sha256"]:
        raise ValueError("Fresh development rows changed")
    rows = jsonl_read(path)
    if [r["id"] for r in rows] != manifest["ids"] or len(rows) != manifest["n"]:
        raise ValueError("Fresh development membership/count changed")
    if any(r["official_split"] != "train" or r["split"] != "fresh_development"
           or r["original_project_split"] != "train" for r in rows):
        raise ValueError("Fresh development must originate in the original train partition")
    return rows
