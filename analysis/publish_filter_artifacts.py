"""Package the completed study without model execution or network operations.

Run prepare before the results commit, then archive that exact commit. Publication
receipts are separate from the immutable, pre-publication experiment closeout.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import zipfile

from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, sha256_file

FAMILY = "desktopfilter_20261110"
EXP = ROOT / "experiments" / FAMILY
MANIFEST = ROOT / "migration" / f"{FAMILY}_CHECKPOINTS_MANIFEST.json"
PREFLIGHT = ROOT / "environment" / f"{FAMILY}-upload-preflight.json"
OUT = ROOT / "dist" / f"{FAMILY}-publication"
TAG = "fixed-prefilter-dev-20261110"
CORE = {"adapter_model.safetensors", "adapter_config.json", "rrl_checkpoint.json"}
FOLDERS = {"src", "scripts", "tests", "configs", "requirements", "docs", "reports",
           "runs", "logs", "experiments", "environment", "migration", "analysis", "data"}
ROOT_FILES = {".gitignore", ".gitattributes", "AGENTS.md", "README.md", "HANDOFF.md",
              "PROJECT_CONTEXT.md", "pyproject.toml"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prepare():
    if MANIFEST.exists() or PREFLIGHT.exists():
        raise FileExistsError("Preserve the existing publication preflight and manifest")
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_filter_controlled import verify

    plan = read(EXP / "protocol_frozen.json")
    verify(plan)
    final = read(EXP / "session_final.json")
    assert final["status"] == "completed_three_paired_seed_development_and_handoff"
    assert read(ROOT / "reports/comparisons" / FAMILY / "evidence_audit.json")["all_checks_passed"]
    closed = EXP / "artifact_manifest.json"
    assert sha256_file(closed) == final["handoff_manifest_sha256"]
    entries = read(closed)["files"]
    assert len(entries) == final["manifest_file_count"]
    mismatches = []
    for item in entries:
        path = ROOT / item["path"]
        assert path.resolve().is_relative_to(ROOT.resolve())
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]):
            mismatches.append(item["path"])
    assert not mismatches, mismatches

    endpoints = read(EXP / "endpoint_costs.json")
    assert set(endpoints) == {f"{a}-s{s}" for a in "UF" for s in (13, 29, 47)}
    selected = [("paired_training_endpoint", "checkpoints/" + row["training_run"],
                 label, row["checkpoint_sha256"]) for label, row in endpoints.items()]
    selected += [
        ("calibration_only", f"checkpoints/{FAMILY}-calibration-train", "calibration", None),
        ("forward_backward_save_reload_diagnostic_only", f"checkpoints/{FAMILY}-preflight", "preflight", None),
        ("common_S2_start", plan["common_start"], "S2", plan["common_start_sha256"]),
    ]
    records = []
    for role, relative, label, expected in selected:
        directory = ROOT / relative
        digest = checkpoint_hash(directory)
        if expected:
            assert digest == expected, label
        files = []
        for path in sorted(directory.iterdir()):
            if path.name in CORE | {"optimizer_diagnostic_state.pt"}:
                files.append({"path": path.relative_to(ROOT).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        assert CORE <= {Path(f["path"]).name for f in files}
        records.append({"label": label, "role": role, "checkpoint": relative,
                        "checkpoint_sha256": digest, "files": files})
    counts = dict(Counter(r["role"] for r in records))
    assert len(records) == 9 and counts["paired_training_endpoint"] == 6
    json_write(MANIFEST, {
        "family": FAMILY, "created_utc": now(), "checkpoint_count": len(records),
        "count_by_role": counts, "checkpoints": records, "model_spec": plan["model_spec"],
        "checkpoint_hash_definition": "runtime.checkpoint_hash: sorted core filenames and SHA256 bytes",
        "optimizer_exact_resume_supported": False,
        "optimizer_note": "Seven Adam/RNG diagnostic states are included as opaque bytes; sampler/dataloader exact automatic resume is not implemented.",
        "excluded": ["public base-model weights", "virtual environment", "caches", "tokenizer copies (use pinned base tokenizer)"],
    })
    json_write(PREFLIGHT, {
        "family": FAMILY, "verified_utc": now(), "original_manifest_sha256": sha256_file(closed),
        "original_manifest_file_count_verified": len(entries), "original_manifest_mismatches": [],
        "frozen_source_data_common_start_verified": True, "checkpoint_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_sha256": sha256_file(MANIFEST), "release_tag": TAG,
        "publication_does_not_change_experiment_runtime_or_predictions": True,
        "no_model_execution": True, "holdout_read_for_evaluation": False,
    })
    print(json.dumps({"verified_original_files": len(entries), "checkpoint_count": len(records),
                      "roles": counts}, ensure_ascii=False))


def included(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts
    if len(path.parts) > 1 and path.parts[:2] == ("data", "holdout"):
        return False
    return ((path.parts[0] in FOLDERS or name in ROOT_FILES)
            and path.suffix.lower() not in {".safetensors", ".pt", ".bin", ".zip", ".exe", ".dll"}
            and not path.name.startswith(".env"))


def verify_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == set(expected)
        assert len(archive.namelist()) == len(expected)
        for name, digest in expected.items():
            hasher = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    hasher.update(block)
            assert hasher.hexdigest() == digest, name


def archive(commit):
    commit = subprocess.check_output(["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True).strip()
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    subprocess.run(["git", "diff", "--quiet", commit, "--"], cwd=ROOT, check=True)
    manifest = read(MANIFEST)
    assert sha256_file(MANIFEST) == read(PREFLIGHT)["checkpoint_manifest_sha256"]
    if OUT.exists():
        raise FileExistsError("Preserve the existing archives")
    OUT.mkdir(parents=True)
    weights = OUT / f"{FAMILY}-lora-and-optimizer-diagnostics.zip"
    expected = {}
    with zipfile.ZipFile(weights, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for checkpoint in manifest["checkpoints"]:
            assert checkpoint_hash(ROOT / checkpoint["checkpoint"]) == checkpoint["checkpoint_sha256"]
            for item in checkpoint["files"]:
                path = ROOT / item["path"]
                assert path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"]
                name = "reasoning-rl-lab/" + item["path"]
                output.write(path, name)
                expected[name] = item["sha256"]
        name = "reasoning-rl-lab/" + MANIFEST.relative_to(ROOT).as_posix()
        output.write(MANIFEST, name)
        expected[name] = sha256_file(MANIFEST)
    verify_zip(weights, expected)

    source = OUT / f"{FAMILY}-source-and-evidence.zip"
    expected = {}
    child = subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=child.stdout, mode="r|") as tree:
        with zipfile.ZipFile(source, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for member in tree:
                if not member.isfile() or not included(member.name):
                    continue
                payload = tree.extractfile(member).read()
                name = "reasoning-rl-lab/" + member.name
                output.writestr(name, payload)
                expected[name] = hashlib.sha256(payload).hexdigest()
    assert child.wait() == 0
    verify_zip(source, expected)
    receipt = {
        "family": FAMILY, "created_utc": now(), "source_code_commit": commit,
        "release_tag": TAG, "publication_status": "archives_verified_locally_ready_for_authorized_upload",
        "archives": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in (source, weights)],
        "source_and_evidence_file_count": len(expected),
        "source_file_hash_mapping_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_selection": "Exact git-archive commit bytes in declared project folders/root files, excluding formal holdout contents and binary weights; this is a portable subset of the commit.",
        "checkpoint_count": manifest["checkpoint_count"], "count_by_role": manifest["count_by_role"],
        "checkpoint_manifest": MANIFEST.relative_to(ROOT).as_posix(), "checkpoint_manifest_sha256": sha256_file(MANIFEST),
        "optimizer_exact_resume_supported": False, "public_model_or_environment_included": False,
        "all_archive_members_sha256_verified": True,
        "original_closeout_receipts_preserved": True,
        "note": "This receipt follows archive construction, avoiding a self-hash cycle. Publication time is separate from the four-hour experimental runtime.",
    }
    receipt_path = OUT / "archive_receipt.json"
    json_write(receipt_path, receipt)
    lines = [f"{sha256_file(p)}  {p.name}" for p in (source, weights, receipt_path)]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"source_code_commit": commit, "source_files": len(expected),
                      "archives": receipt["archives"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "archive"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        if not args.commit:
            parser.error("archive requires --commit")
        archive(args.commit)
