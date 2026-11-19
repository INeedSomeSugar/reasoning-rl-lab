"""Verify the separate LoRA archive or restored adapters without loading a model."""
import argparse
import hashlib
import json
import zipfile

from reasoning_rl.runtime import ROOT, json_write, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive")
    args = parser.parse_args()
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            manifest = json.loads(archive.read("reasoning-rl-lab/migration/CHECKPOINTS_MANIFEST.json"))
            checked = 0
            for relative, expected in manifest["files"].items():
                digest = hashlib.sha256()
                with archive.open("reasoning-rl-lab/" + relative) as stream:
                    for chunk in iter(lambda: stream.read(4*1024*1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError("Archive checksum mismatch: " + relative)
                checked += 1
        result = {"status": "pass", "files_checked": checked,
                  "checkpoint_count": len(manifest["checkpoint_sha256"]),
                  "archive_sha256": sha256_file(args.archive),
                  "scope": "Portable adapter byte integrity; no model quality or AMD execution claim"}
        json_write(ROOT / "environment/checkpoint_archive_verification.json", result)
    else:
        from reasoning_rl.runtime import checkpoint_hash
        manifest = json.loads((ROOT / "migration/CHECKPOINTS_MANIFEST.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != expected:
                raise ValueError("Missing or changed adapter file: " + relative)
        for relative, expected in manifest["checkpoint_sha256"].items():
            if checkpoint_hash(ROOT / relative) != expected:
                raise ValueError("Restored checkpoint hash mismatch: " + relative)
        result = {"status": "pass", "files_checked": len(manifest["files"]),
                  "checkpoint_count": len(manifest["checkpoint_sha256"]),
                  "scope": "Restored laptop adapter bytes; no optimizer or AMD execution claim"}
    print(result)


if __name__ == "__main__":
    main()
