"""Package local LoRA artifacts separately from portable source; no full model/env."""
import datetime
import zipfile

from reasoning_rl.runtime import ROOT, checkpoint_hash, json_write, sha256_file


def main():
    allowed = {"adapter_model.safetensors", "adapter_config.json", "rrl_checkpoint.json"}
    source = ROOT / "checkpoints"
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.name in allowed)
    directories = sorted({p.parent for p in paths if p.name == "adapter_model.safetensors"})
    if not directories:
        raise FileNotFoundError("No actual local adapters to package")
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "Actual laptop LoRA weights, including engineering/recovery artifacts; not AMD training evidence",
                "excluded": ["public model weights", "virtual environments", "caches", "optimizer state"],
                "checkpoint_sha256": {p.relative_to(ROOT).as_posix(): checkpoint_hash(p) for p in directories},
                "files": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}}
    manifest_path = ROOT / "migration/CHECKPOINTS_MANIFEST.json"
    json_write(manifest_path, manifest)
    destination = ROOT / "dist/reasoning-rl-lab-laptop-lora-20261105.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths + [manifest_path]:
            archive.write(path, "reasoning-rl-lab/" + path.relative_to(ROOT).as_posix())
    receipt = {"archive": destination.relative_to(ROOT).as_posix(), "sha256": sha256_file(destination),
               "bytes": destination.stat().st_size, "checkpoint_count": len(directories),
               "public_model_included": False, "optimizer_state_included": False}
    json_write(ROOT / "environment/checkpoint_archive.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
