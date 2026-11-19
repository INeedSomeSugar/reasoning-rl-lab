"""Verify a portable snapshot before making platform-specific changes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/"migration/TRANSFER_MANIFEST.json").read_text(encoding="utf-8"))
bad = []
for name, expected in manifest["files"].items():
    path = (root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        bad.append(name)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        bad.append(name)
if bad:
    raise SystemExit("Transfer verification FAILED: "+", ".join(bad))
print(f"Verified {len(manifest['files'])} files; snapshot {manifest['created_utc']}")
