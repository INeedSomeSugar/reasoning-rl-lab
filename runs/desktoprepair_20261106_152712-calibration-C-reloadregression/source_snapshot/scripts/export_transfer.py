"""Export portable source/data/text evidence, never a CUDA virtualenv."""
import datetime
import hashlib
import json
import os
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
allowed_roots = {"src", "configs", "scripts", "tests", "requirements", "docs", "migration", "data", "environment", "reports", "runs", "experiments", "logs"}
allowed_files = {"AGENTS.md", "README.md", "PROJECT_CONTEXT.md", "HANDOFF.md", "pyproject.toml", ".gitignore", ".gitattributes", "LICENSE"}
suffixes = {".py", ".yaml", ".json", ".jsonl", ".txt", ".md", ".toml", ".csv", ".png", ".svg", ".ps1", ".sh", ".html", ".log"}
files = []
candidate_paths = [root/name for name in allowed_files if (root/name).is_file()]
for directory_name in sorted(allowed_roots):
    directory = root/directory_name
    if not directory.exists():
        continue
    for folder, subdirs, names in os.walk(directory):
        subdirs[:] = [d for d in subdirs if d not in {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", "trainer"}]
        candidate_paths.extend(Path(folder)/name for name in names)
for p in candidate_paths:
    relative = p.relative_to(root)
    # os.walk prunes environment/build folders; only approved small artifacts enter the archive.
    if not p.is_file() or any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"} for part in relative.parts):
        continue
    if relative.parts[0] not in allowed_roots and str(relative) not in allowed_files:
        continue
    if p.suffix not in suffixes and p.name not in allowed_files:
        continue
    if p.name == "TRANSFER_MANIFEST.json" or p.stat().st_size > 20*1024*1024:
        continue
    if "trainer" in relative.parts:
        continue
    files.append(p)
manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "excluded": ["virtualenvs", "CUDA dependencies", "caches", "model weights", "checkpoints", "trainer state"],
            "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
manifest_path = root/"migration/TRANSFER_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
destination = root/"dist/reasoning-rl-lab-desktop-transfer-20261105.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for p in files+[manifest_path]:
        archive.write(p, Path("reasoning-rl-lab")/p.relative_to(root))
print(destination)
print(f"{len(files)+1} files; {destination.stat().st_size} bytes")
