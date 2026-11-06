"""Resume private Release assets and restore only manifest-listed LoRA files.

GitHub credentials stay in memory. Signed download URLs are never logged.
The historical source archive is verified but never extracted into the worktree.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPO = "INeedSomeSugar/reasoning-rl-lab"
TAG = "local-pilot-20261105"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gh_output(*args):
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("GitHub CLI operation failed; check existing login/network")
    return result.stdout


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download_asset(asset, token):
    """Fetch short ranges so a connection reset preserves completed ranges."""
    final = ROOT / "dist" / asset["name"]
    expected = asset["sha256"]
    if final.exists() and sha256(final) == expected:
        print("Reusing verified", final.name, flush=True)
        return
    partial = final.with_name(final.name + ".resumable-partial")
    if not partial.exists() and final.exists():
        shutil.copyfile(final, partial)
    if partial.exists() and partial.stat().st_size > asset["bytes"]:
        raise ValueError("Oversized partial file: " + final.name)
    opener = urllib.request.build_opener(NoRedirect)
    for attempt in range(30):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == asset["bytes"]:
            break
        request = urllib.request.Request(
            asset["api_url"] + "?cache_bust=" + str(time.time_ns()),
            headers={"Authorization": "Bearer " + token, "Accept": "application/octet-stream"},
        )
        try:
            with opener.open(request, timeout=45) as response:
                raise RuntimeError("Expected a signed download redirect")
        except urllib.error.HTTPError as error:
            if error.code != 302:
                raise RuntimeError("GitHub asset redirect failed: HTTP " + str(error.code)) from None
            signed_url = error.headers["Location"]
            error.close()
        parsed = urllib.parse.urlparse(signed_url)
        if parsed.scheme != "https" or not parsed.hostname.endswith(".githubusercontent.com"):
            raise RuntimeError("Unexpected asset download host")
        try:
            while offset < asset["bytes"]:
                stop = min(offset + 2 * 1024 * 1024, asset["bytes"]) - 1
                request = urllib.request.Request(signed_url, headers={"Range": f"bytes={offset}-{stop}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_range = f"bytes {offset}-{stop}/{asset['bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Asset server did not honor the requested byte range")
                    block = response.read()
                if len(block) != stop - offset + 1:
                    raise OSError("Asset byte range ended early")
                with partial.open("ab") as stream:
                    stream.write(block)
                offset = stop + 1
                print(f"{final.name}: {offset}/{asset['bytes']} bytes", flush=True)
            break
        except (OSError, urllib.error.URLError) as error:
            # Avoid printing exception messages containing signed URLs.
            print(f"Retrying {final.name}, attempt {attempt + 1}, {type(error).__name__}", flush=True)
            time.sleep(min(attempt + 1, 5))
    if not partial.exists() or partial.stat().st_size != asset["bytes"] or sha256(partial) != expected:
        raise ValueError("Asset integrity failed; preserved partial: " + final.name)
    if final.exists():
        final.rename(final.with_name(final.name + ".failed-" + str(time.time_ns())))
    partial.rename(final)
    print("Verified", final.name, expected, flush=True)


def verify_and_restore():
    publication = json.loads((ROOT / "environment/publication.json").read_text(encoding="utf-8"))
    assets = publication["release"]["assets"]
    receipts = []
    for asset in assets:
        path = ROOT / "dist" / asset["name"]
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError("Release asset mismatch: " + path.name)
        receipts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    sums = {}
    for line in (ROOT / "dist/SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        digest, filename = line.split(maxsplit=1)
        sums[filename.lstrip("*")] = digest.lower()
    for asset in assets:
        if asset["name"] != "SHA256SUMS.txt" and sums.get(asset["name"]) != asset["sha256"]:
            raise ValueError("SHA256SUMS disagrees with publication manifest")
    manifest = json.loads((ROOT / "migration/CHECKPOINTS_MANIFEST.json").read_text(encoding="utf-8"))
    stage = ROOT / ".cache/desktop-release-staging"
    stage.mkdir(parents=True, exist_ok=True)
    if not stage.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Staging directory resolves outside the project")
    if not (ROOT / "checkpoints").resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Checkpoint directory resolves outside the project")
    archive_path = ROOT / "dist/reasoning-rl-lab-laptop-lora-20261105.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = {}
        for item in archive.infolist():
            name = item.filename
            relative = PurePosixPath(name)
            destination = (stage / name).resolve()
            if ("\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts
                    or not destination.is_relative_to(stage.resolve())
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe ZIP member")
            if name in entries:
                raise ValueError("Duplicate ZIP member")
            entries[name] = item
        archived_manifest = json.loads(archive.read("reasoning-rl-lab/migration/CHECKPOINTS_MANIFEST.json"))
        if archived_manifest != manifest:
            raise ValueError("Archive checkpoint manifest differs from main")
        for relative, expected in manifest["files"].items():
            if not relative.startswith("checkpoints/"):
                raise ValueError("Unexpected restore destination")
            member = "reasoning-rl-lab/" + relative
            destination = (stage / member).resolve()
            if not destination.is_relative_to(stage.resolve()):
                raise ValueError("Unsafe checkpoint path")
            if destination.exists():
                if sha256(destination) != expected:
                    raise ValueError("Staging conflict: " + relative)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entries[member]) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            if sha256(destination) != expected:
                raise ValueError("Staged checkpoint mismatch: " + relative)
        # Check every destination before copying any files. Never overwrite a conflict.
        for relative, expected in manifest["files"].items():
            destination = (ROOT / relative).resolve()
            if not destination.is_relative_to((ROOT / "checkpoints").resolve()):
                raise ValueError("Unsafe checkpoint destination")
            if destination.exists() and sha256(destination) != expected:
                raise ValueError("Existing checkpoint conflict: " + relative)
        for relative, expected in manifest["files"].items():
            destination = ROOT / relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (stage / "reasoning-rl-lab" / relative).open("rb") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            if sha256(destination) != expected:
                raise ValueError("Restored checkpoint mismatch: " + relative)
    result = {"status": "pass", "assets": receipts, "files_restored_and_verified": len(manifest["files"]),
              "adapters": len(manifest["checkpoint_sha256"]), "historical_source_extracted": False,
              "optimizer_resume_available": False, "scope": "Laptop artifact integrity; no AMD execution claim"}
    (ROOT / "environment/desktop_release_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        publication = json.loads((ROOT / "environment/publication.json").read_text(encoding="utf-8"))
        release = json.loads(gh_output("api", f"repos/{REPO}/releases/tags/{TAG}"))
        remote = {a["name"]: a for a in release["assets"]}
        token = gh_output("auth", "token", "--hostname", "github.com").strip()
        try:
            for asset in publication["release"]["assets"]:
                actual = remote[asset["name"]]
                if actual["size"] != asset["bytes"] or actual["digest"] != "sha256:" + asset["sha256"]:
                    raise ValueError("Remote release metadata mismatch")
                download_asset({**asset, "api_url": actual["url"]}, token)
        finally:
            token = None
    verify_and_restore()


if __name__ == "__main__":
    main()
