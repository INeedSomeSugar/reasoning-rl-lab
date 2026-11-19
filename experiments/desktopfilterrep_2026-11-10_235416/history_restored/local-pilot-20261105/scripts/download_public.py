"""Resumable, hash-checked public downloads using stdlib only. No credentials or APIs with fees."""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, expected_sha256=None, workers=4):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def digest(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(4*1024*1024), b""):
                h.update(b)
        return h.hexdigest()
    if destination.exists():
        if expected_sha256 and digest(destination) != expected_sha256:
            raise ValueError(f"Existing download hash mismatch: {destination}")
        print(f"Already downloaded: {destination.name}", flush=True)
        return
    # Resolve redirects first, then validate the server's byte-range support.
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=90) as r:
        range_header = r.headers.get("Content-Range")
        if r.status != 206 or range_header is None:
            temp = destination.with_suffix(destination.suffix+".partial")
            with temp.open("wb") as f:
                while block := r.read(4*1024*1024):
                    f.write(block)
            if expected_sha256 and digest(temp) != expected_sha256:
                raise ValueError("Download hash mismatch")
            temp.replace(destination)
            return
        total = int(range_header.rsplit("/", 1)[1])
    chunk_size = 64*1024*1024
    count = (total+chunk_size-1)//chunk_size
    parts = destination.parent / (destination.name+".parts")
    parts.mkdir(exist_ok=True)
    def part(i):
        start, stop = i*chunk_size, min((i+1)*chunk_size, total)-1
        p = parts / f"{i:05d}"
        offset = p.stat().st_size if p.exists() else 0
        if offset == stop-start+1:
            return
        if offset > stop-start+1:
            raise ValueError(f"Oversized part: {p}")
        for attempt in range(6):
            try:
                offset = p.stat().st_size if p.exists() else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={start+offset}-{stop}"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    actual = r.headers.get("Content-Range", "")
                    if r.status != 206 or not actual.startswith(f"bytes {start+offset}-"):
                        raise RuntimeError(f"Server did not honor requested range: {actual}")
                    with p.open("ab") as f:
                        remaining = stop-start+1-offset
                        while remaining:
                            block = r.read(min(4*1024*1024, remaining))
                            if not block:
                                raise OSError("Download stream ended early")
                            f.write(block)
                            remaining -= len(block)
                print(f"{destination.name}: part {i+1}/{count}", flush=True)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 10))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(part, range(count)))
    temp = destination.with_suffix(destination.suffix+".partial")
    with temp.open("wb") as f:
        for i in range(count):
            with (parts/f"{i:05d}").open("rb") as source:
                while block := source.read(4*1024*1024):
                    f.write(block)
    if temp.stat().st_size != total or (expected_sha256 and digest(temp) != expected_sha256):
        raise ValueError("Download size or SHA256 verification failed")
    temp.replace(destination)
    # Preserve part files until the user chooses to clean the project cache.
    print(f"Verified {destination.name}: {total} bytes sha256={digest(destination)}", flush=True)


def model(key, mirror=False):
    spec = json.loads((ROOT/"configs/models.lock.json").read_text(encoding="utf-8"))[key]
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{spec['repo']}/revision/{spec['revision']}?blobs=true", timeout=60) as r:
        metadata = json.load(r)
    if metadata["sha"] != spec["revision"]:
        raise ValueError("Revision resolution mismatch")
    directory = ROOT/"models"/key
    directory.mkdir(exist_ok=True, parents=True)
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if "/" in name or not (Path(name).suffix in {".json", ".jinja", ".safetensors", ".txt", ".model"} or name in {"README.md", "LICENSE"}):
            continue
        target = directory/name
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}?download=true"
        if name.endswith(".safetensors"):
            if mirror:
                url = url.replace("https://huggingface.co/", "https://hf-mirror.com/", 1)
            download(url, target, item.get("lfs", {}).get("sha256"), workers=6)
        elif not target.exists():
            with urllib.request.urlopen(url, timeout=60) as r:
                target.write_bytes(r.read())
        print(f"Model file ready: {name}", flush=True)
    (directory/"rrl_model_source.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (directory/"rrl_download_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (directory/"rrl_download_route.json").write_text(json.dumps({"metadata_origin": "huggingface.co",
        "weight_route": "hf-mirror.com" if mirror else "huggingface.co", "lfs_sha256_checked": True}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "model":
        model(sys.argv[2], mirror="--mirror" in sys.argv)
    elif sys.argv[1] == "torch":
        download("https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",
            ROOT/".cache/wheels/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl",
            "0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff")
