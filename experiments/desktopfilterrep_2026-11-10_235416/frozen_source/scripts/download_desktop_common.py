"""Cache exact uv-resolved PyPI wheels, checking publisher SHA256 before offline install."""
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from download_public import ROOT, download


def wheel_rank(filename):
    parts = filename.removesuffix(".whl").split("-")
    if len(parts) < 5:
        return 0
    py, abi, platform = parts[-3:]
    if platform == "win_amd64":
        if py == "cp312" and abi in {"cp312", "none"}:
            return 100
        if abi == "abi3" and py.startswith("cp") and py[2:].isdigit() and 32 <= int(py[2:]) <= 312:
            return 80
    if platform == "any" and abi == "none" and any(tag in {"py3", "py312"} for tag in py.split(".")):
        return 60
    return 0


def metadata(package, version):
    cache = ROOT / ".cache/pypi-metadata" / f"{package}-{version}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as response:
                result = json.load(response)
            cache.write_text(json.dumps(result), encoding="utf-8")
            return result
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)


def prepare(pin):
    package, version = pin
    meta = metadata(package, version)
    choices = sorted(((wheel_rank(item["filename"]), item) for item in meta["urls"]
                      if item["packagetype"] == "bdist_wheel" and not item.get("yanked")),
                     key=lambda pair: pair[0], reverse=True)
    if not choices or choices[0][0] == 0 or (len(choices) > 1 and choices[0][0] == choices[1][0]):
        raise ValueError(f"Need one compatible CPython 3.12 Windows wheel for {package}")
    item = choices[0][1]
    return {"package": package, "version": version, "filename": item["filename"],
            "url": item["url"], "size": item["size"], "sha256": item["digests"]["sha256"]}


def fetch(record):
    target = ROOT / ".cache/wheels" / record["filename"]
    download(record["url"], target, record["sha256"], workers=2)
    if target.stat().st_size != record["size"]:
        raise ValueError("Wheel size mismatch: " + record["filename"])
    with zipfile.ZipFile(target) as archive:
        if archive.testzip():
            raise ValueError("Wheel CRC mismatch: " + record["filename"])
    record["verified"] = True
    return record


if __name__ == "__main__":
    plan = (ROOT / "logs/desktop-common-resolved-plan.log").read_text(encoding="utf-8-sig")
    pins = re.findall(r"^ \+ ([a-zA-Z0-9_.-]+)==([^\s]+)", plan, re.MULTILINE)
    if len(pins) != 52:
        raise ValueError("Expected the preserved 52-package resolution; inspect a changed plan explicitly")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(prepare, pins))
    receipt = ROOT / "environment/desktop_common_wheels.json"
    receipt.write_text(json.dumps({"source": "pypi.org", "wheels": records}, indent=2)+"\n", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for record in executor.map(fetch, records):
            receipt.write_text(json.dumps({"source": "pypi.org", "wheels": records}, indent=2)+"\n", encoding="utf-8")
    print("All 52 exact resolved wheels passed publisher SHA256, size and ZIP CRC.", flush=True)
