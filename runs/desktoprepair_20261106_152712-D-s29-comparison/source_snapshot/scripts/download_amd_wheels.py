"""Inspect/download pinned official AMD Windows wheels with resumable ranges."""
import argparse
import hashlib
import html.parser
import json
import time
import zipfile
import urllib.parse
import urllib.request
from pathlib import Path
from download_public import ROOT, download

INDEX = "https://stable.repo.amd.com/rocm/whl-next/"
PINS = {
    "torch": "2.13.0+rocm10.0.0",
    "amd-torch-device-gfx12-0": "2.13.0+rocm10.0.0",
    "amd-torch-device-gfx1201": "2.13.0+rocm10.0.0",
    "rocm-sdk-core": "10.0.0",
    "rocm-sdk-libraries": "10.0.0",
    "rocm-sdk-device-gfx1201": "10.0.0",
}


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")


def inspect():
    receipt = []
    for package, version in PINS.items():
        page = INDEX + package + "/"
        cache = ROOT / ".cache/amd-index" / (package + ".html")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            body = cache.read_text(encoding="utf-8")
        else:
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(page, timeout=45) as response:
                        body = response.read().decode("utf-8")
                    cache.write_text(body, encoding="utf-8")
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        parser = Links()
        parser.feed(body)
        candidates = []
        for link in parser.links:
            url = urllib.parse.urljoin(page, link)
            parsed = urllib.parse.urlparse(url)
            name = urllib.parse.unquote(Path(parsed.path).name)
            parts = name.removesuffix(".whl").split("-")
            if (len(parts) == 5 and parts[0].replace("_", "-") == package and parts[1] == version
                    and parts[2] in {"cp312", "py3"} and parts[4] in {"win_amd64", "any"}):
                digest = urllib.parse.parse_qs(parsed.fragment).get("sha256", [None])[0]
                candidates.append({"package": package, "version": version, "filename": name,
                                   "url": urllib.parse.urldefrag(url)[0], "sha256": digest})
        if len(candidates) != 1:
            raise ValueError(f"Expected one exact Windows wheel for {package}, got {len(candidates)}")
        receipt.append(candidates[0])
        print("Found pinned wheel", candidates[0]["filename"], flush=True)
    path = ROOT / "environment/desktop_amd_wheels.json"
    path.write_text(json.dumps({"index": INDEX, "wheels": receipt}, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    records = inspect()
    if args.download:
        for record in records:
            target = ROOT / ".cache/wheels" / record["filename"]
            download(record["url"], target, record["sha256"])
            with zipfile.ZipFile(target) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise ValueError("Wheel CRC failure: " + record["filename"])
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            record.update(local_sha256=digest, zip_crc_verified=True,
                          vendor_sha256_published=record["sha256"] is not None)
            (ROOT / "environment/desktop_amd_wheels.json").write_text(
                json.dumps({"index": INDEX, "wheels": records}, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(records, indent=2))
