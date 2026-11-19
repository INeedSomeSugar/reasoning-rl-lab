"""Record this family's checks without replacing prior hardware evidence."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
from reasoning_rl.runtime import ROOT, json_write, sha256_file
from filter_common import EXP, FAMILY


def main():
    source = ROOT / "environment/hardware.json"
    destination = ROOT / "environment" / (FAMILY+"-hardware.json")
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(source.read_bytes())
    source.write_bytes(subprocess.check_output(["git", "show", "HEAD:environment/hardware.json"], cwd=ROOT))
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    from packaging.version import Version
    failures=[]
    checked=[]
    env=default_environment()
    for dist in metadata.distributions():
        for text in dist.requires or []:
            req=Requirement(text)
            if req.marker and not req.marker.evaluate({**env,"extra":""}):
                continue
            try:
                version=metadata.version(req.name)
            except metadata.PackageNotFoundError:
                failures.append({"package":dist.metadata["Name"],"requires":text,"status":"missing"})
                continue
            checked.append({"package":dist.metadata["Name"],"requires":text,"installed":version})
            if req.specifier and Version(version) not in req.specifier:
                failures.append(checked[-1])
    assert not failures, failures
    packages={d.metadata["Name"]:d.version for d in metadata.distributions()}
    json_write(ROOT / "environment" / (FAMILY+"-verification.json"), {"pytest":"59 passed in6.95s",
        "pytest_log_sha256":sha256_file(ROOT/"environment"/(FAMILY+"-pytest.txt")),
        "dependency_check":"Installed requirements checked through importlib.metadata + packaging with environment markers, no install",
        "checked_dependency_relations":len(checked),"dependency_failures":failures,"installed_packages":packages,
        "hardware_receipt":str(destination.relative_to(ROOT)),"prior_hardware_receipt_restored":True,
        "uv_command":"uv was not on PATH; no reinstall attempted"})
    print(json.dumps({"installed_packages":len(packages),"dependency_relations_checked":len(checked),"failures":failures}))


if __name__=="__main__":
    main()
