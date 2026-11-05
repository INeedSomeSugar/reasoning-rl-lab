"""Project-local paths, hashes, environment evidence and compute accounting."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def bootstrap():
    for key, directory in {
        "HF_HOME": ".cache/huggingface", "HF_DATASETS_CACHE": ".cache/datasets",
        "TORCH_HOME": ".cache/torch", "TRITON_CACHE_DIR": ".cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/inductor", "XDG_CACHE_HOME": ".cache/xdg",
        "MPLCONFIGDIR": ".cache/matplotlib", "UV_CACHE_DIR": ".cache/uv",
    }.items():
        os.environ[key] = str(ROOT / directory)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def jsonl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def jsonl_read(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_append(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_hash(directory):
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.name in {
        "adapter_model.safetensors", "adapter_config.json", "rrl_checkpoint.json"})
    if not any(p.name == "adapter_model.safetensors" for p in paths):
        raise ValueError(f"Missing adapter weights: {directory}")
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(directory)).encode())
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for directory in ("src", "configs"):
        for p in sorted((ROOT / directory).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml", ".json"):
                h.update(p.relative_to(ROOT).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact bitwise reproducibility across GPU kernels is not claimed.


def hardware_info():
    import psutil
    import torch
    versions = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "os": platform.platform(), "python": platform.python_version(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "versions": versions, "torch_cuda": torch.version.cuda, "torch_hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(), "gpus": [],
        "rx9070xt_access": "not established; no remote host or credentials supplied",
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        result["gpus"].append({"index": i, "name": props.name, "total_bytes": total,
                               "free_bytes": free, "compute_capability": list(torch.cuda.get_device_capability(i))})
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
            text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result


class Ledger:
    """Device elapsed time includes device idle intervals; it is not active-kernel time or FLOPs."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self._measure_depth = 0

    def emit(self, **row):
        row = {"time_unix": time.time(), **row}
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    @contextlib.contextmanager
    def measure(self, phase, **metadata):
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            if self._measure_depth == 0:
                torch.cuda.reset_peak_memory_stats()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        self._measure_depth += 1
        counts = {}
        success = False
        try:
            yield counts
            success = True
        finally:
            self._measure_depth -= 1
            if cuda:
                end_event.record()
                torch.cuda.synchronize()
            self.emit(phase=phase, success=success, wall_seconds=time.perf_counter()-start,
                      device_elapsed_seconds=start_event.elapsed_time(end_event)/1000 if cuda else None,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
                      peak_scope="outer_stage_to_date",
                      hardware=torch.cuda.get_device_name() if cuda else "CPU", **metadata, **counts)


def create_run(name, config):
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, exist_ok=False)
    json_write(directory / "config.json", config)
    json_write(directory / "environment.json", hardware_info())
    json_write(directory / "provenance.json", {"code_sha256": code_hash(), "created_unix": time.time()})
    # Preserve executable source at run time, even if later runs repair or extend the project.
    for folder in ("src", "configs", "requirements"):
        for source in (ROOT / folder).rglob("*"):
            if source.is_file() and source.suffix in {".py", ".yaml", ".json", ".txt"}:
                destination = directory / "source_snapshot" / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    shutil.copyfile(ROOT / "pyproject.toml", directory / "source_snapshot/pyproject.toml")
    return directory, Ledger(directory / "costs.jsonl")
