"""Pinned official data, grouped train/validation split and sealed holdouts."""
import hashlib
import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path
from .runtime import ROOT, json_write, jsonl_read, jsonl_write, sha256_file
from .scoring import canonical_number

GSM_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
NEAR_THRESHOLD = 0.90


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "ReasoningRL-Lab/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        temp = destination.with_suffix(destination.suffix + ".partial")
        temp.write_bytes(content)
        temp.replace(destination)
    return destination


def normalize_question(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))


def question_hash(text):
    return hashlib.sha256(normalize_question(text).encode()).hexdigest()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[b] = a


def prepare(seed=20261105, validation_size=512, threshold=NEAR_THRESHOLD):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    raw = ROOT / "data/raw"
    lock_path = raw / "sources.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request("https://api.github.com/repos/arkilpatel/SVAMP/commits/main",
                                         headers={"User-Agent": "ReasoningRL-Lab/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            svamp_revision = json.load(response)["sha"]
        lock = {"gsm8k": {"repo": "openai/grade-school-math", "revision": GSM_REVISION, "license": "MIT"},
                "svamp": {"repo": "arkilpatel/SVAMP", "revision": svamp_revision, "license": "MIT"}}
        json_write(lock_path, lock)
    sources = {}
    for name, relative in [("gsm_train", "grade_school_math/data/train.jsonl"),
                           ("gsm_test", "grade_school_math/data/test.jsonl"), ("gsm_license", "LICENSE")]:
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{lock['gsm8k']['revision']}/{relative}"
        sources[name] = fetch(url, raw / (name + (".jsonl" if name != "gsm_license" else ".txt")))
    for name, relative in [("svamp", "SVAMP.json"), ("svamp_license", "LICENSE")]:
        url = f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{lock['svamp']['revision']}/{relative}"
        sources[name] = fetch(url, raw / (name + (".json" if name == "svamp" else ".txt")))

    def gsm_rows(path, split):
        result = []
        for i, row in enumerate(jsonl_read(path)):
            answer = row["answer"].rsplit("####", 1)[1].strip()
            assert canonical_number(answer) is not None
            result.append({"id": f"gsm8k-{split}-{i:05d}", "question": row["question"],
                           "answer": answer, "solution": row["answer"], "source": "gsm8k",
                           "official_split": split, "question_sha256": question_hash(row["question"])})
        return result

    train = gsm_rows(sources["gsm_train"], "train")
    test = gsm_rows(sources["gsm_test"], "test")
    external = []
    for i, row in enumerate(json.loads(sources["svamp"].read_text(encoding="utf-8"))):
        question = (row["Body"].strip() + " " + row["Question"].strip()).strip()
        answer = str(row["Answer"])
        assert canonical_number(answer) is not None
        external.append({"id": f"svamp-{i:05d}", "question": question, "answer": answer,
                         "source": "svamp", "official_split": "external",
                         "question_sha256": question_hash(question), "problem_type": row.get("Type")})
    all_rows = train + test + external
    # Questions only: no holdout solutions, predictions or scores enter this hygiene step.
    matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, dtype=np.float32).fit_transform(
        [normalize_question(row["question"]) for row in all_rows])
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(matrix)
    distances, indices = neighbors.radius_neighbors(matrix, radius=1-threshold, return_distance=True)
    uf, quarantined, pairs = UnionFind(len(train)), set(), []
    ntrain = len(train)
    for i, (dists, js) in enumerate(zip(distances, indices, strict=True)):
        for distance, j in zip(dists, js, strict=True):
            j = int(j)
            if j <= i:
                continue
            pairs.append({"a": all_rows[i]["id"], "b": all_rows[j]["id"],
                          "similarity": round(1-float(distance), 6),
                          "exact": all_rows[i]["question_sha256"] == all_rows[j]["question_sha256"]})
            if i < ntrain and j < ntrain:
                uf.union(i, j)
            elif i < ntrain <= j:
                quarantined.add(i)
    # Quarantine entire related train component, then assign components as indivisible units.
    roots_to_drop = {uf.find(i) for i in quarantined}
    groups = {}
    for i in range(ntrain):
        root = uf.find(i)
        if root not in roots_to_drop:
            groups.setdefault(root, []).append(i)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    validation_indices = set()
    for group in shuffled:
        if len(validation_indices) < validation_size:
            validation_indices.update(group)
    kept = {i for group in shuffled for i in group}
    train_out = [{**row, "split": "train"} for i, row in enumerate(train) if i in kept-validation_indices]
    validation_out = [{**row, "split": "validation"} for i, row in enumerate(train) if i in validation_indices]
    outputs = {
        "train": ("data/processed/train.jsonl", train_out),
        "validation": ("data/processed/validation.jsonl", validation_out),
        "test": ("data/holdout/gsm8k_test.jsonl", [{**r, "split": "test"} for r in test]),
        "external": ("data/holdout/svamp.jsonl", [{**r, "split": "external"} for r in external]),
    }
    manifest = {"sources": lock, "source_sha256": {k: sha256_file(v) for k, v in sources.items()},
                "seed": seed, "validation_target": validation_size, "near_threshold": threshold,
                "near_method": "question-only word(1,2) TF-IDF cosine; connected components; conservative quarantine",
                "official_train_count": len(train), "quarantined_train_count": ntrain-len(kept),
                "pretraining_contamination": "unknown for GSM8K and SVAMP; deduplication does not establish absence",
                "splits": {}}
    for split, (relative, rows) in outputs.items():
        path = ROOT / relative
        jsonl_write(path, rows)
        manifest["splits"][split] = {"path": relative, "count": len(rows), "sha256": sha256_file(path),
                                     "id_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()}
    jsonl_write(ROOT / "data/audit/near_pairs.jsonl", pairs)
    json_write(ROOT / "data/audit/quarantine.json", {"ids": [train[i]["id"] for i in range(ntrain) if i not in kept]})
    json_write(ROOT / "data/manifest.json", manifest)
    # IDs and similarities only in audit: sealed questions and answers never appear here.
    json_write(ROOT / "data/audit/summary.json", {
        "counts": {s: info["count"] for s, info in manifest["splits"].items()},
        "near_pairs": len(pairs), "exact_pairs": sum(p["exact"] for p in pairs),
        "quarantined_train_count": ntrain-len(kept), "near_threshold": threshold,
        "residual_cross_split_train_validation_pairs": sum(
            i < ntrain and int(j) < ntrain and ((i in validation_indices) != (int(j) in validation_indices))
            and i in kept and int(j) in kept
            for i, js in enumerate(indices[:ntrain]) for j in js if int(j) > i),
        "scope": "one problem per row; no semantic or pretraining contamination guarantee"})
    return manifest


def load_split(split, intent, limit=None, seed=0):
    allowed = {"train": {"train"}, "select": {"train"}, "develop": {"train", "validation"},
               "evaluate_validation": {"validation"}, "final": {"test", "external"}}
    if split not in allowed.get(intent, set()):
        raise ValueError(f"Data isolation violation: {intent=} cannot read {split=}")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    info = manifest["splits"][split]
    path = ROOT / info["path"]
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"Data file changed after preparation: {path}")
    rows = jsonl_read(path)
    if len(rows) != info["count"] or any(r["split"] != split for r in rows):
        raise ValueError("Split provenance mismatch")
    if split in {"train", "validation"} and any(r["official_split"] != "train" for r in rows):
        raise ValueError("Holdout-derived sample in development split")
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit is not None else rows
