"""Source, environment, RNG and uncommitted-adapter identities."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import sysconfig
import numpy as np
from .registry import BASELINE_ROOT, REPOSITORY, verify_checkout
from .util import ContractError, digest, file_hash

def environment(lock_path):
    # Source checkout metadata on sys.path is not an installed distribution.
    # k-Shape is identified separately by its frozen source commit/file hashes.
    sites = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    packages = sorted((d.metadata["Name"].lower(), d.version)
                      for d in importlib.metadata.distributions(path=sites))
    lock = Path(lock_path)
    if not lock.is_file():
        raise ContractError("Frozen environment lock is required")
    listed = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.strip().split("==")
            listed[name.lower().replace("_", "-")] = version
    actual = {n.replace("_", "-"): v for n, v in packages}
    if actual != listed:
        raise ContractError("Installed environment differs from frozen lock")
    info = {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "packages": packages, "lock_sha256": file_hash(lock)}
    return info, digest(info)

def adapter_source():
    files = {str(p.relative_to(BASELINE_ROOT)).replace("\\", "/"): file_hash(p)
             for p in sorted(BASELINE_ROOT.rglob("*"))
             if p.is_file() and p.suffix in (".py", ".json", ".txt", ".md", ".ps1")
             and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(BASELINE_ROOT).parts)}
    parent = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    return {"adapter_commit": None, "adapter_parent_commit": parent,
            "adapter_source_committed": False, "adapter_tree_sha256": digest(files), "files": files}

def snapshot_adapter(output, info):
    destination = Path(output) / "adapter-source"
    destination.mkdir()
    for name, expected in info["files"].items():
        original = BASELINE_ROOT / name
        if file_hash(original) != expected:
            raise ContractError("Adapter changed during snapshot")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)

def phase_seed(method, dataset, root_seed, phase):
    text = "loster2026-baseline-v1|%s|%s|%s|%s" % (method, dataset, root_seed, phase)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)

@contextmanager
def seeded(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

def verify_native(method, source_root):
    path, record = verify_checkout(method, source_root)
    targets = {
        "euclidean_kmeans": [("sklearn", "sklearn/cluster/_kmeans.py")],
        "kasba": [("aeon", "aeon/clustering/_kasba.py"),
                  ("aeon", "aeon/clustering/averaging/_kasba_average.py"),
                  ("aeon", "aeon/distances/elastic/_msm.py")],
        "kshape": [("kshape", "kshape/core.py")],
    }
    if method == "kshape":
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        importlib.import_module("kshape.core")
    verified = {}
    for package, rel in targets[method]:
        installed = Path(importlib.import_module(package).__file__).resolve().parent.parent / rel
        if method == "kshape" and installed.resolve() != (path / rel).resolve():
            raise ContractError("k-Shape imported from wrong source")
        blob = subprocess.check_output(["git", "-C", str(path), "show", record["source_commit"] + ":" + rel])
        canonical = blob.decode("utf-8").replace("\r\n", "\n")
        actual = installed.read_text(encoding="utf-8").replace("\r\n", "\n")
        if canonical != actual:
            raise ContractError("Installed algorithm differs from frozen source: " + rel)
        verified[rel] = {"git_blob_sha256": hashlib.sha256(blob).hexdigest(),
                         "installed_path": str(installed), "match": True}
    return {"source_repository": record["source_url"], "source_commit": record["source_commit"],
            "source_status": record["source_status"], "license": record["license"],
            "verified_files": verified}

def rng_settings(seed):
    return {"root_seed": seed, "python_seed": seed, "numpy_seed": seed,
            "native_random_state": seed, "restores_caller_rng": True,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "thread_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "gpu": None}
