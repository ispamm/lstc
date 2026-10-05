"""Engineering provenance/RNG replay, legacy random_seed.py and Audit 4."""
from __future__ import annotations
import contextlib
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[2]
REQUIRED = ("numpy", "scipy", "sklearn", "torch")

def environment():
    packages = {}
    for name in REQUIRED + ("pandas", "pytest"):
        try:
            module = importlib.import_module(name)
            packages[name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as error:
            packages[name] = {"available": False, "error_type": type(error).__name__}
    gpu = {"torch_cuda_available": False, "cuda_runtime": None, "gpu": None}
    if packages["torch"]["available"]:
        import torch
        gpu["torch_cuda_available"] = torch.cuda.is_available()
        gpu["cuda_runtime"] = torch.version.cuda
        if gpu["torch_cuda_available"]:
            gpu["gpu"] = torch.cuda.get_device_name(0)
    return {"python": platform.python_version(), "os": platform.platform(),
            "packages": packages, **gpu}

def git_metadata():
    def call(*args):
        result = subprocess.run(["git", "-C", str(REPO), *args],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Cannot capture Git provenance")
        return result.stdout.strip()
    status = call("status", "--porcelain")
    return {"sha": call("rev-parse", "HEAD"), "dirty": bool(status),
            "status": status.splitlines()}

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()

def source_hash():
    records = [(str(p.relative_to(PACKAGE)), sha256_file(p))
               for p in sorted((PACKAGE / "src").rglob("*.py"))]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG snapshot cannot be replayed on CPU-only runtime")
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])

@contextlib.contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)

def resolve_device(requested):
    import torch
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable to this PyTorch installation")
    if requested not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda")
    return torch.device(requested)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

def provenance(config, metadata, device):
    return {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git": git_metadata(), "source_sha256": source_hash(),
            "environment": environment(), "config": config.to_dict(),
            "dataset": metadata, "variant": config.variant,
            "seed": config.seed, "resolved_device": str(device)}
