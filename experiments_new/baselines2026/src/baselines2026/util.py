"""Stable identities and immutable external artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

class ContractError(ValueError):
    pass

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def array_hash(values):
    x = np.ascontiguousarray(values)
    h = hashlib.sha256(canonical_json({"shape": list(x.shape), "dtype": x.dtype.str}).encode())
    h.update(x.tobytes())
    return h.hexdigest()

def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def external_output(path, repository):
    path, repository = Path(path).resolve(), Path(repository).resolve()
    if path == repository or repository in path.parents:
        raise ContractError("Generated artifacts must be outside the Git repository")
    if path.drive.upper() != "G:":
        raise ContractError("Wave 1 outputs must reside on the authorized G: drive")
    return path
