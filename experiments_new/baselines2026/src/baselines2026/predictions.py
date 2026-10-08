"""Versioned immutable, ordered predictions; no label dependency."""
import re
import numpy as np
from .util import ContractError, read_json, write_json_new

SCHEMA = "baselines2026.predictions.v1"

def integer_clusters(values, N):
    x = np.asarray(values)
    if x.ndim != 1 or len(x) != N or N == 0:
        raise ContractError("Exactly N one-dimensional predictions required")
    if x.dtype.kind not in "iuf":
        raise ContractError("Predictions must be numeric integers, not booleans/strings")
    if x.dtype.kind == "f":
        if not np.isfinite(x).all() or np.any(x != np.floor(x)):
            raise ContractError("Nonfinite/noninteger prediction")
        if np.any(x < -(2**63)) or np.any(x >= 2**63):
            raise ContractError("Cluster ID outside int64")
    elif x.dtype.kind == "u" and np.any(x > np.uint64(2**63 - 1)):
        raise ContractError("Cluster ID outside int64")
    return x.astype(np.int64)

def validate_submission(p, identity):
    if p.get("schema") != SCHEMA or p.get("execution_status") != "SUCCESS":
        raise ContractError("Only successful v1 submissions are evaluable")
    for key in ("dataset", "N", "L", "k", "input_hashes", "raw_array_sha256", "sample_id_sha256"):
        if p.get(key) != identity.get(key):
            raise ContractError("Dataset identity/hash mismatch: " + key)
    ids = p.get("sample_ids")
    if not isinstance(ids, list) or len(ids) != identity["N"]:
        raise ContractError("Sample count mismatch")
    if not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ContractError("Invalid/duplicate sample identifiers")
    if ids != identity["sample_ids"]:
        raise ContractError("Predictions must be in canonical sample order")
    for key, width in (("config_sha256", 64), ("environment_sha256", 64),
                       ("adapter_tree_sha256", 64), ("source_commit", 40)):
        if not isinstance(p.get(key), str) or not re.fullmatch("[a-f0-9]{%d}" % width, p[key]):
            raise ContractError("Missing/invalid provenance: " + key)
    if not isinstance(p.get("method"), str) or not p["method"]:
        raise ContractError("Missing method")
    if type(p.get("seed")) is not int or not 0 <= p["seed"] < 2**32:
        raise ContractError("Invalid root seed")
    if not isinstance(p.get("run_id"), str) or not p["run_id"]:
        raise ContractError("Missing run identity")
    return integer_clusters(p.get("predicted_clusters"), identity["N"])

def submission(identity, clusters, provenance):
    p = {**identity, **provenance, "schema": SCHEMA, "execution_status": "SUCCESS",
         "predicted_clusters": integer_clusters(clusters, identity["N"]).tolist()}
    validate_submission(p, identity)
    return p

def save_submission(path, payload, identity):
    validate_submission(payload, identity)
    write_json_new(path, payload)

def load_submission(path, identity):
    p = read_json(path)
    validate_submission(p, identity)
    return p

def utilization(clusters, k):
    x = np.asarray(clusters)
    return {"occupied_clusters": len(np.unique(x)), "oracle_k": int(k),
            "fraction_occupied": len(np.unique(x)) / k,
            "collapsed": bool(k > 1 and len(np.unique(x)) == 1)}
