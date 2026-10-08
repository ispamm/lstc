"""Only the three audited READY methods may be admitted."""
from pathlib import Path
import subprocess
from .util import ContractError, read_json

READY_IDS = ("euclidean_kmeans", "kshape", "kasba")
SOURCE_DIRS = {
    "euclidean_kmeans": "scikit-learn__scikit-learn",
    "kshape": "TheDatumOrg__kshape-python",
    "kasba": "aeon-toolkit__aeon",
}
BASELINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = BASELINE_ROOT.parents[1]
AUDIT_REGISTRY = REPOSITORY / "analysis/baselines/baseline-registry-2026.json"

def methods():
    records = read_json(AUDIT_REGISTRY)["baselines"]
    ready = {x["method_id"]: x for x in records if x["readiness"] == "READY FOR IMPLEMENTATION"}
    if set(ready) != set(READY_IDS):
        raise ContractError("Audit registry Wave 1 readiness changed")
    return ready

def method_record(method):
    if method not in READY_IDS:
        raise ContractError("Conditional/blocked baselines are not admitted")
    return methods()[method]

def verify_checkout(method, source_root):
    record = method_record(method)
    path = Path(source_root) / SOURCE_DIRS[method]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("rev-parse", "HEAD") != record["source_commit"]:
        raise ContractError("Frozen external source commit mismatch")
    if git("status", "--porcelain"):
        raise ContractError("External source working tree is not clean")
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git").lower() != record["source_url"].lower():
        raise ContractError("External source remote mismatch")
    return path, record

def default_config(method):
    method_record(method)
    return read_json(BASELINE_ROOT / "configs/wave1.json")["methods"][method]
