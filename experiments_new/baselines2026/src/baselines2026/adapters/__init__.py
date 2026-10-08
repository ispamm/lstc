"""Thin native wrappers. The fit API has no ground-truth membership argument."""
from dataclasses import dataclass
import numpy as np
from threadpoolctl import threadpool_limits
from ..predictions import integer_clusters
from ..provenance import seeded
from ..registry import default_config
from ..util import ContractError

@dataclass
class FitResult:
    model: object
    clusters: object
    details: dict

def native_model(method, k, seed, source_root):
    # Imports/source are independently verified before this factory.
    config = default_config(method)
    if method == "euclidean_kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, random_state=seed, **config["parameters"])
    if method == "kshape":
        import sys
        from pathlib import Path
        path = Path(source_root) / "TheDatumOrg__kshape-python"
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from kshape.core import KShapeClusteringCPU
        return KShapeClusteringCPU(n_clusters=k, **config["parameters"])
    if method == "kasba":
        from aeon.clustering import KASBA
        return KASBA(n_clusters=k, random_state=seed, **config["parameters"])
    raise ContractError("Method is not READY for Wave 1")

def native_shape(method, x):
    if method == "kshape":
        return x[:, :, None]
    if method == "kasba":
        return x[:, None, :]
    return x

def fit_predict(method, x, k, seed, source_root):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= k <= len(x):
        raise ContractError("Valid finite N,L and known k required")
    with seeded(seed), threadpool_limits(limits=1):
        model = native_model(method, k, seed, source_root)
        model.fit(native_shape(method, x))  # NO y argument
        clusters = integer_clusters(model.labels_, len(x)).copy()
    details = {"partition": "native fit labels_ (pooled clustering)",
               "native_class": type(model).__module__ + "." + type(model).__name__,
               "effective_batch_size": None,
               "n_iter": int(model.n_iter_) if getattr(model, "n_iter_", None) is not None else None,
               "stop": "native label-free convergence or published maximum",
               "parameters": model.get_params(deep=False)}
    # All current parameters are JSON-native; no estimator/callable substitution.
    return FitResult(model, clusters, details)

def infer(method, model, x, seed):
    with seeded(seed), threadpool_limits(limits=1):
        return integer_clusters(model.predict(native_shape(method, x)), len(x))
