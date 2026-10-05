"""Tiny artificial execution check only: no UCR and no quality interpretation."""
from dataclasses import replace
import tempfile
import numpy as np
from .config import Config, VARIANTS
from .data import Dataset, normalize_rows
from .training import fit

def synthetic_smoke(output_root=None):
    rng = np.random.RandomState(23)
    raw = rng.normal(size=(16, 24))
    series = normalize_rows(raw)
    data = Dataset(series, np.array([0, 1] * 8),
                   {"dataset": "SyntheticSmoke", "N": 16, "L": 24, "k": 2,
                    "train_count": 8, "test_count": 8,
                    "normalization": "per-series-z-ddof0", "oracle_k": True,
                    "input_kind": "artificial execution fixture; not scientific data"})
    def run(root):
        return [fit(data, replace(Config(), dataset="SyntheticSmoke", variant=variant,
                                  seed=23, latent_dim=8, pretrain_epochs=1,
                                  joint_epochs=2, batch_size=8, num_workers=0, device="cpu"), root)
                for variant in VARIANTS]
    if output_root is not None:
        return run(output_root)
    with tempfile.TemporaryDirectory(prefix="loster2026-smoke-") as root:
        return run(root)
