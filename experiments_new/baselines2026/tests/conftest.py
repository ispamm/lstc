import os
import sys
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from baselines2026.datasets import TrainingInput

SOURCE_ROOT = Path(r"G:\Articoli\Articoli da Completare\LoSTer 2026\baselines\src")

@pytest.fixture
def source_root():
    return SOURCE_ROOT

@pytest.fixture
def toy():
    x = np.array([[0.,1.,2.,3.],[1.,2.,1.,0.],[3.,1.,0.,2.],[2.,0.,3.,1.]])
    return TrainingInput("Toy", x, tuple("Toy/TRAIN/%d" % i for i in range(4)),
                         2, {"TRAIN": "1"*64, "TEST": "2"*64}, {})

@pytest.fixture
def payload(toy):
    from baselines2026.predictions import submission
    return submission(toy.identity(), [0,0,1,1],
                      {"method": "euclidean_kmeans", "seed": 0, "run_id": "toy-run",
                       "config_sha256": "3"*64, "environment_sha256": "4"*64,
                       "source_commit": "5"*40, "adapter_tree_sha256": "6"*64})
