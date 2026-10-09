import os,sys
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
@pytest.fixture
def source_root():return Path(r"G:\Articoli\Articoli da Completare\LoSTer 2026\baselines\src")
