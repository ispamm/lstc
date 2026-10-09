"""Process bootstrap: no scientific source vendoring, no global installs."""
import os,sys
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONHASHSEED"]="0"
os.environ["MPLBACKEND"]="Agg"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.dont_write_bytecode=True
if sys.version_info < (3,8):
    import importlib_metadata,importlib
    sys.modules["importlib.metadata"]=importlib_metadata
    importlib.metadata=importlib_metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baselines2026.wave2a.cli import main
if __name__=="__main__":main()
