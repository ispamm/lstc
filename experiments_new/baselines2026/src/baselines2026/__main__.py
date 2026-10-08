"""Spawn-safe entry point; CPU limits set before importing numerical libraries."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

if __name__ == "__main__":
    from .runner import main
    main()
