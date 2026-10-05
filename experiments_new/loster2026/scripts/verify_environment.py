"""Read-only actual import/device check. No installation or pilot execution."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loster2026.reproducibility import environment, REQUIRED
if __name__ == "__main__":
    info = environment()
    missing = [name for name in REQUIRED if not info["packages"][name]["available"]]
    info.update(verdict="NOT READY" if missing else "DEPENDENCIES IMPORTABLE; RUN REGRESSION SUITE",
                missing_or_unimportable_required=missing,
                reference_versions={"torch": "1.13.1+cu117", "numpy": "1.22.4",
                                    "sklearn": "1.4.1.post1", "scipy": "1.13.0"},
                reference_versions_verified=False)
    print(json.dumps(info, indent=2))
    sys.exit(2 if missing else 0)
