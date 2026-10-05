"""Execute tiny artificial CPU-only pipeline checks; never loads UCR."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loster2026.reproducibility import environment, REQUIRED
if __name__ == "__main__":
    env = environment()
    missing = [name for name in REQUIRED if not env["packages"][name]["available"]]
    if missing:
        print(json.dumps({"status": "BLOCKED", "missing_or_unimportable": missing}))
        sys.exit(2)
    from loster2026.smoke import synthetic_smoke
    results = synthetic_smoke()
    print(json.dumps({"status": "EXECUTION_CHECK_PASSED", "variants": [r["variant"] for r in results],
                      "scientific_quality_interpretation": False}))
