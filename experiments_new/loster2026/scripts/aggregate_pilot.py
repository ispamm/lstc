"""Read-only aggregation/registered recommendation; never trains or freezes."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loster2026.selection import summarize

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(args.results_root.glob("*/*/seed-*/metrics.json"))]
    print(json.dumps(summarize(records), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
