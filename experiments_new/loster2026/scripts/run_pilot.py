"""Dry-run by default. No data loaded unless --execute is supplied explicitly."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataclasses import replace
from loster2026.config import load_campaign, assert_pilot_contract, validate_manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/pilot/manifest.json").read_text(encoding="utf-8"))
    runs = validate_manifest(manifest)
    if not args.execute:
        print(json.dumps({"execution": False, "run_count": len(runs), "runs": runs}, indent=2))
        return 0
    if args.data_root is None:
        parser.error("--execute requires --data-root (canonical UCR folder)")
    # Check readiness before touching data or generated output paths.
    from loster2026.reproducibility import environment, REQUIRED
    env = environment()
    missing = [name for name in REQUIRED if not env["packages"][name]["available"]]
    if missing:
        parser.error("Missing/unimportable dependencies: " + ", ".join(missing))
    if args.run_index is not None and not 0 <= args.run_index < len(runs):
        parser.error("--run-index must be 0..119")
    from loster2026.data import load_ucr
    from loster2026.training import fit
    expected = manifest["dataset_metadata"]
    selected = runs if args.run_index is None else [runs[args.run_index]]
    for row in selected:
        campaign, base = load_campaign(ROOT / "configs" / row["config_id"])
        config = replace(base, dataset=row["dataset"], seed=row["seed"],
                         variant=row["variant"], device=args.device)
        assert_pilot_contract(config)
        data = load_ucr(args.data_root, config.dataset, expected[config.dataset])
        fit(data, config, args.output_root)
    return 0

if __name__ == "__main__":
    sys.exit(main())
