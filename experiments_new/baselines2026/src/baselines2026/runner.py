"""One authorized real fit per method; separate-process evaluation and exact reuse."""
import argparse
import pickle
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from .adapters import fit_predict, infer
from .datasets import load_training_input, normalize_rows
from .predictions import submission, save_submission, load_submission, utilization
from .provenance import (adapter_source, environment, phase_seed, rng_settings,
                         snapshot_adapter, verify_native)
from .registry import BASELINE_ROOT, REPOSITORY, READY_IDS, default_config
from .resources import ResourceMonitor, hardware
from .timing import PipelineTimer
from .util import ContractError, array_hash, digest, external_output, file_hash, read_json, write_json_new

def run_once(method, data_root, source_root, output_root, gate_path, lock_path):
    config = read_json(BASELINE_ROOT / "configs/wave1.json")
    name, seed = config["dataset"], config["seed"]
    if name != "SyntheticControl" or seed != 0:
        raise ContractError("This stage authorizes only SyntheticControl, development seed0")
    proof = read_json(gate_path)
    adapter = adapter_source()
    env, env_hash = environment(lock_path)
    native = verify_native(method, source_root)
    if (proof.get("schema") != "baselines2026.validation.v1" or not proof.get("passed")
        or proof["adapter_tree_sha256"] != adapter["adapter_tree_sha256"]
        or proof["environment_sha256"] != env_hash
        or set(proof["synthetic_passed"]) != set(READY_IDS)
        or proof["unit_tests"]["failed"] or proof["unit_tests"]["errors"]):
        raise ContractError("Unit/synthetic/source validation gate missing or stale")
    # This is the only real dataset read by training in this stage.
    data = load_training_input(data_root, name, config["expected"])
    identity = data.identity()
    config_hash = digest({"dataset": name, "seed": seed, "known_k": data.k,
                          "method": method, "recipe": default_config(method),
                          "preprocessing": config["preprocessing"], "threads": config["cpu_threads"]})
    key = {"method": method, "dataset": name, "seed": seed, "input_hashes": data.input_hashes,
           "config_sha256": config_hash, "source_commit": native["source_commit"],
           "environment_sha256": env_hash, "adapter_tree_sha256": adapter["adapter_tree_sha256"]}
    run_id = digest(key)
    folder = external_output(output_root, REPOSITORY) / method / run_id
    if folder.exists():
        p = load_submission(folder / "predictions.json", identity)
        for field, expected in {**key, "run_id": run_id}.items():
            if p.get(field) != expected:
                raise ContractError("Existing run identity mismatch")
        manifest = read_json(folder / "manifest.json")
        for artifact, expected in manifest["artifact_sha256"].items():
            if file_hash(folder / artifact) != expected:
                raise ContractError("Frozen artifact changed")
        print("REUSED WITHOUT FIT", method, run_id, flush=True)
        return folder
    folder.mkdir(parents=True)
    snapshot_adapter(folder, adapter)
    timer = PipelineTimer()  # CPU; synchronize hook available for future CUDA adapters.
    try:
        with ResourceMonitor() as resource, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with timer.pipeline():
                with timer.stage("normalization_and_format"):
                    x = normalize_rows(data.raw)
                with timer.stage("native_initialization_and_fit"):
                    fitted = fit_predict(method, x, data.k, seed, source_root)
                inference_seed = phase_seed(method, name, seed, "inference")
                with timer.stage("full_pool_inference"):
                    inference = infer(method, fitted.model, x, inference_seed)
                # Canonical native fitted partition is the transductive result.
                with timer.stage("assignment_validation"):
                    clusters = fitted.clusters
                    payload = submission(identity, clusters, {**key, "run_id": run_id,
                                         "preprocessing": config["preprocessing"],
                                         "normalized_array_sha256": array_hash(x)})
        warning_records = [{"category": type(w.message).__name__, "message": str(w.message)}
                           for w in caught]
        with (folder / "model.pkl").open("xb") as stream:
            pickle.dump(fitted.model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (folder / "model.pkl").open("rb") as stream:
            reloaded = pickle.load(stream)  # Only self-produced trusted artifacts.
        if not np.array_equal(reloaded.labels_, clusters):
            raise ContractError("Native partition failed checkpoint reload")
        reload_inference = infer(method, reloaded, x, inference_seed)
        if not np.array_equal(reload_inference, inference):
            raise ContractError("Native inference failed checkpoint reload")
        save_submission(folder / "predictions.json", payload, identity)
        manifest = {**key, "run_id": run_id, "execution_status": "SUCCESS",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_identity": identity, "normalized_array_sha256": array_hash(x),
                    "source": native, "adapter": adapter, "environment": env,
                    "hardware": hardware(), "rng": rng_settings(seed),
                    "inference_seed": inference_seed, "native_fit": fitted.details,
                    "preprocessing": config["preprocessing"], "utilization": utilization(clusters, data.k),
                    "pipeline_wall_seconds": timer.total, "stage_seconds": timer.stages,
                    "inference_seconds": timer.stages["full_pool_inference"],
                    "resources": resource.result(), "warnings": warning_records,
                    "fit_partition_matches_fresh_inference": bool(np.array_equal(clusters, inference)),
                    "checkpoint_reload": {"partition_exact": True, "inference_exact": True},
                    "timing_scope": "raw numeric array before normalization through fit/ordered inference; file I/O/import/source verification/checkpoint/evaluation excluded",
                    "timing_quality": "integration smoke only; not publication efficiency",
                    "artifact_sha256": {p: file_hash(folder / p) for p in ("model.pkl", "predictions.json")}}
        write_json_new(folder / "manifest.json", manifest)
        print("SUCCESS", method, run_id, "N", data.N, "pipeline_seconds", timer.total, flush=True)
        return folder
    except Exception as error:
        kind = "RESOURCE FAILURE" if isinstance(error, MemoryError) else ("ADAPTER BUG" if isinstance(error, ContractError) else "UNKNOWN")
        write_json_new(folder / "failure.json", {"method": method, "run_id": run_id,
                       "classification": kind, "error": repr(error), "traceback": traceback.format_exc()})
        raise

def main():
    p = argparse.ArgumentParser(description="Restricted Wave 1 infrastructure CLI")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--method", choices=READY_IDS, required=True)
    for key in ("data-root", "source-root", "output-root", "gate", "lock"):
        run.add_argument("--" + key, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "run":
        run_once(args.method, args.data_root, args.source_root, args.output_root, args.gate, args.lock)
    else:
        from .evaluator import evaluate_file
        external_output(args.output, REPOSITORY)
        result = evaluate_file(args.predictions, args.data_root, args.output)
        print(result)
