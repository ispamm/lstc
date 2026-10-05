"""Active main.py and experiment.py, Audit 4 paired preparation and monitoring.
No labels in batches/loss/stopping; matching occurs once, before the optimizer.
"""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .architecture import Autoencoder, TwoViewModel
from .assignments import rbf_logits, hard_assignments, deterministic_assignments
from .augmentation import two_snapshots, array_hash
from .alignment import match_initial_centers
from .checkpointing import save_checkpoint, load_checkpoint, trusted_load
from .diagnostics import (view_diagnostics, correspondence, occupancy,
                          component_gradients, total_gradient_norm)
from .efficiency import timed, synchronize, reset_gpu_peak, memory_and_parameters
from .losses import objective
from .protocol import temperature, changed_fraction, should_stop
from .reproducibility import (capture_rng, restore_rng, seed_all, resolve_device,
                              source_hash, provenance, write_json, preserve_rng)

def _loader(x, augmented, config):
    tensors = TensorDataset(torch.from_numpy(x), torch.from_numpy(augmented))
    loader = DataLoader(tensors, batch_size=min(config.batch_size, len(x)),
                        shuffle=True, drop_last=config.drop_last,
                        num_workers=config.num_workers, pin_memory=True)
    if len(loader) == 0:
        raise ValueError("No training batches")
    return loader

def encode(model, x, device, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            _, z = model(torch.from_numpy(x[start:start + batch_size]).to(device).unsqueeze(-1))
            outputs.append(z)
    return torch.cat(outputs, dim=0)

def infer(model, x, device, batch_size, sigma):
    z = encode(model.original, x, device, batch_size)
    return deterministic_assignments(z, model.centers_original, sigma).detach().cpu().numpy()

def _pretrain(model, loader, column, config, device):
    # legacy net/model.py pretrain_ae/pretrain_ae_augmented: 50 fixed Adam epochs.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr,
                                 betas=config.adam_betas, eps=config.adam_eps,
                                 weight_decay=config.weight_decay)
    model.train()
    history = []
    for epoch in range(config.pretrain_epochs):
        values = []
        for pair in loader:
            x = pair[column].to(device).unsqueeze(-1)
            optimizer.zero_grad()
            reconstruction, _ = model(x)
            loss = F.mse_loss(reconstruction, x)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reconstruction pretraining")
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach().item()))
        history.append(float(np.mean(values)))
    return history

def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

def prepare_initialization(dataset, config, device):
    from sklearn.cluster import KMeans
    seed_all(config.seed)
    timing = {}
    reset_gpu_peak(device)
    with timed(device, timing, "augmentation_seconds"):
        a_train, a_init, augmentation_info = two_snapshots(dataset.series, config)
    x = dataset.x
    model = TwoViewModel(x.shape[1], dataset.k, config).to(device)
    loader = _loader(x, a_train, config)
    with timed(device, timing, "original_pretrain_seconds"):
        original_history = _pretrain(model.original, loader, 0, config, device)
    with timed(device, timing, "augmented_pretrain_seconds"):
        augmented_history = _pretrain(model.augmented, loader, 1, config, device)
    args = dict(n_clusters=dataset.k, init=config.kmeans_init,
                n_init=config.kmeans_n_init, random_state=config.seed,
                max_iter=config.kmeans_max_iter, tol=config.kmeans_tol, algorithm=config.kmeans_algorithm)
    with timed(device, timing, "original_kmeans_seconds"):
        zo = encode(model.original, x, device, config.batch_size)
        centers_o = KMeans(**args).fit(zo.cpu().numpy()).cluster_centers_
    # Legacy LoSTer wrapper constructs a fresh AE, then reloads pretraining.
    state_o = _cpu_state(model.original)
    model.original = Autoencoder(x.shape[1], config).to(device)
    model.original.load_state_dict(state_o)
    with timed(device, timing, "augmented_kmeans_seconds"):
        za = encode(model.augmented, a_init, device, config.batch_size)
        centers_a = KMeans(**args).fit(za.cpu().numpy()).cluster_centers_
    state_a = _cpu_state(model.augmented)
    model.augmented = Autoencoder(x.shape[1], config).to(device)
    model.augmented.load_state_dict(state_a)
    with torch.no_grad():
        model.centers_original.copy_(torch.from_numpy(centers_o).to(device))
        model.centers_augmented.copy_(torch.from_numpy(centers_a).to(device))
    labels_o = deterministic_assignments(zo, model.centers_original, config.sigma).cpu().numpy()
    labels_a = deterministic_assignments(za, model.centers_augmented, config.sigma).cpu().numpy()
    init_info = {
        "original": view_diagnostics(zo, model.centers_original, labels_o, dataset.k, None, config.sigma),
        "augmented_A_init": view_diagnostics(za, model.centers_augmented, labels_a, dataset.k, None, config.sigma),
        "correspondence": correspondence(labels_o, labels_a, dataset.k)}
    return {"schema": 1, "config_initialization_id": config.initialization_id(),
            "source_sha256": source_hash(), "dataset_metadata": dataset.metadata,
            "X_sha256": array_hash(x), "device": str(device),
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "model_state": _cpu_state(model), "A_train": a_train, "A_init": a_init,
            "augmentation": augmentation_info, "labels_original": labels_o,
            "labels_augmented": labels_a, "initial_diagnostics": init_info,
            "pretraining_losses": {"original": original_history, "augmented": augmented_history},
            "timing": timing, "preparation_resources": memory_and_parameters(model, device),
            "rng_joint_boundary": capture_rng()}

def _initialization(dataset, config, device, output_root):
    cache_key = config.initialization_id() + "-" + source_hash()[:16] + "-" + array_hash(dataset.x)[:16]
    cache = Path(output_root) / "artifacts" / "initialization" / config.dataset / (
        "seed-%d" % config.seed) / (cache_key + ".pt")
    reused = cache.exists()
    if reused:
        initial = trusted_load(cache, "cpu")
        if (initial["device"] != str(device) or initial["torch_version"] != torch.__version__ or
                initial["numpy_version"] != np.__version__ or
                initial["dataset_metadata"] != dataset.metadata):
            raise ValueError("Incompatible paired initialization cache")
    else:
        initial = prepare_initialization(dataset, config, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("xb") as stream:
            torch.save(initial, stream)
    return initial, reused, cache_key

def initialize_variant(initial, length, k, config, device):
    model = TwoViewModel(length, k, config).to(device)
    model.load_state_dict(initial["model_state"], strict=True)
    alignment = {"matching_calls": 0, "ground_truth_used": False}
    if config.variant == "LoSTer-AlignedInit":
        centers, alignment = match_initial_centers(
            initial["labels_original"], initial["labels_augmented"],
            model.centers_augmented.detach().cpu().numpy())
        with torch.no_grad():
            model.centers_augmented.copy_(torch.from_numpy(centers).to(device))
    # Same boundary AFTER all arm-specific construction; matching consumes no RNG.
    restore_rng(initial["rng_joint_boundary"])
    return model, alignment

def optimizer_and_scheduler(model, config):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.joint_lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step,
                                                gamma=config.scheduler_gamma)
    return optimizer, scheduler

def fit(dataset, config, output_root):
    from .metrics import evaluate
    config.validate()
    if not config.dataset.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Unsafe dataset path component")
    device = resolve_device(config.device)
    root = Path(output_root)
    key = Path(config.dataset) / config.variant / ("seed-%d" % config.seed)
    artifact_dir, log_dir, result_dir = [root / part / key for part in ("artifacts", "logs", "results")]
    for path in (artifact_dir, log_dir, result_dir):
        if path.exists():
            raise FileExistsError("Refusing to overwrite an existing run: " + str(key))
    artifact_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    manifest = provenance(config, dataset.metadata, device)
    manifest["status"] = "preparing"
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)
    timing = {}
    synchronize(device)
    full_start = time.perf_counter()
    try:
        initial, reused, cache_key = _initialization(dataset, config, device, root)
        model, alignment = initialize_variant(initial, dataset.x.shape[1], dataset.k, config, device)
        optimizer, scheduler = optimizer_and_scheduler(model, config)
        loader = _loader(dataset.x, initial["A_train"], config)
        previous_o, previous_a = None, None
        cumulative_o = np.zeros(dataset.k, dtype=np.int64)
        cumulative_a = np.zeros(dataset.k, dtype=np.int64)
        dead_o = np.zeros(dataset.k, dtype=np.int64)
        dead_a = np.zeros(dataset.k, dtype=np.int64)
        previous_centers_o = model.centers_original.detach().clone()
        previous_centers_a = model.centers_augmented.detach().clone()
        records, updates, samples = [], 0, 0
        reset_gpu_peak(device)
        joint_start = time.perf_counter()
        for epoch in range(config.joint_epochs):
            synchronize(device)
            epoch_start = time.perf_counter()
            tau = temperature(config, epoch)
            model.train()
            values, gradients, batch_occupancy = [], None, []
            lr = optimizer.param_groups[0]["lr"]
            diagnostics_seconds = 0.0
            for batch_index, (x, xa) in enumerate(loader):
                x, xa = x.to(device).unsqueeze(-1), xa.to(device).unsqueeze(-1)
                optimizer.zero_grad()
                xr, z = model.original(x)
                xar, za = model.augmented(xa)
                q = hard_assignments(rbf_logits(z, model.centers_original, config.sigma), tau)
                qa = hard_assignments(rbf_logits(za, model.centers_augmented, config.sigma), tau)
                terms = objective(x, xa, xr, xar, z, za, q, qa,
                                  model.centers_original, model.centers_augmented, config)
                if not all(bool(torch.isfinite(v).item()) for v in terms.values()):
                    raise FloatingPointError("Nonfinite objective; no rescue intervention")
                if config.gradient_diagnostics and epoch in config.gradient_epochs and batch_index == 0:
                    synchronize(device)
                    diagnostic_start = time.perf_counter()
                    gradients = component_gradients(terms, model, config.alpha)
                    synchronize(device)
                    diagnostics_seconds += time.perf_counter() - diagnostic_start
                terms["total"].backward()
                norm = total_gradient_norm(model)
                if not np.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient; no clipping or reseeding")
                optimizer.step()
                values.append({name: float(value.detach().item()) for name, value in terms.items()})
                batch_occupancy.append({
                    "original": np.bincount(q.detach().argmax(1).cpu().numpy(), minlength=dataset.k).tolist(),
                    "augmented": np.bincount(qa.detach().argmax(1).cpu().numpy(), minlength=dataset.k).tolist()})
                updates += 1
                samples += len(x)
            scheduler.step()  # Exact legacy order: step after epoch, before stopping.
            # Deterministic diagnostics consume no additional stochastic forward.
            with preserve_rng():
                zo = encode(model.original, dataset.x, device, config.batch_size)
                za = encode(model.augmented, initial["A_train"], device, config.batch_size)
                po = deterministic_assignments(zo, model.centers_original, config.sigma).cpu().numpy()
                pa = deterministic_assignments(za, model.centers_augmented, config.sigma).cpu().numpy()
                vo = view_diagnostics(zo, model.centers_original, po, dataset.k, previous_o, config.sigma)
                va = view_diagnostics(za, model.centers_augmented, pa, dataset.k, previous_a, config.sigma)
                for diag, centers, old in ((vo, model.centers_original, previous_centers_o),
                                          (va, model.centers_augmented, previous_centers_a)):
                    diag["center_displacements"] = (centers.detach() - old).norm(dim=1).cpu().tolist()
                    diag["near_collapse"] = diag["maximum_cluster_fraction"] >= config.near_collapse_fraction
                no, na = np.array(vo["counts"]), np.array(va["counts"])
                cumulative_o += no
                cumulative_a += na
                dead_o = np.where(no == 0, dead_o + 1, 0)
                dead_a = np.where(na == 0, dead_a + 1, 0)
                vo.update(cumulative_hard_use=cumulative_o.tolist(), dead_center_duration=dead_o.tolist())
                va.update(cumulative_hard_use=cumulative_a.tolist(), dead_center_duration=dead_a.tolist())
                metrics = evaluate(dataset.labels, po)  # Never passed to should_stop.
                correspondence_info = correspondence(po, pa, dataset.k)
            stop = should_stop(po, previous_o, epoch + 1, config)
            synchronize(device)
            row = {"epoch": epoch + 1, "tau": tau, "learning_rate": lr,
                   "next_learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_epoch_seconds": time.perf_counter() - epoch_start,
                   "losses": {name: float(np.mean([v[name] for v in values])) for name in values[0]},
                   "component_gradient_norms": gradients, "last_batch_total_gradient_norm": norm,
                   "sparse_gradient_diagnostics_seconds": diagnostics_seconds,
                   "original": vo, "augmented_A_train": va, "batch_hard_counts": batch_occupancy,
                   "correspondence": correspondence_info, "metrics_evaluation_only": metrics,
                   "effective_batch": min(config.batch_size, len(dataset.x)),
                   "updates": updates, "processed_original_records": samples,
                   "stop_triggered": stop}
            records.append(row)
            with (log_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            previous_o, previous_a = po.copy(), pa.copy()
            previous_centers_o = model.centers_original.detach().clone()
            previous_centers_a = model.centers_augmented.detach().clone()
            if stop:
                break
        synchronize(device)
        timing["joint_operational_seconds"] = time.perf_counter() - joint_start
        timing["observed_fit_operational_seconds"] = time.perf_counter() - full_start + dataset.preprocessing_seconds
        with timed(device, timing, "inference_seconds"):
            predicted = infer(model, dataset.x, device, config.batch_size, config.sigma)
        resources = memory_and_parameters(model, device)
        resources["observed_joint_gpu_allocated_bytes"] = resources["peak_gpu_allocated_bytes"]
        resources["observed_joint_gpu_reserved_bytes"] = resources["peak_gpu_reserved_bytes"]
        for name in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "host_peak_bytes"):
            before = initial["preparation_resources"][name]
            now = resources[name]
            if before is not None:
                resources[name] = max(before, now or 0)
        resources["preparation_peak_reused_from_cache"] = reused
        prep_seconds = sum(initial["timing"].values())
        timing.update(initial["timing"])
        timing["input_loading_and_normalization_seconds"] = dataset.preprocessing_seconds
        timing["stage_summed_complete_fit_estimate_seconds"] = prep_seconds + timing["joint_operational_seconds"] + dataset.preprocessing_seconds
        timing["timing_quality"] = "instrumented operational run; not benchmark-quality warm/repeated timing"
        timing["training_samples_per_second"] = samples / timing["joint_operational_seconds"]
        timing["inference_samples_per_second"] = len(predicted) / max(timing["inference_seconds"], 1e-15)
        manifest.update(status="completed", final_epoch=epoch + 1, final_temperature=tau,
                        stop_reason="assignment-change" if stop else "max-epochs",
                        paired_cache_key=cache_key, preparation_cache_reused=reused,
                        augmentation=initial["augmentation"], alignment=alignment,
                        initialization_diagnostics=initial["initial_diagnostics"],
                        timing=timing, resources=resources)
        checkpoint = artifact_dir / "complete.pt"
        save_checkpoint(checkpoint, model, config, epoch + 1, tau, optimizer, scheduler,
                        predicted.tolist(), metrics, manifest, capture_rng(),
                        {"final_augmented_assignments": pa.tolist(), "diagnostics": records[-1]})
        # Engineering round-trip verification, not a new trained model.
        with preserve_rng():
            restored, payload = load_checkpoint(checkpoint, device)
            reloaded = infer(restored, dataset.x, device, config.batch_size, config.sigma)
        if not np.array_equal(predicted, reloaded):
            raise AssertionError("Checkpoint deterministic assignments changed")
        result = {"dataset": config.dataset, "variant": config.variant, "seed": config.seed,
                  "metrics": metrics, "final_epoch": epoch + 1, "complete": True,
                  "finite": True, "collapse_original": records[-1]["original"]["complete_collapse"],
                  "collapse_augmented": records[-1]["augmented_A_train"]["complete_collapse"],
                  "fit_seconds": timing["stage_summed_complete_fit_estimate_seconds"],
                  "fit_seconds_kind": "stage-summed estimate including both pretrains",
                  "resources": resources, "paired_cache_key": cache_key}
        write_json(result_dir / "metrics.json", result)
        manifest["checkpoint_round_trip_exact"] = True
        write_json(manifest_path, manifest)
        return result
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(manifest_path, manifest)
        raise
