"""Replaces legacy experiment.py:135 incomplete save (F10).
Complete active state, separately for every dataset/variant/seed.
"""
from pathlib import Path
import inspect
import torch
from .architecture import TwoViewModel
from .config import from_dict

def trusted_load(path, device="cpu"):
    # Only locally created full-state artifacts: explicit full-state loading.
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)

def save_checkpoint(path, model, config, epoch, tau, optimizer, scheduler,
                    assignments, metrics, provenance, rng, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite completed seed checkpoint")
    payload = {"schema": 1, "config": config.to_dict(), "variant": config.variant,
               "dataset": config.dataset, "seed": config.seed,
               "length": model.original.length, "k": len(model.centers_original),
               "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               "final_epoch": epoch, "final_temperature": tau,
               "optimizer": optimizer.state_dict() if optimizer else None,
               "scheduler": scheduler.state_dict() if scheduler else None,
               "assignments": assignments, "metrics": metrics,
               "provenance": provenance, "rng": rng, "extra": extra or {}}
    # Exclusive creation protects against concurrent duplicate run invocations.
    with path.open("xb") as stream:
        torch.save(payload, stream)

def load_checkpoint(path, device="cpu"):
    payload = trusted_load(path, device)
    state = payload["model_state"]
    for key in ("centers_original", "centers_augmented"):
        if key not in state:
            raise ValueError("Checkpoint omits ACTIVE " + key)
    config = from_dict(payload["config"])
    model = TwoViewModel(payload["length"], payload["k"], config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
