"""Audit 4 contract; defaults from legacy utils/config.py and main.py/run.sh."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path

VARIANTS = ("LoSTer-Legacy-Clean", "LoSTer-NoResoftmax", "LoSTer-AlignedInit")
PILOT_DATASETS = ("SyntheticControl", "Beef", "ECG200", "OSULeaf", "ShapesAll",
                  "SemgHandMovementCh2", "CinCECGTorso", "StarLightCurves")
PILOT_SEEDS = (0, 1, 2, 3, 4)

@dataclass(frozen=True)
class Config:
    dataset: str = "SyntheticControl"
    seed: int = 0
    variant: str = VARIANTS[0]
    latent_dim: int = 256
    encoder_blocks: int = 3
    decoder_blocks: int = 3
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    pretrain_epochs: int = 50
    joint_epochs: int = 100
    batch_size: int = 128
    small_n_batch_rule: str = "min(batch_size,N)"
    drop_last: bool = True
    num_workers: int = 4
    pretrain_optimizer: str = "Adam"
    pretrain_lr: float = 0.001
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    optimizer: str = "SGD"
    joint_lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    scheduler: str = "StepLR"
    scheduler_step: int = 5
    scheduler_gamma: float = 0.1
    tau_initial: float = 10.0
    beta: float = 0.65
    tau_minimum: float = 0.01
    sigma: float = 1.0
    instance_temperature: float = 1.0
    cluster_temperature: float = 1.0
    alpha: float = 1.0
    stopping_tolerance: float = 0.001
    stopping_first_epoch: int = 2
    normalization: str = "per-series-z-ddof0"
    augmentation_order: tuple = ("sign", "equal-segment-permutation", "cubic-time-warp")
    sign_probability: float = 0.5
    max_segments_exclusive: int = 5
    warp_mode: str = "HistoricalWarp"
    warp_sigma: float = 0.2
    warp_knots: int = 4
    augmentation_snapshots: int = 2
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 1
    kmeans_algorithm: str = "lloyd"
    kmeans_max_iter: int = 300
    kmeans_tol: float = 1e-4
    metrics: tuple = ("ARI", "NMI_arithmetic", "RI", "ACC")
    instance_norm_eps: float = 1e-12
    legacy_cosine_eps: float = 1e-8
    noresoftmax_entropy_eps: float = 1e-8
    noresoftmax_count_norm_floor: float = 1.0
    self_mask: float = 1e9
    gradient_epochs: tuple = (0, 1, 4, 9, 19, 39, 59, 79, 99)
    gradient_diagnostics: bool = True
    near_collapse_fraction: float = 0.98
    device: str = "auto"

    def validate(self):
        if self.variant not in VARIANTS:
            raise ValueError("Only the three Audit 4 variants are supported")
        if self.seed < 0 or self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Invalid seed or device")
        if min(self.latent_dim, self.encoder_blocks, self.decoder_blocks,
               self.batch_size, self.joint_epochs, self.pretrain_epochs) < 1:
            raise ValueError("Positive architecture/budget fields required")
        if self.sigma <= 0 or self.tau_minimum <= 0 or self.num_workers < 0:
            raise ValueError("Invalid numeric configuration")
        expected_warp = {"HistoricalWarp": "cubic-time-warp", "MonotoneWarp": "monotone-time-warp"}
        if self.warp_mode not in expected_warp or self.warp_sigma < 0 or self.warp_knots < 0:
            raise ValueError("Invalid warp mode or parameters")
        if self.warp_mode == "MonotoneWarp" and self.variant != VARIANTS[0]:
            raise ValueError("Warp validation supports Legacy-Clean only")
        if (self.normalization != "per-series-z-ddof0" or not self.drop_last or
                self.augmentation_snapshots != 2 or
                tuple(self.augmentation_order) != ("sign", "equal-segment-permutation", expected_warp[self.warp_mode]) or
                self.pretrain_optimizer != "Adam" or self.optimizer != "SGD" or
                self.scheduler != "StepLR" or self.sign_probability != 0.5):
            raise ValueError("Unsupported methodological change")
        return self

    def to_dict(self):
        return asdict(self)

    def initialization_id(self):
        # Exclude ONLY variant: each paired arm shares the same full preparation.
        values = self.to_dict()
        values.pop("variant")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

def from_dict(values):
    values = dict(values)
    for key in ("adam_betas", "augmentation_order", "metrics", "gradient_epochs"):
        if key in values:
            values[key] = tuple(values[key])
    return Config(**values).validate()

def load_campaign(path):
    campaign = json.loads(Path(path).read_text(encoding="utf-8"))
    base = from_dict(campaign["base"])
    return campaign, base

def assert_pilot_contract(config):
    expected = replace(Config(), dataset=config.dataset, seed=config.seed,
                       variant=config.variant, device=config.device)
    if config.to_dict() != expected.to_dict():
        raise ValueError("Pilot config differs from frozen Audit 4 contract")
    if config.dataset not in PILOT_DATASETS or config.seed not in PILOT_SEEDS:
        raise ValueError("Not an Audit 4 pilot combination")

def planned_runs():
    names = ("legacy-clean", "no-resoftmax", "aligned-init")
    return [{"dataset": dataset, "variant": variant, "seed": seed,
             "config_id": "pilot/" + name + ".json"}
            for dataset in PILOT_DATASETS
            for variant, name in zip(VARIANTS, names)
            for seed in PILOT_SEEDS]

def validate_manifest(manifest):
    runs = manifest["runs"]
    expected = {(d, v, s) for d in PILOT_DATASETS for v in VARIANTS for s in PILOT_SEEDS}
    actual = [(r["dataset"], r["variant"], r["seed"]) for r in runs]
    if len(actual) != 120 or len(set(actual)) != 120 or set(actual) != expected:
        raise ValueError("Manifest must contain exactly the 120 unique paired runs")
    for r in runs:
        idx = VARIANTS.index(r["variant"])
        expected_id = ("pilot/legacy-clean.json", "pilot/no-resoftmax.json", "pilot/aligned-init.json")[idx]
        if r["config_id"] != expected_id:
            raise ValueError("Wrong config identifier")
    return runs
