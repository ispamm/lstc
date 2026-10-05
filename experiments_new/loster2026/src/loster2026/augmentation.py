"""Active rotation/permutation/time_warp, legacy data/augmentation.py (F13-F15).
No online draws, repairs or monotonicity enforcement; diagnostics do not alter draws.
"""
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline

def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def augment(series, config):
    x = np.asarray(series, dtype=np.float64)[:, :, None]
    rng = np.random  # Legacy RandomState global draw semantics, replayed per seed.
    flip = rng.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    axis = np.arange(x.shape[2])
    rng.shuffle(axis)
    x = flip[:, None, :] * x[:, :, axis]
    steps = np.arange(x.shape[1])
    segments = rng.randint(1, config.max_segments_exclusive, size=x.shape[0])
    permuted = np.zeros_like(x)
    for i, row in enumerate(x):
        if segments[i] > 1:
            splits = np.array_split(steps, segments[i])
            # Index permutation preserves legacy distribution/RNG consumption and
            # handles ragged equal-segment lengths in modern NumPy.
            order = rng.permutation(len(splits))
            permuted[i] = row[np.concatenate([splits[j] for j in order])]
        else:
            permuted[i] = row
    knots = config.warp_knots + 2
    multipliers = rng.normal(1.0, config.warp_sigma, size=(len(x), knots, 1))
    knot_steps = np.linspace(0, x.shape[1] - 1.0, knots)
    ret = np.zeros_like(x)
    minima, scales, invalid_steps, clipped_counts = [], [], [], []
    for i, row in enumerate(permuted):
        warped = CubicSpline(knot_steps, knot_steps * multipliers[i, :, 0])(steps)
        if warped[-1] == 0 or not np.isfinite(warped).all():
            raise ValueError("Nonfinite/zero-endpoint legacy warp; no resampling")
        scale = (x.shape[1] - 1) / warped[-1]
        before_clip = scale * warped
        coordinates = np.clip(before_clip, 0, x.shape[1] - 1)
        differences = np.diff(coordinates)
        ret[i, :, 0] = np.interp(steps, coordinates, row[:, 0])
        minima.append(float(differences.min()))
        scales.append(float(scale))
        invalid_steps.append(int((differences <= 0).sum()))
        clipped_counts.append(int(((before_clip < 0) | (before_clip > x.shape[1] - 1)).sum()))
    if not np.isfinite(ret).all():
        raise ValueError("Nonfinite augmented values; no silent repair")
    a = ret[:, :, 0].astype(np.float32)
    info = {"snapshot_sha256": array_hash(a), "finite": True,
            "minimum_coordinate_difference": min(minima),
            "row_minimum_coordinate_differences": minima, "endpoint_scales": scales,
            "nonincreasing_steps_per_row": invalid_steps,
            "nonincreasing_step_fraction": sum(invalid_steps) / (len(x) * (x.shape[1] - 1)),
            "affected_series_fraction": sum(n > 0 for n in invalid_steps) / len(x),
            "clipped_coordinates_per_row": clipped_counts}
    return a, info

def two_snapshots(series, config):
    train, train_info = augment(series, config)
    initial, initial_info = augment(series, config)
    return train, initial, {"A_train": train_info, "A_init": initial_info}
