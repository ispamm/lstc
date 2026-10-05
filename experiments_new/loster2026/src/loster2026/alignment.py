"""Audit 4 C2-once extends legacy main.py KMeans initialization (F09).
Not center-coordinate matching and never part of backprop.
"""
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

def match_initial_centers(original_labels, augmented_labels, centers_augmented):
    original_labels = np.asarray(original_labels, dtype=np.int64)
    augmented_labels = np.asarray(augmented_labels, dtype=np.int64)
    k = len(centers_augmented)
    if (original_labels.shape != augmented_labels.shape or len(original_labels) == 0 or
            min(original_labels.min(), augmented_labels.min()) < 0 or
            max(original_labels.max(), augmented_labels.max()) >= k):
        raise ValueError("Paired in-range initialization assignments required")
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (original_labels, augmented_labels), 1)
    # Deterministic tie policy: ascending row/column input, fixed SciPy solver
    # version stored with permutation; no jitter changes the primary objective.
    rows, columns = linear_sum_assignment(-counts)
    permutation = np.empty(k, dtype=np.int64)
    permutation[rows] = columns
    info = {"contingency": counts.tolist(), "permutation": permutation.tolist(),
            "same_index_agreement_before": float(np.trace(counts) / len(original_labels)),
            "same_index_agreement_after": float(counts[rows, columns].sum() / len(original_labels)),
            "tie_policy": "ascending rows/columns; deterministic versioned SciPy solver; no cost perturbation",
            "scipy_version": scipy.__version__, "matching_calls": 1,
            "input_views": "X and A_init", "ground_truth_used": False}
    return np.asarray(centers_augmented)[permutation].copy(), info
