"""Evaluation only (Audit 1 F23/F24, Audit 4 explicit arithmetic NMI/ACC)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def contingency(y, predicted):
    if len(y) != len(predicted) or len(y) == 0:
        raise ValueError("Nonempty aligned labels and predictions required")
    _, a = np.unique(y, return_inverse=True)
    _, b = np.unique(predicted, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(counts, (a, b), 1)
    return counts

def accuracy(y, predicted):
    counts = contingency(y, predicted)
    rows, cols = linear_sum_assignment(-counts)
    return float(counts[rows, cols].sum() / len(y))

def rand_index(y, predicted):
    # Integer pair counts implement Rand score explicitly, including N=1.
    counts = contingency(y, predicted)
    choose2 = lambda n: int(n) * (int(n) - 1) // 2
    total = choose2(len(y))
    if total == 0:
        return 1.0
    together = sum(choose2(v) for v in counts.flat)
    rows = sum(choose2(v) for v in counts.sum(1))
    columns = sum(choose2(v) for v in counts.sum(0))
    apart = total - rows - columns + together
    return float((together + apart) / total)

def evaluate(y, predicted):
    contingency(y, predicted)
    return {"ARI": float(adjusted_rand_score(y, predicted)),
            "NMI_arithmetic": float(normalized_mutual_info_score(y, predicted, average_method="arithmetic")),
            "RI": rand_index(y, predicted), "ACC": accuracy(y, predicted)}
