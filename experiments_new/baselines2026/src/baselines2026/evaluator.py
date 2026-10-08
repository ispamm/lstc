"""Independent evaluator v1: never called by a training adapter."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .datasets import load_evaluation_input
from .predictions import load_submission
from .util import file_hash, write_json_new

VERSION = "baselines2026.metrics.v1"

def contingency(y, predicted):
    if len(y) != len(predicted) or len(y) == 0:
        raise ValueError("Nonempty aligned labels/predictions required")
    _, a = np.unique(y, return_inverse=True)
    _, b = np.unique(predicted, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(counts, (a, b), 1)
    return counts

def metrics(y, predicted):
    c = contingency(y, predicted)
    rows, cols = linear_sum_assignment(-c)
    choose2 = lambda n: int(n) * (int(n) - 1) // 2
    total = choose2(len(y))
    together = sum(choose2(v) for v in c.flat)
    a = sum(choose2(v) for v in c.sum(1))
    b = sum(choose2(v) for v in c.sum(0))
    ri = (together + total - a - b + together) / total if total else 1.0
    return {"ARI": float(adjusted_rand_score(y, predicted)),
            "NMI_arithmetic": float(normalized_mutual_info_score(y, predicted, average_method="arithmetic")),
            "RI": float(ri), "ACC": float(c[rows, cols].sum() / len(y))}

def evaluate_file(path, data_root, output, expected=None):
    from .util import read_json
    p = read_json(path)
    data = load_evaluation_input(data_root, p["dataset"], expected)
    p = load_submission(path, data.training_identity)
    scores = metrics(data.labels, p["predicted_clusters"])
    result = {"evaluator_version": VERSION, "evaluator_source_sha256": file_hash(__file__),
              "submission_sha256": file_hash(path), "run_id": p["run_id"],
              "method": p["method"], "dataset": p["dataset"], "seed": p["seed"],
              "N": p["N"], "metrics": scores,
              "role": "evaluation only, after immutable assignment export"}
    write_json_new(output, result)
    return result
