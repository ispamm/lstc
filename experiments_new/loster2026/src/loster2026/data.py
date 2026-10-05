"""Canonical headerless UCR ingestion, legacy data/load_data.py (F11/F12/F30).
CSV parsing is explicit and avoids a pandas runtime dependency; no rows are repaired.
"""
from __future__ import annotations
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .reproducibility import sha256_file

@dataclass
class Dataset:
    series: object       # float64 normalization/augmentation, as historical UCR.
    labels: object       # Evaluation only; not returned by training batches.
    metadata: dict
    preprocessing_seconds: float = 0.0
    _x: object = field(default=None, init=False, repr=False)

    @property
    def x(self):
        if self._x is None:
            self._x = self.series.astype(np.float32)
        return self._x

    @property
    def k(self):
        return self.metadata["k"]

def normalize_rows(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("Nonempty fixed-length [N,L>=2] data required")
    if not np.isfinite(x).all():
        raise ValueError("NaN/infinite data: no imputation or deletion permitted")
    std = x.std(axis=1, ddof=0)
    if not np.isfinite(std).all() or np.any(std == 0):
        raise ValueError("Zero/nonfinite variance: undefined z-normalization")
    normalized = (x - x.mean(axis=1, keepdims=True)) / std[:, None]
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced nonfinite values")
    return normalized

def _read_headerless(path):
    labels, rows = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(row) < 3 or not row[0].strip():
                raise ValueError("Invalid headerless record at row %d" % number)
            try:
                rows.append([float(value) for value in row[1:]])
            except ValueError as error:
                raise ValueError("Non-numeric sample at row %d" % number) from error
            labels.append(row[0])
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Empty or variable-length TSV")
    return labels, np.asarray(rows, dtype=np.float64)

def load_ucr(data_root, name, expected=None):
    preprocessing_start = time.perf_counter()
    base = Path(data_root) / name
    train_path = base / (name + "_TRAIN.tsv")
    test_path = base / (name + "_TEST.tsv")
    yt, xt = _read_headerless(train_path)
    yv, xv = _read_headerless(test_path)
    if xt.shape[1] != xv.shape[1]:
        raise ValueError("TRAIN and TEST sequence lengths differ")
    labels = np.asarray(yt + yv)
    try:
        numeric_labels = labels.astype(np.float64)
    except ValueError:
        numeric_labels = None
    if numeric_labels is not None:
        if not np.isfinite(numeric_labels).all():
            raise ValueError("Nonfinite oracle labels")
        labels = numeric_labels
    elif any(v.strip().lower() in ("nan", "inf", "-inf", "") for v in labels):
        raise ValueError("Missing/nonfinite oracle label")
    classes, encoded, counts = np.unique(labels, return_inverse=True, return_counts=True)
    series = normalize_rows(np.concatenate([xt, xv], axis=0))
    metadata = {"dataset": name, "train_count": len(xt), "test_count": len(xv),
                "N": len(series), "L": series.shape[1], "k": len(classes),
                "class_counts": [{"label": str(c), "count": int(n)} for c, n in zip(classes, counts)],
                "oracle_k": True, "label_use": "oracle k and evaluation only",
                "pool_order": "TRAIN then TEST, original row order",
                "normalization": "per-series-z-ddof0",
                "input_sha256": {"TRAIN": sha256_file(train_path), "TEST": sha256_file(test_path)}}
    if expected:
        for key in ("N", "L", "k", "train_count", "test_count"):
            if key in expected and metadata[key] != expected[key]:
                raise ValueError("Canonical %s mismatch: %s != %s" % (key, metadata[key], expected[key]))
    return Dataset(series, encoded.astype(np.int64), metadata,
                   preprocessing_seconds=time.perf_counter() - preprocessing_start)
