"""Canonical headerless UCR data; separate training and evaluation capabilities.

Outer population z-score matches the verified public LoSTer utility, copied
without importing any LoSTer model. Native centroid/distance transforms remain.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .util import ContractError, array_hash, digest, file_hash

@dataclass(frozen=True)
class TrainingInput:
    dataset: str
    raw: object
    sample_ids: tuple
    k: int
    input_hashes: dict
    splits: dict

    @property
    def N(self):
        return self.raw.shape[0]

    @property
    def L(self):
        return self.raw.shape[1]

    def identity(self):
        return {"dataset": self.dataset, "N": self.N, "L": self.L, "k": self.k,
                "sample_ids": list(self.sample_ids), "sample_id_sha256": digest(list(self.sample_ids)),
                "input_hashes": self.input_hashes, "raw_array_sha256": array_hash(self.raw),
                "splits": self.splits, "pool_order": "TRAIN then TEST"}

@dataclass(frozen=True)
class EvaluationInput:
    training_identity: dict
    labels: object

def normalize_rows(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise ContractError("Nonempty fixed-length [N,L>=2] required")
    if not np.isfinite(x).all():
        raise ContractError("No imputation/deletion: inputs must be finite")
    std = x.std(axis=1, ddof=0)
    if not np.isfinite(std).all() or np.any(std == 0):
        raise ContractError("Undefined z-score for zero/nonfinite variance")
    normalized = (x - x.mean(axis=1, keepdims=True)) / std[:, None]
    if not np.isfinite(normalized).all():
        raise ContractError("Nonfinite normalization")
    return normalized

def _read_split(path):
    labels, rows = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(row) < 3 or not row[0].strip():
                raise ContractError("Invalid headerless record at row %s" % number)
            try:
                rows.append([float(v) for v in row[1:]])
            except ValueError as error:
                raise ContractError("Non-numeric sequence") from error
            labels.append(row[0])
    if not rows or len({len(r) for r in rows}) != 1:
        raise ContractError("Empty/ragged split")
    x = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ContractError("Nonfinite series; no sample repair")
    return labels, x

def _load(data_root, name, expected):
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ContractError("Invalid dataset identifier")
    labels, arrays, ids, hashes, splits = [], [], [], {}, {}
    for split in ("TRAIN", "TEST"):
        p = Path(data_root) / name / (name + "_" + split + ".tsv")
        before = file_hash(p)
        y, x = _read_split(p)
        if file_hash(p) != before:
            raise ContractError("Input changed during ingestion")
        labels.extend(y)
        arrays.append(x)
        ids.extend("%s/%s/%d" % (name, split, i) for i in range(len(x)))
        hashes[split] = before
        splits[split] = {"path": str(p.resolve()), "records": len(x), "sha256": before}
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ContractError("TRAIN/TEST length mismatch")
    y = np.asarray(labels)
    try:
        y = y.astype(np.float64)
    except ValueError:
        if any(v.strip().lower() in ("nan", "inf", "-inf", "") for v in y):
            raise ContractError("Missing label metadata")
    else:
        if not np.isfinite(y).all():
            raise ContractError("Nonfinite oracle label")
    _, encoded = np.unique(y, return_inverse=True)
    raw = np.concatenate(arrays)
    raw.setflags(write=False)
    encoded = encoded.astype(np.int64)
    encoded.setflags(write=False)
    training = TrainingInput(name, raw, tuple(ids), len(np.unique(encoded)), hashes, splits)
    if expected:
        actual = {"N": training.N, "L": training.L, "k": training.k,
                  "TRAIN": len(arrays[0]), "TEST": len(arrays[1])}
        for key, val in expected.items():
            if key in actual and actual[key] != val:
                raise ContractError("Canonical metadata mismatch: " + key)
        if "input_hashes" in expected and expected["input_hashes"] != hashes:
            raise ContractError("Canonical file hashes mismatch")
    return training, encoded

def load_training_input(data_root, name, expected=None):
    # Only oracle k survives ingestion. No labels/class membership in this object.
    training, _ = _load(data_root, name, expected)
    return training

def load_evaluation_input(data_root, name, expected=None):
    training, labels = _load(data_root, name, expected)
    return EvaluationInput(training.identity(), labels)
