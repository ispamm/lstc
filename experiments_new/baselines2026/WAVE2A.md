# Wave 2A: TS2Vec + k-means and FCACC

Exactly one full native ECG200 seed0 TS2Vec fit and one full native
SyntheticControl seed0 FCACC fit are admitted after separate validation gates.
An exclusive external per-method authorization ledger prevents a second fit.
No CORE-44, Final36, other real-data fit, TFMCC implementation or capacity rescue.

Official external source pins:
TS2Vec b0088e14a99706c05451316dc6db8d3da9351163 (MIT);
FCACC 78b5e5a138ed8c83fea64e5b6668a5cded79d2dc (license NOT STATED).
Both clean trees and every Git blob are verified; no scientific source is vendored.

TS2Vec uses existing Python3.7.9 in a new isolated G: environment with
torch1.8.1+cu111, numpy1.19.2, scipy1.6.1, sklearn0.24.2 and pandas1.0.1.
Anaconda Library/bin must be prepended to the worker PATH for its SSL DLLs.
The importlib-metadata6.7 backport supplies Python3.7's missing stdlib API to the
unchanged common provenance module. Forecast-only Bottleneck/statsmodels are
outside the encoder import closure. Core author scientific pins are retained.
Lloyd is spelled algorithm='full' by sklearn0.24.2; this is its native Lloyd
implementation, an API-name mapping, with explicit ten restarts/300/tol1e-4.
Official CLI B8, d320/h64/depth10/lr.001/maxLength3000/unit0, native 200/600
input.size budget, native augmentations/objectives/SWA and full_series extraction
are unchanged. A deterministic independent kmeans phase seed feeds one head.

FCACC uses existing Python3.9.13 in a new isolated G: environment.
Its README recommends Python3.8.10 (not present) and core numpy1.21.4,
pandas2.0.3, torch1.10.0+cu113, sklearn1.3.2, scipy1.10.1 and matplotlib3.5.0.
Those core pins, plus Pillow8.4.0, are retained; unrelated notebooks, torchvision,
image and visualization extras are omitted from the verified import closure.
This is a validated compatible execution environment, not a complete author lock.
Source KMeans n_init='warn' resolves to native10, random_state0 is retained.
Native runner B8/d64/h64/depth10/pre100/joint30/.001/.0001/m1.5/w_c.2/hard_w.2
and all other class defaults remain. DataTransform actually calls jitter with
sigma.8; the released implementation is preserved even where audit prose calls
the third view scaled. No mathematical objective repair is introduced.
Native pooled fuzzy encodings use the averaged net; the raw encoder is separately
trained and reloaded as released. Per-epoch diagnostic forward passes, loader
shuffle/RNG, KMeans and mode transitions are retained by AST filtering only the
label metrics, metric prints and feature-file exports. Constant dummy targets
satisfy the native loader tuple API; real class memberships are absent.
Final native raw-forward/allocation behavior is retained, including unused
N,L,d float64 arrays, and predictions are scattered by row indices with strict
unique/complete coverage. No memory optimization or reduced scientific budget.

Common dataset, predictions, evaluator, timing and provenance remain unchanged.
Checkpoint saves raw and active averaged states, n_averaged/modes, native fuzzy
centers or head state, configuration/seed/input identity, ordered representation
and prediction exports. Hash/strict-key checks reject wrong or missing active
states. Reload must reproduce both representations and predictions exactly.
The runner counts preprocessing, initialization, all native training and
diagnostic effects, inference/head and canonical validation; CUDA synchronizes
timing boundaries. CPU tree RSS and GPU allocated/reserved peaks are recorded,
with sampled nvidia-smi utilization/device-memory snapshots. Integration timings
are not publication efficiency and do not forecast the complete panel.

Use the appropriate new environment Python with scripts/wave2a.py:
validate --method ... --source-root G:/.../baselines/src --output a NEW external
validation folder; run additionally requires --data-root canonical UCR,
--gate that method's passing gate.json and --output external runs root.
replay --artifact completed folder does no training. evaluate independently
reloads immutable predictions and real labels only after export.
GPU workers must run sequentially. Tests alone use tiny generated N16,L16 inputs
with explicit three TS2Vec steps or FCACC one+two epochs; these are software tests,
not real-data scientific fits or an accuracy tuning recipe.
All new Wave2A files remain uncommitted. Full exact locks are under configs/.

Native TS2Vec init_dl_program offsets are retained: root0 means Python0, NumPy1,
Torch CPU2 and CUDA3. Deterministic cuDNN is explicitly enabled for fixed-seed
replay; benchmark=False and TF32=False follow the native helper. No training
score was available when this source correspondence was frozen.
