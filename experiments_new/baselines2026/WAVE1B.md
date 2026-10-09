# Wave 1B: R-Clustering and univariate FASA

This extension admits exactly the two explicitly authorized conditional methods.
Original Wave1 modules/configs/README and the independent evaluator are unchanged.
Historical Audit08 and registry remain immutable. Real-fit scope: ECG200, seed0,
exactly one native fit per method. CORE-44 is not authorized.

R-Clustering: jorgemarcoes/R-Clustering, submission commit
3ed571eebe9eb8d399a0c995a6917f62d838f0fc (GPL-3.0).
Notebook cell10 is materialized verbatim only on G:, adding its missing global
NumPy import. Native cache/signatures/fastmath/parallel semantics remain.
Only scientific cell12 assignments execute in their original order; archive
loops, installs/downloads, loaders, metrics/labels and spreadsheet writes do not.
Capturing the native KMeans constructor retains its estimator with unchanged
arguments. Common canonical z-normalization is the declared task adaptation;
native float32 kernels/features, StandardScaler, both native auto PCA fits and
KMeans n_init10/default global RNG behavior remain. Requested500 means420.
The native argmax(explained_variance_ratio<.01) may return zero; report protocol
incompatibility, never clamp/refit. Seed NumPy and Numba compiled streams to the
root seed, without adding new PCA/KMeans head/restart policies.

FASA: TheDatumOrg/MUFASA, commit9c05d415efee81fca1a87c1e91623db613ea2ecc.
Import only external Clustering/FASA_I_I/FASA_I_I.py. This is univariate FASA,
not multivariate MUFASA. No scientific source is vendored into this repository
and permission to redistribute is not asserted; license remains NOT STATED.
Keep native FFT/SBD/aligned centroids, random partition/zero centers,
max100/stable labels and empty-cluster recovery. Native den=inf/zscore/nan_to_num
handling stays; warnings/zero centers are reported, never add epsilon/repair.
Explicit scoped NumPy seed controls initialization; itr is not a seed.
Member indices must cover every canonical record once; scatter to ordered IDs.
Native FASA has no out-of-sample predict API: time membership extraction and
replay saved member state, without inventing an alternative classifier.

New isolated CPU environment on G:
baselines/envs/wave1b-cpu-py39-v1, Python3.9.13, full20-package lock in
configs/requirements-wave1b.lock.txt. NumPy1.26.4/SciPy1.13.1/sklearn1.4.1.post1/
Numba0.60.0/tqdm4.67.1. Native array inputs bypass unrelated aeon/HDF5 loaders.
No global, original Wave1 or LoSTer package installations.

Use new environment Python with scripts/validate_wave1b.py:
--source-root, new G: --output-root, --wave1-python pointing to original locked
Wave1 Python. New/native/common contract tests run first; all57 unchanged Wave1
tests then run in their original environment with new external cache/log/temp
paths. Two same-seed and one changed-seed full-default tiny fits per method
verify stochastic state, centers/labels and checkpoint replay. Gate pins the
current adapter tree and new environment fingerprints. No real dataset fit.

scripts/run_wave1b.py accepts --source-root --data-root --gate --output-root.
It performs one ECG200 seed0 fit per method, independent central evaluation,
fresh-process checkpoint replay and no-fit resume. Preserve failures and stop
that method without scientific rescue. Do not launch into another root after
using the authorized two fits; identical existing identities can be reused.

Common prediction schema and evaluator unchanged: ARI, arithmetic NMI, RI and
rectangular Hungarian ACC after immutable export; no score-based choices.
Full CPU pipeline includes normalization, native feature/scaling/PCA/clustering/
extraction and cold JIT; source preparation overhead is included. Excludes
raw decoding/checkpoint/export/replay/evaluation. FASA inference_seconds=null
because only native membership extraction exists; its stage is recorded.
CPU threads1, sampled process-tree RSS20ms, all artifacts/caches/checkouts on G:.
Timing is integration evidence, not publication efficiency.

New extension stays uncommitted: adapter_commit=null, parent SHA plus per-file/
tree hash and archived source bundle identify executed bytes. Old Wave1 output
bundles retain their historical hashes and are not modified.

Windows cache-path correction: external derived-source directory uses the first16
SHA256 characters; full64 content hashes and exact source verification remain.
This avoids MAX_PATH on Numba temporary files without changing scientific code,
cache=True, seeds, numeric precision, PCA/KMeans or timing boundaries. The initial
ECG200 dispatch failed during eager JIT before feature fitting; retain its logs.
Only R may consume its still-unexecuted native fit after renewed validation.
