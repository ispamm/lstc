# Baselines 2026: Wave 1

Independent infrastructure for exactly the three READY FOR IMPLEMENTATION
comparators in Audit 08: Euclidean k-means, author CPU k-Shape, aeon KASBA.
The registry is read-only; conditional/blocked methods are rejected.
CORE-44 is not authorized. The real-fit CLI is restricted to SyntheticControl,
development seed 0, one native fit per method. This is integration validation.

## Scientific identities and adapters

- Euclidean: scikit-learn1.4.1.post1, commit719c0c6036ac99d45ffb45ed7c8b4e4b221384b5.
  Lloyd, k-means++, n_init10, max300, tol1e-4; minimum inertia only.
- k-Shape: TheDatumOrg/kshape-python commit778e735624d15848556e4d23fe38c321d453937a,
  CPU class, random partition/zero centers/max100/stable assignment, n_jobs1.
  Native FFT/SBD/eigenvector centroids and empty-cluster handling are unchanged.
- KASBA: aeon1.1.0 commit0fed29f21908cfad2981ea645f51aaf018f0fe89,
  MSM c1, max300/tol1e-6, subset.5/step.05/decay.1, native elastic initialization,
  stochastic barycenters (inner cap50) and triangle pruning.

Only source-verified native implementations are called. Installed scientific
files must match the frozen Git blobs. k-Shape imports its external author
checkout directly, without CuPy or vendoring. Adapters reshape N,L into N,L,1
or N,1,L as required and never pass y. No scientific source is edited.

Canonical raw TSV records are TRAIN then TEST, headerless, all rows. TrainingInput
has raw float64 X, IDs, k, file/split/array hashes and no membership labels.
Ground truth is loaded independently by the evaluator only after prediction
export. Oracle k is the permitted unique class count, not training supervision.
Common outer per-series population z-score is Audit08's declared pooled task;
the verified public utility's semantics are copied with no LoSTer model imports.
Zero variance/nonfinite/ragged input is rejected, never repaired. Native centroid
normalization and MSM/FFT transforms remain. No LoSTer augmentation is supplied.

Predictions use baselines2026.predictions.v1 JSON: method/dataset/root seed,
canonical ordered sample_ids/predicted_clusters, N/L/k, input/raw/ID hashes,
source commit, config/environment/adapter hashes, run_id and SUCCESS status.
Duplicate/missing/reordered IDs, count/hash/provenance mismatch and nonfinite or
noninteger clusters are rejected. Natural empty clusters/collapse are reported
separately and never rejected just because fewer than k centers are occupied.
Files are exclusive-create, under G:, never inside Git.

ARI is primary, explicit arithmetic NMI secondary, pair-count RI and rectangular
Hungarian ACC descriptive. The evaluator has a version and source hash, accepts
the same schema for future LoSTer/baseline exports, and runs separately from fit.
Historical published scores are not read, altered or recomputed.

Manifests record native source/license/blob checks, full package lock/fingerprint,
Python and CPU/thread identity, raw/normalized/ID/config hashes, seeds/settings,
actual iterations, stages, sampled process-tree peak RSS, warnings, utilization,
native fitted-partition/fresh-inference agreement and checkpoint reload.
The current adapter is deliberately uncommitted: adapter_commit=null, explicit
parent Git SHA and content-tree hash, plus an archived source bundle identify
its exact bytes. Do not claim the parent commit contains the adapter.

Pipeline timing starts on the raw numeric pool before normalization and includes
native initialization/all internal restarts/refinement, fit and ordered full-pool
inference. Import/file decoding/source checks and export/checkpoint/evaluation
are separately excluded operational work. CUDA synchronization is an injectable
boundary hook, tested; all three current methods are CPU-only. BLAS/OpenMP/Numba
threads are1; k-Shape retains its native one-worker pool. Native JIT compilation
on a cold per-smoke cache is included. Smoke seconds are not publication-quality
efficiency. Sampled RSS may miss short peaks; GPU memory is not applicable.

Native fitted labels_ are the canonical transductive partition. A separate full
predict pass is timed and compared; if the native fitted partition differs from
fresh nearest-center assignment, preserve/disclose rather than silently replace.
Both stored labels and fresh inference must reload exactly. Existing identical
complete artifacts are validated and reused without a second fit; corrupted,
partial or stale identities are refused. Pickles are trusted self-produced
artifacts only, never a format to load untrusted submissions.

## Environment and operation

New isolated environment:
G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/envs/wave1-cpu-py39-v1
Python3.9.13; exact full closure in configs/requirements-wave1.lock.txt.
NumPy1.26.4/SciPy1.13.1/sklearn1.4.1.post1/pandas2.2.3/Numba.60/aeon1.1.
No global or LoSTer venv package changes. HTTPS certificate validation remains
enabled; a machine-trusted Windows CA bundle was used to resolve legacy pip TLS.

Sources live under G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/src.
Use the environment's Python for scripts/validate_wave1.py with --source-root
and a new external --output-root. It runs unit tests, native parity, then two
same-seed full-default synthetic fits per method; gate.json pins source-tree
and environment identities. No real UCR data is opened by validation.

After gate.json passes, scripts/run_wave1.py accepts --source-root, --data-root
(pointing at existing data/ucr), --gate and a new external --output-root.
It executes exactly one SyntheticControl seed0 fit per method, separate-process
central evaluation and a no-fit reuse check. Failed logs are retained; recipes
are not tuned to smoke scores. Other Development8 tasks and Final36 stay closed.

Failures are reported as SOURCE ISSUE, DEPENDENCY ISSUE, PROTOCOL INCOMPATIBILITY,
NUMERICAL FAILURE, RESOURCE FAILURE, ADAPTER BUG or UNKNOWN based on evidence.
No failed seed or score is silently substituted. Runtime exceptions retain a
failure.json; distinguish diagnosed errors from the fallback UNKNOWN class.
Wave2/blocked source conditions and publication statistics remain outstanding.
