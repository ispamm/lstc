# LoSTer 2026 experimental harness

This package implements [Audit 4](../../analysis/audits/04-method-and-experiment-plan.md). **No real pilot numbers have been generated.** [models/LoSTer](../../models/LoSTer/) remains a frozen reference; production code does not import it.

Current verdict (2026-10-05 environment remediation): **READY**. The dedicated Python 3.9 environment passed all 52 tests without skips, all three CPU synthetic smoke checks, and all three tiny CUDA smoke checks with exact checkpoint prediction round-trips. GTX 1080 CUDA tensor execution was confirmed. See [Audit 5](../../analysis/audits/05-implementation-readiness.md), which preserves the initial NOT READY finding and records the remediation. The real pilot remains unexecuted and requires separate authorization.

## Three methods, one implementation

- **LoSTer-Legacy-Clean:** historical active dense architecture, independent views/centers, two fixed augmentation snapshots, sigma=1 RBF logits, hard ST Gumbel, original reductions/coefficients, extra softmax and smoothed entropy.
- **LoSTer-NoResoftmax:** cluster contrast and entropy receive Q_ST directly, using Audit 4's entropy log floor1e-8 and count-unit column norm floor1. Other terms/settings are identical.
- **LoSTer-AlignedInit:** a label-free contingency of paired X/A_init assignments determines one maximum-agreement augmented-center permutation after independent KMeans fits and before optimizer construction. No coordinate matching, shared weights/centers or online matching.

Training batches contain no labels. Oracle k comes explicitly from class metadata; metrics are evaluation-only. Stopping compares consecutive full-pool deterministic original assignments, starting after epoch two, with strict fraction<0.001 or100 epochs. Monitoring never reseeds/balances centers.

Only the active AE path is implemented. RevIN, global AE residual and dormant centers are absent. Fresh AE construction/reload at the preparation wrapper boundary retains the historical initialization draw order. Corrected cohorts and explicit iteration/runtime choices mean whole historical stochastic trajectories are not claimed identical. Numerical components and paired-arm replay are tested.

## Configurations and data

The three JSON files under configs/pilot contain complete explicit settings. The tracked manifest has **120 unique runs**: eight datasets Ã— three variants Ã— seeds0..4, plus config identifiers, report hash and canonical N/L/k.

Datasets: SyntheticControl, Beef, ECG200, OSULeaf, ShapesAll, SemgHandMovementCh2, CinCECGTorso, StarLightCurves.

configs/final/benchmark.json preserves CORE40, EXTENDED44, development8, primary36 and final seeds100..104. The selected method is null until the registered decision; this is not a final execution manifest.

UCR files must be canonical headerless TSVs under <data-root>/<dataset>/<dataset>_{TRAIN,TEST}.tsv. Explicit tab parsing retains all records in TRAIN-then-TEST order. Per-series z-normalization uses population SD. NaN, infinity, zero variance, inconsistent length/count and non-numeric samples fail without repair. Counts, classes, oracle k and file hashes enter provenance.

Batch size is min(128,N), shuffled/drop_last=True. Augmentation uses float64 normalized inputs and float32 model arrays, with two separately precomputed snapshots. Index permutation preserves the historical random draws/distribution for ragged segment lengths in modern NumPy. Non-monotone warp coordinates are diagnosed, not sorted/repaired/resampled.

## Commands

From the repository root, use `$pilotPython` from the dedicated environment setup below:

~~~powershell
& $pilotPython experiments_new/loster2026/scripts/verify_environment.py
& $pilotPython experiments_new/loster2026/scripts/run_tests.py
& $pilotPython experiments_new/loster2026/scripts/smoke_test.py
& $pilotPython experiments_new/loster2026/scripts/run_pilot.py
~~~

run_pilot.py is **dry-run by default**: it prints the plan without loading data/models or creating run outputs. The smoke script uses only16 artificial length24 series, width8, one pretraining epoch per view and at most two joint epochs, for all three variants in a temporary directory. It has no scientific quality interpretation.

Only after a future explicit authorization and readiness check:

~~~powershell
& $pilotPython experiments_new/loster2026/scripts/run_pilot.py --execute --data-root "<canonical UCR root>" --device cuda
~~~

Optional --run-index 0..119 selects one planned run; omission selects all120. --device cpu/cuda are supported; unavailable requested CUDA fails. No real-pilot command was executed here.

~~~powershell
& $pilotPython experiments_new/loster2026/scripts/aggregate_pilot.py
~~~

Aggregation is read-only, reports coverage/duplicates/failures, seed means/sample SD and Audit4's numeric gates/tie order. It cannot silently aggregate surviving pairs or freeze a method. Its recommendation still requires the conceptual/hypothesis review in Audit4.

Tests use standard-library unittest; pytest is optional. Missing scientific dependencies produce explicit skips, which do not imply readiness. Fixtures/checkpoints/synthetic outputs use temporary directories. Read-only AST extraction accesses legacy reference definitions without importing CUDA-dependent launchers or writing legacy bytecode caches.

## Environment

Use **64-bit CPython 3.9**, validated here with **3.9.13**. This is the controlled environment for the 2026 reruns, preserving the audited historical library semantics where possible; it does **not** recreate the original historical runtime exactly. Leave existing environments and the default interpreter unchanged. Keep the virtual environment, package cache and temporary validation artifacts outside the Google Drive repository; do not relocate the repository.

[requirements-pilot.txt](requirements-pilot.txt) pins the five direct dependencies: NumPy 1.22.4, pandas 1.5.3, scikit-learn 1.4.1.post1, SciPy 1.13.0 and PyTorch 1.13.1+cu117. pandas enables the literal historical-header regression. [environment-lock.txt](environment-lock.txt) also pins the six runtime dependencies and SHA256 hashes of the validated Windows CPython 3.9 wheels. Use that lock for reproducible Windows installations. Its wheel hashes are specific to this platform/interpreter; another platform needs separate validation.

From the repository root, replace the local-directory placeholder with an external, non-synced location. Discover the existing base interpreter instead of using the default `python` alias:

~~~powershell
py -0p
$pilotBasePython = (& py -3.9 -c "import sys; print(sys.executable)").Trim()
& $pilotBasePython -c "import sys,struct; print(sys.version); print(sys.executable); assert sys.version_info[:2] == (3,9) and struct.calcsize('P') == 8"
$pilotLocalRoot = '<local non-synced working directory>'
$pilotVenvRoot = Join-Path $pilotLocalRoot 'venv'
if (Test-Path -LiteralPath $pilotVenvRoot) { throw 'Use a fresh environment location' }
& $pilotBasePython -m venv $pilotVenvRoot
$pilotPython = Join-Path $pilotVenvRoot 'Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path $pilotLocalRoot 'pip-cache'
$env:TEMP = Join-Path $pilotLocalRoot 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
& $pilotPython -m pip install -r experiments_new/loster2026/environment-lock.txt
& $pilotPython -m pip check
& $pilotPython experiments_new/loster2026/scripts/verify_environment.py
~~~

The bundled pip 22.0.4 and setuptools 58.1.0 were sufficient; no build-tool upgrade or pytest installation was needed. If HTTPS certificate verification fails, use an approved CA bundle through process-local `PIP_CERT`; keep it outside the repository and retain certificate verification. This machine's successful install used its existing trusted Windows CA certificates plus pip's bundled CA certificates. For an existing approved local bundle, set `$env:PIP_CERT = Join-Path $pilotLocalRoot 'windows-ca-bundle.pem'` before installation. No system trust store was changed.

For CPU-only validation with the same CUDA wheel, hide CUDA explicitly before starting Python, then restore visibility for GPU checks:

~~~powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
& $pilotPython experiments_new/loster2026/scripts/verify_environment.py
& $pilotPython experiments_new/loster2026/scripts/run_tests.py
& $pilotPython experiments_new/loster2026/scripts/smoke_test.py
Remove-Item Env:\CUDA_VISIBLE_DEVICES
& $pilotPython experiments_new/loster2026/scripts/verify_environment.py
& $pilotPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.arange(4.,device='cuda'); assert (x*x).sum().item()==14.; torch.cuda.synchronize()"
& $pilotPython experiments_new/loster2026/scripts/run_pilot.py
~~~

The official [PyTorch CUDA 11.7 wheel index](https://download.pytorch.org/whl/cu117/torch/) supplies the pinned CUDA build. This environment verified actual CUDA tensor execution and all three tiny synthetic CUDA pipelines on NVIDIA GeForce GTX 1080, driver 552.22, runtime 11.7. `--device cpu` supports CPU execution; `--device auto` falls back to CPU when CUDA is unavailable. Explicit `--device cuda` fails if unavailable. **The 120-run real pilot must wait for confirmed GPU execution and separate authorization.** The default smoke script stays CPU-only.

The environment script reports imports/device information; its conservative `reference_versions_verified=false` field does not compare pins or certify the suite. The version comparison, complete tests and CPU/CUDA smoke evidence are recorded in Audit 5. Optional pytest absence is harmless because every test uses standard-library unittest.

## Outputs and pairing

Generated artifacts/logs/results are ignored; their README files remain tracked.

~~~text
artifacts/<dataset>/<variant>/seed-<n>/manifest.json
artifacts/<dataset>/<variant>/seed-<n>/complete.pt
logs/<dataset>/<variant>/seed-<n>/epochs.jsonl
results/<dataset>/<variant>/seed-<n>/metrics.json
artifacts/initialization/<dataset>/seed-<n>/<preparation-key>.pt
~~~

Existing run folders/checkpoints are not overwritten. The shared initialization artifact contains both pretrained AEs/fitted centers, A_train/A_init, initial assignments/diagnostics and the RNG boundary. Keys include config excluding variant only, input/source hashes; cohort/device/version are validated. Every arm restores the boundary. AlignedInit changes only its prescribed permutation.

Complete checkpoints contain both AEs, both ACTIVE centers, full config/identity, final epoch/tau, optimizer/scheduler, original/augmented assignments, metrics, provenance and RNG. Strict reload rejects missing active centers and verifies identical deterministic assignments. These are locally generated full-state artifacts.

Manifests omit usernames, home/executable paths, hostname and secrets. They record UTC time, Git SHA/dirty state, source hash, Python/library/OS/device, config and cohort metadata. Failed runs remain labeled failed.

## Diagnostics and timing

Epoch JSONL separates all objective terms, tau/LR, hard occupancy/entropy/utilization/change, center usage/dead duration/displacement, S0/RBF entropy, batch counts, correspondence and stopping. Sparse component gradients use the first already-existing batch at epochs0,1,4,9,19,39,59,79,99; no additional stochastic forward, RNG consumption or .grad mutation occurs.

Instrumentation synchronizes CUDA timing boundaries and records stage/full operational time, loading/normalization, inference throughput, active/registered/deployed parameters, host memory and CUDA peaks. Cached preparation and stage-summed cost estimates are labeled. **Unit/smoke timing is not benchmark evidence.** Disposable warm-up and repeated whole-pool inference belong to later authorized measurements.

The dedicated environment has passed the readiness gates in Audit 5. No UCR download, real-pilot training or manuscript modification occurred. No implementation/environment-remediation changes have been committed or pushed.
