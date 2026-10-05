# LoSTer 2026 Implementation Readiness

Date: 2026-10-05. Specification: [Audit 4](04-method-and-experiment-plan.md). Scope: isolated implementation and artificial regression checks only.

## Current verdict after environment remediation (2026-10-05)

**READY.** All 52 required unit/regression tests passed in one dedicated environment, with zero failures, errors or skips. All three CPU synthetic variant checks and all three tiny CUDA variant checks passed, including exact deterministic predictions after checkpoint reload. GTX 1080 CUDA tensor execution succeeded. The final static manifest check confirmed 120 unique, paired planned runs. Protected files and all scientific implementation/configuration/test files are unchanged.

The original implementation assessment below is retained as dated historical context. Its NOT READY finding, partial-environment counts and then-pending gates are superseded by the dated remediation section at the end of this same Audit 5. No Audit 6 was created. Readiness is permission to consider a separately authorized pilot, not a claim of scientific clustering quality or an executed pilot.

## Initial verdict (historical; before environment remediation, 2026-10-05)

**NOT READY for the real pilot.**

The three specified methods, configurations, persistence, monitoring and planned-run system are implemented. Across the existing partial environments, **51 of 52 distinct tests passed**, with **zero remaining failures**. The full synthetic end-to-end test is the one test not executed in any inspected environment. No interpreter contains both PyTorch and scikit-learn with the other runtime dependencies.

This is not a claim of complete pipeline validation or reproduction under the frozen reference versions. No packages were installed, no UCR dataset was downloaded or used, and none of the120 real pilot runs was executed.

## Preservation and Stage A

Starting status contained only the untracked Audit4 report. It was committed with exactly **Add LoSTer 2026 method and experiment plan** and pushed only to **origin revision-2026**.

- Commit: **4a813d9a136753522dd7925f5f6182a6725dfd05**.
- Annotated tag: **pre-pilot-2026**, on that commit.
- Tag message: **Frozen LoSTer revision state before 2026 experimental implementation**.
- Tag pushed only to origin; clean status verified before new implementation work.
- No implementation/Audit5 commit or push is made.

A preservation inventory of **300 pre-existing files outside the allowed edit scope** was captured after StageA, including content SHA256, path, byte length and mtime. Its aggregate hash was **ff1d9c9ee0ebfd200368b956b928dd888971d52cd86dc79f0157616696ca0ab2**. The same inventory/hash matched after implementation. The excluded edit scope is the new package, this report and .gitignore; .git is excluded from content preservation because the requested commit/tag necessarily changed Git state.

**models/LoSTer, all paper sources, Drafts, csv_logs, figures and legacy results remain untouched.** The explicit Git diff for models/LoSTer and paper was empty. No legacy output or checkpoint was overwritten.

## Files Created

There are **35 new package files**, plus this report. .gitignore is the only modified pre-existing working file, with eight appended ignore lines and its original bytes/newline style retained.

Package root: [experiments_new/loster2026](../../experiments_new/loster2026/).

| Area | New files |
|---|---|
| Documentation/output placeholders | README.md; artifacts/README.md; logs/README.md; results/README.md |
| Pilot configuration | configs/pilot/legacy-clean.json; no-resoftmax.json; aligned-init.json; manifest.json |
| Final design | configs/final/benchmark.json |
| Scripts | scripts/run_pilot.py; aggregate_pilot.py; verify_environment.py; run_tests.py; smoke_test.py |
| Source package | src/loster2026/__init__.py; config.py; protocol.py; architecture.py; assignments.py; losses.py; data.py; augmentation.py; alignment.py; metrics.py; training.py; checkpointing.py; diagnostics.py; reproducibility.py; efficiency.py; selection.py; smoke.py |
| Tests | tests/support.py; test_contract.py; test_data_metrics.py; test_torch_regression.py |
| Readiness record | analysis/audits/05-implementation-readiness.md |

The source reimplements the active AE path rather than copying the complete historical tree. No RevIN/global AE residual/dormant center implementation is carried into the trained model. Source comments identify legacy files/functions and relevant Audit1 findings/Audit4 choices.

Legacy files inspected read-only included net/model.py, utils/losses.py, utils/gumbel.py, utils/random_seed.py, data/augmentation.py, main.py and experiment.py. Regression tests extract selected definitions from the original files with AST and execute them only on artificial tensors; no legacy launcher, CUDA-only entry point or result script is run.

## Exact Variant Implementation

A single [loss implementation](../../experiments_new/loster2026/src/loster2026/losses.py) and [training implementation](../../experiments_new/loster2026/src/loster2026/training.py) serve all three variants.

**LoSTer-Legacy-Clean:** three encoder blocks/three decoder blocks, independent views, dropout0.1 in ordinary blocks, final prediction block without dropout/LayerNorm. Reconstruction means over BL, k-means means over Bd, half-sum of view k-means terms, alpha1, two summed directional instance/cluster cross-entropies and summed negative entropy. Instance rows are normalized; cluster columns use extra softmax(Q_ST), cosine and same-index positives. Positive cross-view terms remain in denominators; same-view diagonals receive the historical1e9 mask. Sigma1, instance/cluster temperatures1, tau=max(10×0.65^epoch,0.01), original SGD/StepLR and strict deterministic original-view stopping are retained.

**LoSTer-NoResoftmax:** only the cluster representation/entropy branch changes. It uses Q_ST directly, occupancy×log(max(occupancy,1e-8)), and dot products divided by count-column norms floored at1. Empty anchors are retained; no pseudocount/balancing/reseeding occurs. All other losses, architectures, optimizers, schedules, views, normalization, metrics and stopping are identical.

**LoSTer-AlignedInit:** independent ordinary KMeans fits first produce assignments on corresponding X and A_init rows. A k×k label-free contingency is maximized by linear assignment; augmented center rows alone are permuted once before optimizer construction. Independent updates continue thereafter. Deterministic ascending row/column inputs and the exact SciPy version/permutation are recorded; no cost jitter or gradient-through-matching is invented. Repeated diagnostic matching does not modify centers or positives.

ST sampling delegates to the same active PyTorch hard=True Gumbel API as the historical experiment. Deterministic inference uses the original AE and ACTIVE original centers only. The code supports CPU, CUDA and automatic device selection without touching the CUDA-only historical implementation.

## Data, Configurations and Planned Runs

The explicit tab parser treats all UCR records as headerless, preserves TRAIN-then-TEST order and checks numeric finite samples, equal length and nonzero population SD before per-series normalization. It never removes, imputes, resamples or invents constant-series behavior. Numeric/string class IDs are encoded for evaluation; k/class counts are explicitly oracle metadata. Training batches contain only X/A_train.

Counts, N,L,k, class counts, normalization, pool order and input-file hashes enter run provenance. Loading/normalization timing is recorded separately. Batch size is min(128,N), shuffled/drop_last=True, with an assertion that at least one batch exists.

Two fixed augmentation snapshots remain separate. Normalized float64 inputs undergo equiprobable sign inversion, uniform1–4 equal-segment permutation, cubic warp with sigma0.2/four interior knots, endpoint scaling/clipping and interpolation. Model arrays are float32. Index-based segment permutation handles ragged arrays without changing distribution/draws; exact old-kernel output and RNG equivalence was tested on the divisible-length fixture. Coordinate minima/non-increasing steps/clipping/endpoint scales are logged without monotone repair.

The [pilot manifest](../../experiments_new/loster2026/configs/pilot/manifest.json) contains exactly **120 unique configurations**, validated for complete paired coverage:

- Datasets: SyntheticControl, Beef, ECG200, OSULeaf, ShapesAll, SemgHandMovementCh2, CinCECGTorso, StarLightCurves.
- Variants: LoSTer-Legacy-Clean, LoSTer-NoResoftmax, LoSTer-AlignedInit.
- Seeds: **0,1,2,3,4**.
- Each entry includes dataset, variant, seed and config identifier.

The runner defaults to planning only; dry-run confirmed execution=false/run_count120/unique120. No important scientific setting is implicit in an untracked launch command. Explicit base configs cover widths/blocks/dropout, both budgets/optimizers/LRs, scheduler, tau/beta/floor, sigma, temperatures, alpha, tolerance, augmentation, normalization, metrics and numerical conventions. Pilot contract validation rejects unintended changes.

Final configuration preserves exact CORE40/EXTENDED44, development8, primary36 and seeds100..104. The final selected method remains null. Audit4's numeric method-selection gates and deterministic tie sequence are implemented in read-only aggregation; missing/failed legacy controls prevent a numerical decision, and any recommendation still requires conceptual review. No method is automatically frozen.

## Regression Coverage and Tolerances

| Required check | Evidence |
|---|---|
| Dense residual formula | Learned skip, dropout and LayerNorm checked against explicit formula and actual legacy class state. |
| Encoder/decoder | Output/latent shapes, loaded legacy AE numerical equivalence, final-block exception and independent weights. |
| RBF/sigma | Direct legacy logit comparison at sigma1 and2.3; default1 equals explicit1. |
| Deterministic inference | Nearest-center assignments and first-index tie behavior. |
| k-means/instance losses | Direct numerical comparison to extracted original classes. |
| Cluster/entropy/S0 | Exact extra-softmax representation, combined contrast/entropy value and backward comparison, empty columns included. |
| Full objective | Original grouping, half k-means coefficient and reductions compared numerically. |
| Schedule/stopping | Exponential floor, second-epoch eligibility, strict boundary at0.001 and ordered change calculation. |
| Header bug | Actual historical pandas call loses one row per split; corrected parser retains all synthetic fixture rows/labels/k/counts. |
| Metrics | Known perfect/cross/collapsed clusterings; explicit arithmetic versus geometric NMI; Hungarian ACC and integer-pair RI. |
| NoResoftmax isolation | Reconstruction/k-means/instance unchanged; direct Q representation and finite bounded empty-column backward. |
| AlignedInit isolation | Only augmented-center rows permute; all other states, optimizers and replayed RNG draws agree. |
| Augmentation | Old transform output/RNG match; two snapshots replay but differ; ragged segments preserve length. |
| Checkpoint | Destroy/reload state and exact predictions; augmented active centers retained; missing ACTIVE original centers rejected. |
| Gradients/pretrain | One real artificial pretraining epoch executes; sparse component diagnostics preserve RNG, existing gradients and resulting optimizer update exactly across all variants. |
| Safety/configuration | Duplicate/missing manifests, accidental hyperparameter changes, failed legacy decision, new collapse, tiny gains and deterministic ties. |

Float32 component comparisons use **rtol1e-6, atol1e-7**; float64 comparisons use **rtol1e-10, atol1e-12**. Saved/reloaded assignments, replayed random draws, augmentation arrays, isolated invariant terms and post-diagnostic optimizer states use **exact equality**. Tolerances were not relaxed to obtain passing tests.

A newly added diagnostic test initially failed because it assumed old PyTorch zero_grad leaves .grad=None. That optimizer version instead preserves zero tensors from the previous update. The test was corrected to snapshot and compare the actual pre-existing None/tensor state exactly, while retaining exact RNG and resulting-weight assertions. The corrected test passed; no scientific implementation workaround was introduced.

## Test Results

Final per-interpreter results for the **52-test suite**:

| Interpreter/environment | Passed | Failed/errors | Skipped | Meaning |
|---|---:|---:|---:|---|
| Default Python3.13.15 | 17 | 0 | 35 | Static/config/protocol/selection coverage; scientific modules absent. |
| Existing elenv Python3.9.2 | 46 | 0 | 6 | Tensor/loss/model/checkpoint/gradient/pretrain and NumPy/SciPy coverage. Skips: four metric tests, one pandas header test, one full synthetic test. |
| Base Anaconda Python3.7.9 | 32 | 0 | 20 | Includes pandas header and all four metric tests; torch-dependent tests/full smoke unavailable. |

The union is **51 distinct tests passed / zero remaining failures / one unexecuted full-pipeline test**. This union is not a successful single-environment complete-suite run. The Python3.7 stack is diagnostic coverage, not the proposed reference execution environment.

The test runner uses unittest, so missing pytest does not block it. Environment verification and smoke guards intentionally return a nonzero readiness status when dependencies are absent; those statuses are not failing numerical regression tests.

## Synthetic Smoke Status

**BLOCKED / NOT EXECUTED end-to-end.**

The [synthetic smoke entry point](../../experiments_new/loster2026/scripts/smoke_test.py) and the full three-variant regression test exist. They are designed for16 artificial length24 series, latent width8, one pretraining epoch per view and1–2 joint epochs, temporary output, evaluation, diagnostics and complete checkpoint reload. The same preparation is cached across arms.

The guard reported missing numpy/scipy/sklearn/torch under default Python and missing sklearn under elenv. Base Anaconda lacks torch. No substitute KMeans/metric implementation, binary mixing or package installation was used to force execution.

Separately, artificial **component execution** did pass: real short pretraining, a stochastic joint-objective update with/without gradient monitoring for all variants, centroid permutation, and exact synthetic checkpoint inference round trip. Those checks are not mislabeled as the missing full KMeans/training/evaluation pipeline smoke result and have no quality interpretation.

## Installed Environments and GPU

Hardware inspection ran python --version and nvidia-smi before CUDA-specific checks. The host exposes **NVIDIA GeForce GTX1080, 8GiB**, driver **552.22**, with nvidia-smi advertising CUDA12.4 capability. That capability is not the installed PyTorch runtime version.

| Environment | Actual imported packages/readiness |
|---|---|
| Default Python3.13.15 | numpy, scipy, sklearn, torch, pandas and pytest absent. Runtime NOT READY. |
| Registered Python3.9.13 | NumPy1.20.1/SciPy1.6.1 import; sklearn/torch/pandas/pytest absent. Runtime NOT READY. |
| Base Anaconda Python3.7.9 | NumPy1.21.6, pandas1.2.4, SciPy1.6.2, sklearn1.0.2, pytest4.0.2; torch absent. Runtime NOT READY. |
| elenv Python3.9.2 | NumPy1.20.1, SciPy1.6.1, PyTorch1.7.1 with CUDA10.1 runtime; sklearn/pandas/pytest absent. PyTorch reports CUDA available and GTX1080. Runtime NOT READY. |

Direct unactivated Anaconda imports initially raised DLL errors. Adding only each existing environment's own Library/bin/Scripts to that shell process's PATH resolved imports. No persistent environment setting or package was changed. Tests ran on CPU; no GPU scientific training was performed.

Audit reference versions PyTorch1.13.1+cu117/NumPy1.22.4/sklearn1.4.1.post1/SciPy1.13.0 have not been validated together. Component passes under older packages do not establish whole reference-stack fidelity. A complete compatible environment and full synthetic test remain the readiness blockers.

## Provenance, Outputs and Checkpoints

Generated run content is ignored under artifacts/logs/results; only README placeholders exist now. Source/config/test files are not ignored.

Each future seed has separate dataset/variant/seed folders for its manifest, complete checkpoint, epoch JSONL and metrics JSON. A pre-existing run/checkpoint raises rather than overwriting another seed.

Paired initialization cache includes both pretrained AEs, both independent fitted centers, X/A_train/A_init identities, initial assignments/diagnostics and the RNG boundary. Config/source/input/runtime/cohort checks prevent incompatible reuse. The preparation key excludes variant only; no cross-seed sharing occurs.

The checkpoint records both AE states, both ACTIVE center matrices, variant/dataset/seed/config, final epoch/tau, optimizer/scheduler, original/augmented assignments, metrics, Git/environment/provenance and RNG. Strict reconstruction requires the active centers; the executed unit test verified exact predictions after destroying/reloading objects.

Per-run JSON provenance records UTC timestamp, Git SHA/dirty state, source digest, Python/library/OS, CUDA/device, complete configuration and cohort/hash metadata. It intentionally omits usernames, home/executable paths, hostname and secrets.

## Diagnostics and Efficiency Readiness

Implemented epoch logs cover total/reconstruction/half-sum k-means/instance/cluster-contrast/entropy components, tau/LR, elapsed time, occupancy/minimum/max fraction/hard entropy/utilization, assignment changes, center displacement/dead duration/cumulative use, S0/RBF entropy, correspondence, batches/updates and stopping. Monitoring is graph-free and never changes center membership or stopping.

Sparse component norms use the first already-existing batch at Audit4's zero-based epochs0,1,4,9,19,39,59,79,99. Same-graph differentiation writes no parameter gradients and takes no additional stochastic forward. An exact invariant-update regression passed for all three variants.

Instrumentation provides synchronized CUDA timing boundaries, stage/full operational time, loading/normalization, inference throughput, registered/active/deployed parameters, host peak working set/RSS and GPU allocated/reserved peaks. Observed joint peaks and cache-reused preparation estimates are identified separately. Cost estimates include both pretrains; cache amortization is not represented as standalone deployment cost.

No benchmark-quality time/memory claim is made. Audit4's disposable calibration, warm-up/repeated complete inference and telemetry-separated measurement procedure must be applied in later authorized measurements. The single-pass operational instrumentation is prepared for that work, not a substitute for it.

## Unresolved Questions and Next Gate

No unresolved mathematical design ambiguity required an invented method choice. Audit4 supplies S1 zero conventions, C2-once semantics, fixed transforms/reductions and stopping. Engineering choices are explicit: tab parsing, first-existing-batch diagnostic position, versioned deterministic SciPy tie handling and strict output isolation.

Actual blockers:

1. One inspected interpreter must provide the complete compatible runtime stack; elenv specifically lacks scikit-learn, and default Python lacks all four runtime dependencies.
2. Full synthetic preparation/KMeans/joint training/evaluation/logging/cache/checkpoint workflow must pass for all three methods in that one environment.
3. The frozen reference-version assumptions must be checked there; partial older-version component tests are insufficient.

The observed synthetic warp-coordinate diagnostics do not establish validity/frequency on the future pilot data. Audit4 already defines the conditional validity investigation; no repair is inserted here. No methodological variant is selected or final method frozen.

## Completion Verification

Allowed working-tree edits are only .gitignore, experiments_new/loster2026 and this report. New implementation/Audit5 remain uncommitted. Generated package output directories contain only README placeholders; ignored-output rules were checked with git check-ignore.

Final checks: git status --short; git diff --stat; git diff -- models/LoSTer paper. The last command is empty, and preservation checksum verification covers the untracked immutable/legacy files as well. The annotated tag and branch remain at the committed Audit4 state.

Stop here. No real pilot, installations, manuscript changes, implementation commit or implementation push was performed.

## Environment remediation and final validation ? 2026-10-05

### Scope and isolation

The user authorized a dedicated environment and synthetic validation only. The repository remained on H: in Google Drive; it was not copied or moved. A new `venv` was created in the user-designated external local working area on G:. Wheelhouse, package cache, CA bundle, test logs, preservation inventories and CPU/CUDA smoke artifacts are local to that external area. Absolute environment/interpreter paths are recorded in the session report and local validation material, not embedded in scientific configs or either environment specification.

`py -0p` and direct execution identified the existing registered **CPython 3.9.13, 64-bit AMD64** interpreter. It successfully created the new environment with `include-system-site-packages=false`. Existing `elenv`, default Python 3.13 and other installations were not modified. No Python was installed. The new environment's bundled **pip 22.0.4** and **setuptools 58.1.0** were sufficient; neither was upgraded and wheel/pytest were not installed.

### Exact dependencies and compatibility before installation

| Package | Installed version | Declared Python requirement |
|---|---|---|
| NumPy | 1.22.4 | >=3.8 |
| pandas | 1.5.3 | >=3.8 |
| scikit-learn | 1.4.1.post1 | >=3.9 |
| SciPy | 1.13.0 | >=3.9 |
| PyTorch | 1.13.1+cu117 | >=3.7.0 |

Before installation, official [NumPy metadata](https://pypi.org/pypi/numpy/1.22.4/json), [pandas metadata](https://pypi.org/pypi/pandas/1.5.3/json), [scikit-learn metadata](https://pypi.org/pypi/scikit-learn/1.4.1.post1/json), [SciPy metadata](https://pypi.org/pypi/scipy/1.13.0/json) and the [official CUDA 11.7 PyTorch wheel index](https://download.pytorch.org/whl/cu117/torch/) confirmed exact CPython 3.9 Windows AMD64 wheels. Wheel metadata independently confirmed Python 3.9.13 compatibility and every resolved runtime dependency constraint. In particular, SciPy's NumPy >=1.22.4,<2.3 and scikit-learn's NumPy >=1.19.5,<2.0 constraints both admit the exact 1.22.4 pin. No scientific pin was substituted.

Only those five packages and their six required runtime dependencies were installed: **joblib 1.5.3; threadpoolctl 3.7.0; python-dateutil 2.9.0.post0; pytz 2026.5; six 1.17.0; typing-extensions 4.16.0**. `pip check` reported **No broken requirements found**. Actual imports confirmed all five scientific versions in the new environment. After creating the lock, pip also accepted every locked requirement against the installed environment without installing or changing any package.

Downloads used PyPI and the official PyTorch index, with wheels/cache on G:, followed by offline installation from that wheelhouse. The exact PyTorch wheel SHA256 matched its official index hash: `e775fa85f412bd1bf816b8798dadb3b852b71e33e8008e9db29b6190ed94fe27`.

An initial HTTPS certificate-chain error was resolved using a local bundle containing pip's bundled CA certificates and existing trusted Windows CA certificates, supplied through process-local `PIP_CERT`. Certificate verification stayed enabled; neither the system trust store nor other environments changed. Transient connection resets were recovered by normal retries. These installation problems are resolved and are not scientific version conflicts.

New specifications: [requirements-pilot.txt](../../experiments_new/loster2026/requirements-pilot.txt) contains the five exact direct pins; [environment-lock.txt](../../experiments_new/loster2026/environment-lock.txt) contains all eleven runtime pins and hashes of the validated Windows CPython 3.9/universal wheels. README now documents interpreter discovery, external environment/cache/temp placement, installation, CPU fallback, GPU checks and verification commands. This is a **controlled environment for the 2026 reruns preserving audited library semantics where possible**, not an exact reconstruction of the original historical runtime.

### CPU validation

The first full suite executed CPU model paths with all packages importable and returned 52/52 passes. The attempted empty CUDA visibility setting was recognized as removing the variable in PowerShell, leaving CUDA visible. To eliminate that isolation uncertainty, a fresh process used **CUDA_VISIBLE_DEVICES=-1**; the environment verifier then reported CUDA unavailable by design while still importing the pinned CUDA wheel. The full unchanged suite was repeated under this explicit CPU isolation:

| Required test result | Count |
|---|---:|
| Run | 52 |
| Passed | 52 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |

Native Python exit code was **0**. Previously unavailable metric, literal pandas header-loss and full end-to-end tests all executed. Corrected headerless loader tests retained every TRAIN/TEST fixture row in the expected order; count/class/oracle-k, population normalization and invalid-input checks passed. Legacy mathematics, augmentation/RNG, NoResoftmax isolation, AlignedInit permutation/replay, diagnostic noninterference, stopping, persistence and manifest regressions passed at their original tolerances. No source, tests, expected values or tolerances were changed.

After that isolated suite passed, the existing `synthetic_smoke(output_root)` ran in the external CPU smoke directory. The fixture remains 16 artificial length-24 series, seed 23, latent width 8, batch 8, workers 0, one pretraining epoch per view and at most two joint epochs. Each variant completed **two joint epochs**:

| Variant | CPU smoke | Diagnostic epoch records | Exact checkpoint prediction reload |
|---|---|---:|---|
| LoSTer-Legacy-Clean | PASS | 2 | PASS |
| LoSTer-NoResoftmax | PASS | 2 | PASS |
| LoSTer-AlignedInit | PASS | 2 | PASS |

An external validation wrapper checked completed manifests, finite/complete execution, all four metric fields, sparse gradient diagnostics, checkpoint existence, recorded exact reload equality and a shared paired initialization key. These checks cover preprocessing, model construction, both short pretrains, both KMeans fits, joint training, metrics, diagnostics and checkpoint persistence. Metric values and timings were not interpreted as clustering quality or benchmark evidence.

### GTX 1080 and CUDA validation after CPU success

| Property | Observed result |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 |
| Memory | 8192 MiB |
| NVIDIA driver | 552.22 |
| Compute capability | 6.1 |
| PyTorch | 1.13.1+cu117 |
| PyTorch CUDA runtime | 11.7 |
| CUDA available with GPU visible | Yes |
| Actual CUDA matrix multiplication and backward | PASS |
| Comparison with CPU tensor results and gradients | Exact equality |

The sanity test used only a 4x4 float32 matrix, asserted device-resident outputs/gradients, synchronized CUDA, and compared both forward and backward results exactly with CPU. This confirms execution on the GTX 1080, beyond visibility in nvidia-smi. No CUDA adjustment or scientific version substitution was needed.

The harness already supports explicit CUDA in `fit()`, so the same tiny artificial fixture was then run through that existing path in a separate external CUDA output directory:

| Variant | Tiny CUDA smoke | Joint epochs | Exact checkpoint prediction reload |
|---|---|---:|---|
| LoSTer-Legacy-Clean | PASS | 2 | PASS |
| LoSTer-NoResoftmax | PASS | 2 | PASS |
| LoSTer-AlignedInit | PASS | 2 | PASS |

All six retained CPU/CUDA variant manifests were completed, had two diagnostic records and recorded exact prediction reload. Paired initialization keys matched across variants within each device. No CPU/GPU trajectory or quality equivalence is claimed; these are execution checks only. The real pilot's GPU capability requirement is satisfied on this machine.

### Final static preflight and preservation

Read-only manifest/config inspection confirmed **8 datasets ? 3 variants ? 5 seeds = 120 unique planned runs**, seeds **0,1,2,3,4**, and **40 dataset/seed groups** each containing all three variants. Every config identifier matched its variant; every resulting config satisfied the frozen Audit 4 contract. Within every group the three initialization/config hashes excluding only variant were identical. No dataset loader, UCR metadata fetch/download or real pilot fit was invoked for this preflight.

The starting Git status was exactly the prior intended state: modified `.gitignore`, untracked Audit 5, and the untracked harness. Protected-path diffs were empty. A full before/after inventory verified **335 existing files unchanged in bytes, size and mtime**, excluding only this audit and the harness README. All scientific sources, scripts, configs and tests, plus the prior `.gitignore`, stayed unchanged during remediation. The same **300 frozen pre-existing files** retained aggregate inventory SHA256 **ff1d9c9ee0ebfd200368b956b928dd888971d52cd86dc79f0157616696ca0ab2**. The only added repository files are the two environment specifications; all environment/cache/synthetic artifacts remain outside Git on G:.

Final checks: `git status --short`, `git diff --stat`, and `git diff -- models/LoSTer paper Drafts csv_logs figures`. The protected-path diff is empty and the external virtual environment is absent from Git status. The tracked diff still consists only of the eight prior `.gitignore` additions; the harness and Audit 5 remain untracked pending review.

### Remaining notes and final gate

**Final verdict: READY. No unresolved readiness blocker or scientific warning remains.** Optional pytest absence is intentional; the complete standard-library unittest suite has no skips. The wheel lock is scoped to the validated Windows CPython 3.9 platform. The environment verifier's hardcoded conservative `reference_versions_verified=false` field is not a version-comparison or readiness result; this audit separately verifies exact pins and all required gates. Neither note changes scientific meaning or pilot execution.

**No UCR dataset was downloaded or used, no real pilot run was executed, no legacy/model/manuscript/result file was modified, and no commit or push was made.** All saved results in this remediation are artificial execution checks in the external working area. The pilot remains 120 planned, unexecuted runs and requires separate user authorization.
