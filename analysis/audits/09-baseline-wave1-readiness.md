# LoSTer 2026 Baseline Wave 1

## Executive Summary

Prepared 2026-10-09 (Europe/Rome). Implemented and validated exactly the three READY FOR IMPLEMENTATION methods in [Audit 08](08-baseline-protocol-gate.md) and the immutable [registry](../baselines/baseline-registry-2026.json): Euclidean k-means, k-Shape and KASBA. All three are **READY FOR DEVELOPMENT BENCHMARK**, within the tested scope below. No conditional or blocked method was promoted. **CORE-44 authorization: NO.**

The final validation gate reports **57 passed, 0 failed, 0 errors, 0 skipped**. Each method passed two standalone same-seed tiny-input fits with exact partition/center repeatability and checkpoint reload. Subsequently exactly **one SyntheticControl seed-0 real fit per method** succeeded: three real fits total, 600 predictions each, separate-process central evaluation, artifact persistence and an identical-identity resume that performed no fit. No real-fit warnings, collapse, numerical failure or new baseline blocker occurred. Other Development8 tasks and Final36 were not executed.

Stage A committed only Audit 08 and the registry as **2fb270eb39900790f1e6c1c915463a555495e88a**, message `Add baseline protocol and reproducibility registry`, and pushed only `origin revision-2026`; status was clean afterward. The Wave 1 framework and this report remain **uncommitted and unpushed**. Audit 08 and the registry have not been edited after preservation. Their design-stage environment descriptions remain historical; this report records the new verified state.

No scientific changes were made to LoSTer or the native baseline algorithms. Frozen LoSTer remains **LoSTer-Legacy-Clean + MonotoneWarp**, commit **cab7b6ffecab92bfcf6fe46e73e32dc0c040ee59**, tag **loster-2026-frozen**. The repository-wide safety snapshot covered 95 protected files; all retained their hashes. The 41 frozen scientific source/configuration files across `experiments_new/loster2026/src/`, `configs/` and `models/LoSTer/` match the tag byte for byte after Git's checkout/EOL filters.

## Methods Selected from Audit 08

The report's abbreviated "Euclidean" readiness row and the registry's full name refer to the same method. Semantic comparison established exactly three matching READY records; there was no readiness disagreement.

| Method / registry ID | Canonical source and frozen commit | Environment and adapter | Remaining minor risk |
| --- | --- | --- | --- |
| Euclidean k-means / euclidean_kmeans | [scikit-learn](https://github.com/scikit-learn/scikit-learn), **719c0c6036ac99d45ffb45ed7c8b4e4b221384b5**, release 1.4.1.post1, BSD-3-Clause | CPU, Python 3.9.13; NumPy/SciPy/scikit-learn/threadpoolctl; native KMeans on N,L | Canonical third-party library relative to Lloyd, expressly selected by Audit 08; fixed n_init10 convention and platform/thread sensitivity must remain explicit |
| k-Shape / kshape | [TheDatumOrg/kshape-python](https://github.com/TheDatumOrg/kshape-python), **778e735624d15848556e4d23fe38c321d453937a**, package 1.0.6, MIT | CPU author checkout, NumPy and source-imported scikit-learn; native KShapeClusteringCPU on N,L,1 | Windows pool/spawn and long-L eigensolver/copy cost; native class does not expose an iteration count |
| KASBA / kasba | [aeon-toolkit/aeon](https://github.com/aeon-toolkit/aeon), **0fed29f21908cfad2981ea645f51aaf018f0fe89**, v1.1.0, BSD-3-Clause | CPU, aeon/NumPy/SciPy/pandas/scikit-learn/Numba; native KASBA on N,1,L | Cold JIT and length-squared MSM/barycenter cost; compatibility on this short task does not establish long-task feasibility |

The framework rejects conditional/blocked method IDs. It uses the external source directories already selected in Audit 08; no replacement implementation, duplicated source repository or scientific source edit was introduced.

## Common Data Contract

`experiments_new/baselines2026/src/baselines2026/datasets.py` reads the existing official headerless TSV files from:

`G:/Articoli/Articoli da Completare/LoSTer 2026/data/ucr`

All rows are preserved in **TRAIN then TEST** order. IDs are `<dataset>/<TRAIN|TEST>/<zero-based-record-index>`. Metadata includes raw float64 sequences, split paths/counts/SHA-256, pooled N, fixed L, oracle k, raw-array hash and ordered-ID hash. File hashes are checked before and after ingestion. Ragged, nonfinite or malformed records and metadata/hash mismatches are rejected, without imputation or sample deletion.

TrainingInput contains no membership labels; its array is read-only. The trusted ingestion layer transiently reads label metadata to derive permitted oracle k, then discards membership. Adapters receive X and k, never y. EvaluationInput is loaded independently after immutable export. No class-count-by-class metadata reaches the fit API.

The common outer preprocessing is Audit 08's per-series population z-score, ddof=0, float64. Semantics match the frozen public model-independent utility, including rejection of zero/nonfinite variance. Native centroid normalization, FFT alignment and elastic transformations remain intrinsic. No LoSTer augmentation is used. Shape conversion is N,L for Euclidean, N,L,1 for k-Shape, and N,1,L for KASBA.

SyntheticControl identity: **N=600, L=60, k=6; TRAIN=300, TEST=300**.

| Raw input | SHA-256 |
| --- | --- |
| TRAIN | 58af636cbd5592bdda45ba638629e40f17c08354aa545a61fd41b92582bdd176 |
| TEST | cd44464b200820913f928a952af78d512ee3924f6243053d28a6d45c0aacdb76 |

All three normalized arrays had SHA-256 **f165b336a5d53eca0a185e49ba8d5a50fd4a72369d3f56a4ae191b8819e3bfa2**. The config pins expected raw hashes and dimensions. No large dataset was copied into Git.

## Prediction Contract

Version **baselines2026.predictions.v1**, UTF-8 JSON, exclusive creation. Required fields include method, dataset, root seed, canonical ordered sample_ids, predicted_clusters, N/L/k, raw input and array/ID hashes, configuration hash, source commit, environment fingerprint, adapter tree hash, run_id and SUCCESS execution status.

Validation requires exactly N one-dimensional finite integer cluster IDs within int64 bounds, unique string IDs in the exact canonical order and matching dataset identity/hashes. Booleans, strings, fractional/nonfinite values, dimensional errors, malformed provenance and missing/duplicate/reordered samples are rejected. Negative or noncontiguous integer cluster IDs are valid. Natural underutilization is accepted; occupied-cluster count, fraction and collapse are reported separately.

Native `fit.labels_` is the transductive partition exported in canonical order. A distinct native full-pool `predict` pass is timed and compared without labels. All three real fitted partitions equaled fresh inference. This preserves native fit results if a future native fitted partition differs from a fresh nearest-center assignment; no silent reassignment is performed.

All generated predictions, models, logs, environment files, test XML, Numba caches and source bundles reside on G:. Predictions and model hashes are verified by the no-fit resume path. An incomplete/corrupt/stale result is refused, rather than overwritten or implicitly refitted.

## Independent Metrics

`evaluator.py` is **baselines2026.metrics.v1**. Its file SHA-256 for these runs is **e34c448fa192adf5608809a555a52d24ab128b5cf6c362f56321819f5a28ded7**.

The evaluator reloads frozen predictions and canonical evaluation labels, validates exact sample alignment and raw dataset identity, then computes:

- **ARI**, primary, scikit-learn's adjusted Rand index.
- **NMI_arithmetic**, secondary, explicitly `average_method="arithmetic"`.
- **RI**, descriptive, integer pair counting of agreeing same/different relations; singleton convention 1.
- **ACC**, descriptive, maximum rectangular Hungarian contingency matching divided by N; no forced square label range.

Each output records evaluator version/source hash, submission hash, run ID, method, dataset, seed and N. Real evaluation runs in a separate process after export and checkpoint verification. It cannot select initialization, stopping, hyperparameters, checkpoint or assignments.

Perfect/permuted, singleton/degenerate, single-cluster, rectangular matching and independent enumerated-pair examples passed. Five assignment examples also exactly matched the frozen public LoSTer metric utility; historical score files were not recomputed. The schema/evaluator accepts future compatible LoSTer exports, so a later campaign can use this same implementation.

## Resource/Timing Contract

CPU only; no GPU use or GPU-memory comparison. Actual host: **Intel Core i7-8700K @ 3.70 GHz**, 6 physical / 12 logical CPUs, **16,977,186,816 bytes** physical RAM. Each manifest records processor family/model/stepping, platform, CPU counts, RAM and device. BLAS/OpenMP/Numba thread environment is fixed to 1; threadpoolctl scopes native work to one thread. k-Shape retains its one-worker native pool. Effective neural batch size and peak GPU memory are null/not applicable.

A monotonic `perf_counter` measures the common pipeline from the decoded raw numeric pool before normalization through native initialization, every internal training/restart/refinement stage, full-pool inference and assignment validation. Raw file decoding, import/source verification, checkpoint/export/reload and central evaluation are excluded operational work, matching Audit 08's clock boundary. Normalization, native fit, full-pool inference and validation stages are recorded individually. A CUDA synchronization hook exists and its before/after boundary behavior is tested; no CUDA adapter is claimed validated.

Host RSS is sampled every 0.02 seconds over the fit process and children. It may miss brief peaks and may count shared pages across processes; it is a measured process-tree RSS estimate, not guaranteed peak private allocation. Per-method real-run Numba caches were new; KASBA cold JIT is charged. The k-Shape worker startup/pool overhead is charged.

**These are integration-smoke timings, not publication-quality efficiency results.** No inference warmup/repeated-pass efficiency study or campaign timeout/admission validation was conducted.

## Baseline Adapters

Before wrapper implementation, Audit 08's Method Details/Complete-Pipeline Timing descriptions and pinned native entry points were inspected.

| Method | Preserved native scientific recipe / selection |
| --- | --- |
| Euclidean | KMeans: squared Euclidean Lloyd refinement, k-means++, 10 internal initializations, max_iter300, tol1e-4, algorithm=lloyd, copy_x=True, random_state0. Native minimum inertia selects the restart; no target labels |
| k-Shape | Author CPU class: random initial partition, zero centers, native FFT/SBD alignment and eigenvector centroid, native empty-cluster behavior, max_iter100/stable labels, n_jobs1. NumPy global RNG scoped to seed0; no target labels |
| KASBA | aeon native MSM c=1, elastic initialization, stochastic barycenter/pruning pipeline; subset.5, initial step.05, decay.1, max_iter300, tol1e-6, random_state0. Actual native barycenter inner cap50 and fastmath behavior preserved; no new restart or target-selected state |

Installed sklearn `cluster/_kmeans.py`, aeon `clustering/_kasba.py`, `clustering/averaging/_kasba_average.py` and `distances/elastic/_msm.py` matched pinned Git blobs after line-ending normalization. k-Shape's imported `core.py` came from its clean pinned author checkout and matched the Git blob. Releases/wheel versions and source pins are both recorded.

Scientific consequence: these are non-scientific adapters plus protocol/provenance instrumentation. They preserve native objectives, initialization, iteration limits and learned-state selection. Common pooling/z-scoring is the predeclared matched task adaptation, not a claim to reproduce each paper's original split table. No manuscript/code inconsistency was repaired; no methodological improvement or optional refactoring was applied to frozen LoSTer. The environment-enumeration bug described below affected validation metadata only.

## Environment Provenance

One newly created isolated compatible CPU environment serves exactly these three methods:

`G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/envs/wave1-cpu-py39-v1`

Python **3.9.13**. Key pins: NumPy1.26.4, SciPy1.13.1, scikit-learn1.4.1.post1, aeon1.1.0, pandas2.2.3, Numba0.60.0/llvmlite0.43.0, threadpoolctl3.5.0, psutil6.1.1 and pytest8.3.5. The complete **27-distribution** closure, including packaging tools, is in [requirements-wave1.lock.txt](../../experiments_new/baselines2026/configs/requirements-wave1.lock.txt). k-Shape1.0.6 is imported from its separately pinned source checkout, rather than installed as a wheel. Installed distribution enumeration is restricted to the environment's site-packages, so source egg-info appearing on sys.path cannot mutate the environment fingerprint.

- Lock SHA-256: **f69d997a98fc38d6b785842cba04a310c5fc78f725400f268d068e0a29b195fa**.
- Environment fingerprint: **2273ecc22bd24e18b79d2775f6920f8d9af3ba0ab76b96671d13172f5c35cad5**.
- Adapter content-tree SHA-256: **30b60e97568814e97254d34a77b71948cb99277de44fb90752e309b54a4e7a7d**.
- Adapter parent Git commit: **2fb270eb39900790f1e6c1c915463a555495e88a**.
- **adapter_commit=null, adapter_source_committed=false**, because this task expressly prohibits committing Wave 1. The parent SHA does not contain the adapter. Each real run archives all 22 exact framework files with per-file hashes and the tree hash; these bytes identify the executed uncommitted implementation.

Source requirement review established the selected NumPy/SciPy/pandas/sklearn/Numba versions satisfy aeon's release bounds and the CPU implementations' imports. `pip check` and import verification passed. No global package or LoSTer dedicated-venv package was changed.

The initial environment install failed TLS certificate validation under legacy pip/Python3.9 (**DEPENDENCY ISSUE**, resolved before validation). A PEM bundle of 278 trusted Windows ROOT/CA certificates was exported and supplied via pip's `--cert`; certificate validation remained enabled. No unverified heavy dependency or scientific implementation change was needed. Official release references: [aeon1.1.0](https://pypi.org/project/aeon/1.1.0/), [Numba0.60.0](https://pypi.org/project/numba/0.60.0/), [sklearn1.4.1.post1](https://pypi.org/project/scikit-learn/1.4.1.post1/), [pip HTTPS certificates](https://pip.pypa.io/en/stable/topics/https-certificates/).

## Regression Tests

Three pytest attempts were retained, with no real-data fit before the final gate:

| Attempt | Passed | Failed | Errors / skipped | Diagnosis and disposition |
| --- | ---: | ---: | --- | --- |
| validation-001 | 56 | 1 | 0 / 0 | **ADAPTER BUG**: environment enumeration counted author-checkout kshape egg-info as an installed package; restrict enumeration to isolated site-packages, preserve separate source provenance |
| validation-002 | 56 | 1 | 0 / 0 | **ADAPTER BUG / diagnostic side effect**: a diagnostic import created untracked kshape/__pycache__, correctly rejected by the clean-source check; archive bytecode outside the checkout, use bytecode-disabled validation; no source byte was edited |
| validation-003 | **57** | **0** | **0 / 0** | Complete passing gate, 24.24 seconds pytest-reported |

These are resolved infrastructure failures, not hidden failed real baseline fits. Logs/XML and the diagnostic bytecode archive remain on G:. No failed seed or experimental score was substituted.

The passing tests cover source/adapter-native parity for all three methods, exact fitted centers/partitions, native inference equivalence, RNG scope/restoration and stable phase seeds, rejection of non-READY IDs, no-membership fit signature, published settings, raw-reader/headerless sample integrity, normalization agreement, malformed predictions/hashes/IDs/provenance, int64 boundaries, immutable persistence, collapse acceptance, independent metrics, timing synchronization hook, environment lock integrity, external output paths and RSS recording.

Protected diffs are empty. All 95 safety-snapshot hashes and 41 frozen scientific source/configuration comparisons passed. The three source HEADs and working trees remain unchanged/clean. Audit 08 and its registry have zero post-preservation diff. Source bundles and prediction/model hashes were independently rechecked after the smokes.

## Synthetic Smoke Tests

Standalone validation input is synthetic **N=16, L=16, k=2**, generated with fixture RNG19. Each method fitted twice with root seed0 and the same full native/default scientific recipe used for the real smoke. Unit parity tests also perform tiny-input native/adapter fits; they are separate from these six standalone fits.

| Method | Standalone fits | Same-seed labels / centers | Checkpoint labels / inference | Prediction export / reload | Warnings |
| --- | ---: | --- | --- | --- | --- |
| Euclidean | 2 | Exact / exact | Exact / exact | PASS | 0 |
| k-Shape | 2 | Exact / exact | Exact / exact | PASS | 0 |
| KASBA | 2 | Exact / exact | Exact / exact | PASS | 0 |

All synthetic diagnostic metrics were 1.0, after immutable assignment export. They were used to validate plumbing, not select hyperparameters or declare comparative quality. All tiny runs occupied 2/2 clusters. Stochastic native initialization is seed-controlled; these results establish repeatability on this environment/fixture, not cross-platform bitwise guarantees.

## Development8 Integration Smoke

Exactly three real native fits: **SyntheticControl, root seed0**, no target-label fit/checkpoint selection. Recorded completion UTC was 2026-10-08 22:17:38 through 22:18:24 (2026-10-09 local). All 600 ordered fitted labels exported, all six clusters occupied, no collapse/warnings, fresh native inference matched the fitted partition, checkpoint labels/inference reloaded exactly, and a separate identical invocation reported `REUSED WITHOUT FIT` with unchanged predictions.

| Method | Native iterations exposed | Pipeline seconds | Inference seconds | Sampled process-tree RSS MiB | Fits / evaluations / no-fit resumes |
| --- | ---: | ---: | ---: | ---: | --- |
| Euclidean | 12 in selected restart; native 10 restarts preserved | 0.5009495 | 0.0815656 | 112.629 | 1 / 1 / 1 |
| k-Shape | Not exposed; native stable-label/max100 stopping preserved | 15.7982458 | 1.2704504 | 215.914 | 1 / 1 / 1 |
| KASBA | 7 | 20.3933042 | 0.2555435 | 271.191 | 1 / 1 / 1 |

Diagnostic evaluator outputs only:

| Method | ARI | NMI arithmetic | RI | ACC |
| --- | ---: | ---: | ---: | ---: |
| Euclidean | 0.6171803441 | 0.8055523039 | 0.8705119644 | 0.5683333333 |
| k-Shape | 0.5970926390 | 0.7053078890 | 0.8825876461 | 0.7183333333 |
| KASBA | 0.6125427693 | 0.7895195960 | 0.8693489149 | 0.5750000000 |

These scores have no competitiveness, tuning or statistical-comparison interpretation. No other seven Development8 tasks, Final36 tasks, final seeds or CORE-44 pipeline were run.

## Reproducibility

External evidence root:

`G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/wave1`

Authoritative validation: `validation-003/gate.json`, `unit-tests.xml`, logs and `synthetic/<method>/result.json`. Failed attempts remain at `validation-001/` and `validation-002/`. Real evidence: `development-smoke-001/summary.json`, fit/evaluation/resume logs, and:

| Method | Run ID under development-smoke-001/runs/<method>/ |
| --- | --- |
| Euclidean | 292de83e89601192664d6a959e23363a547dd428b4cd1a56f0e13a206b7ace72 |
| k-Shape | f161a1a124ce80e88f12b3de24ebe10faa72c463046dfb03beffedea5b161f45 |
| KASBA | 133b639a9835312991848b826b30d1ac4c291e8c7de03c037f9096ba84807a43 |

Each run directory contains predictions.json, evaluation.json, model.pkl, manifest.json and adapter-source/. The manifest ties source commit, parent commit/content tree, environment/lock, raw and normalized inputs, ordered IDs, known k, method recipe, root/phase seeds, hardware/threads, native parameters, timing, utilization, warnings and checkpoint checks. Evaluation records the frozen prediction SHA-256.

Fit root seed is0. Python and NumPy global state are seeded/scoped/restored; Euclidean/KASBA receive explicit random_state0. k-Shape uses scoped global NumPy RNG. Independent inference phase seeds were Euclidean228138347, k-Shape719336909 and KASBA3673602196, derived by the documented SHA-256 phase mapping. PYTHONHASHSEED=0 and thread environment variables=1 were applied to workers. No labels are used to derive seeds.

The [framework README](../../experiments_new/baselines2026/README.md) and scripts document rerunning validation into a fresh G: evidence directory. Existing real runs can be verified/reused through the restricted runner with their original identities; do not launch a second real-fit output root under this completed three-fit authorization. The local pickle is a trusted self-produced checkpoint; submissions are JSON and do not require untrusted pickle loading.

## Remaining Conditions

Readiness is bounded to the successful adapter/environment/contracts and one-task integration. Further execution requires a new instruction and a deliberate expansion of the currently SyntheticControl-only runner. Outstanding campaign gates include:

- Commit/review/freeze the adapters and evaluator later, preserving archived smoke identities; this task leaves them uncommitted.
- Validate long-L/high-N resource feasibility, fixed timeout/host-memory policy and declared failure/retry rules before campaign execution. Short-task CPU timings do not satisfy that gate.
- Freeze publication efficiency warmup/pass/cache/device/thread policies and perform the required timed repeats separately.
- Preserve root/final seed policy; this stage did not fit final100..104 or consume other Development8 tasks.
- Integrate future LoSTer exports with the common schema/evaluator without modifying its frozen scientific behavior.
- Resolve five conditional baseline admissions and all three blocked sources; no claim of all-eleven reproduction, complete Development8 or CORE-44 readiness.

Python3.9 availability and this exact numeric stack are part of reproducibility. Cross-platform/backend upgrades require a newly identified environment and validation; existing evidence remains immutable.

## Blocked Sources Still Outstanding

No new baseline was blocked in Wave 1. Audit 08's three source-level blocks remain unchanged and were not trained:

| Method | Outstanding blocker |
| --- | --- |
| CKM | No official/author implementation or complete faithful recipe identified; no automatic substitution of local historical/third-party code |
| DTCC-2023 | Declared TensorFlow2.4 versus TF1/contrib/private-API source; disconnected cluster-contrast gradient; label-best published output/faithful label-free extraction and final-paper correspondence unresolved; legacy dependency/resource closure unverified |
| CDCC | Released integer augmentation-selector behavior and one-step recurrent temporal-axis handling differ from intended method; stochastic evaluation state/source correspondence and small-N batch issues require resolution; correcting view/axis behavior would change the scientific implementation |

These are unchanged audited blockers, not empirical failures from this smoke. License/distribution gaps and TFMCC's environment/global resource recipe remain as recorded in Audit 08; nothing here promotes those methods.

## Readiness Decision

| Method | Decision | Evidence / bound |
| --- | --- | --- |
| Euclidean k-means | **READY FOR DEVELOPMENT BENCHMARK** | Native/source checks, contracts, tiny repeat/reload and one real smoke pass; CPU/thread/source convention and further campaign gates remain |
| k-Shape | **READY FOR DEVELOPMENT BENCHMARK** | Author CPU Windows spawn/seed/native parity/export/reload pass; long-L resource cost and unavailable native iteration count explicitly retained |
| KASBA | **READY FOR DEVELOPMENT BENCHMARK** | Isolated aeon/Numba closure, native MSM/seed/recipe parity and one real smoke pass; long-task and cold-JIT policy remain |

Counts: **3 development-ready, 0 READY WITH WARNINGS, 0 newly BLOCKED** among these three. Runtime warnings were zero. The remaining conditions are campaign requirements, not evidence of a failed validated method. **CORE-44 authorization: NO.**

## Next Wave Recommendation

Follow Audit 08's order: first resolve conditions for Wave 1 extensions **R-Clustering and FASA** (environment/compiled RNG/PCA and seed/member export/native zero-norm behavior/distribution permission), then Wave 2 **official TS2Vec + k-means and FCACC**, followed by **TFMCC** after dependency and a globally frozen resource recipe are verified. Keep CKM/DTCC/CDCC in a source-resolution wave until their gate is reopened.

This is a recommendation only: no conditional promotion, additional development experiment, baseline implementation, final-panel execution or CORE-44 authorization is implied.
