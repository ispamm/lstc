# LoSTer 2026 Baseline Wave 1B

## Executive Summary

Prepared **2026-10-09, Europe/Rome**. Implemented exactly the two explicitly authorized conditional methods: **R-Clustering** and **univariate FASA**. Both completed one native **ECG200, seed0** integration fit with 200 canonical predictions, independent central evaluation, exact fresh-process checkpoint replay and a no-fit resume.

Final validation: **72 Wave1B/common tests + 57 unchanged Wave1 regression tests = 129 passing test executions; 0 failures, errors or skips**. The common 40 tests run in both environments, so this is an execution count, not 129 distinct tests. Both standalone synthetic suites passed three fits at seeds0,0,1. Fixed-seed labels, centers and stochastic state were exact; changed seeds changed native stochastic state.

An initial R dispatch failed during eager native JIT, **before feature fitting**, because a Numba temporary cache filename exceeded Windows MAX_PATH. The failed dispatch/logs remain immutable. A path-only adapter correction was validated before R's single actual native fit. FASA was not fitted again. All scientific source, recipes, seeds, PCA/centroid rules and environment versions were retained.

Verdicts: **R-Clustering: READY WITH WARNINGS** (native420/zero-PC domain caveats); **FASA: READY WITH WARNINGS** (unstated source license/native zero-norm caveat). No new scientific source blocker. The separate implementation-status registry now records **3 READY FOR DEVELOPMENT BENCHMARK, 2 READY WITH WARNINGS, 3 CONDITIONAL_NOT_IMPLEMENTED, 3 BLOCKED**. This is local development integration readiness, not all-panel feasibility. **CORE-44 authorization: NO.**

## Wave 1 Preservation

Read/reviewed [Audit08](08-baseline-protocol-gate.md), [Audit09](09-baseline-wave1-readiness.md), the [historical registry](../baselines/baseline-registry-2026.json) and the framework README. Initial worktree contained exactly 23 intended Wave1 files. Its bytes matched the validated archived source bundles; protected diffs were empty and the frozen-source comparison passed.

Committed only the framework and Audit09 as **5ff48377deac2b5934b450951eb41b1e3b08e7c9**, message `Add validated baseline Wave 1 framework and adapters`, pushed only `origin revision-2026`, and verified clean status afterward. No upstream push. Audit09's statements about its earlier uncommitted execution remain an immutable historical record; this new preservation commit contains the same validated bytes.

All Wave1 code/configs/README/evaluator files remain unchanged. Wave1B is additive: a new package, configs/lock, scripts, tests and WAVE1B.md. This report and the new implementation-status registry are separate. No Wave1B commit or push was performed.

## Selected Sources and Method Correspondence

| Method | Official source / frozen commit | Executed entry point | License |
| --- | --- | --- | --- |
| R-Clustering | [jorgemarcoes/R-Clustering](https://github.com/jorgemarcoes/R-Clustering/tree/3ed571eebe9eb8d399a0c995a6917f62d838f0fc), **3ed571eebe9eb8d399a0c995a6917f62d838f0fc**, submission | Notebook cell10 definitions and scientific assignments in cell12 | GPL-3.0; source remains external |
| FASA | [TheDatumOrg/MUFASA](https://github.com/TheDatumOrg/MUFASA/tree/9c05d415efee81fca1a87c1e91623db613ea2ecc), **9c05d415efee81fca1a87c1e91623db613ea2ecc** | Clustering/FASA_I_I/FASA_I_I.py:FASA_I_I, selected by the native univariate runner | **NOT STATED**; no vendoring or redistribution assertion |

Audit08's R-Clustering method description corresponds to *Time series clustering with random convolutional kernels* and its publication notebook. FASA corresponds to the univariate FFT/SBD aligned-centroid algorithm in *MUFASA: Fast and Accurate Multivariate Time-Series Clustering*, not the multivariate companion. These descriptions and pinned native implementations were inspected before adapter work.

Both clones stay under `G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/src`. HEAD, origin and clean working-tree checks passed. Working scientific files matched pinned Git blobs after checkout line-ending normalization.

R notebook Git blob SHA256: **118079ffae9f241878e47d4810fa37b90ecc3f765fd2a953c36d8ea39d8fb8d7**. Scientific cell10 SHA256: **a1013296dd31269aae53d436193388cc0b5bca23755201269d203eaec8985c18**; cell12 SHA256: **16a50a06d41c7d2be8fdf901867d999601ac5583ac1f2ecce13fd5aa94388430**. Notebook cell2/4 imports and cell8's float32 conversion were inspected but their loader/test/download paths were not run.

FASA core Git blob SHA256: **527e81e03616dd46a61c47b1c84187aa05f61b9f48839ab8bdbb95460af4b146**. No scientific source was copied into the public repository, including the archived local adapter bundles, which contain wrappers and contracts only.

## Implementation Differences and Scientific Consequences

These are thin non-scientific adapters; no methodological improvement or optional scientific refactoring was introduced.

**R-Clustering.** Cell10 is materialized verbatim as an external Python module, with only `import numpy as np` added to supply the notebook global. Its original signatures, cache=True, fastmath and parallel behavior remain. The scientific cell12 assignments execute unchanged, in their original order, as AST. The enclosing all-UCR loop, installs/downloads, source loader exceptions, labels/metrics and spreadsheet writing are excluded. The published500 request is supplied and known k comes from the common contract. A constructor proxy captures the fitted KMeans object without altering arguments or fit calls.

Common canonical pooling/z-normalization replaces the unpinned notebook loader context, as predeclared in Audit08; native float32 conversion, kernels, feature scaling and both PCA fits remain. No claim is made to reproduce unmatched original-paper scores. The hard-coded permuted84 kernel table, random quantile permutation, calibration-series sampling, dilation allocation and padding behavior are unchanged.

The native feature count is **84*floor(500/84)=420**. This is the known manuscript/code discrepancy, disclosed rather than repaired. The exact selector `argmax(explained_variance_ratio_ < .01)` is preserved; a zero result causes a classified failure before attempting the downstream zero-dimensional KMeans path. This early guard changes failure reporting, not the representation or learned model. No clamp, cumulative-variance replacement or label-informed dimension choice.

Host NumPy and compiled Numba are both seeded to the root seed. PCA/KMeans keep their original global RNG defaults (random_state=None); no independent head seed, restart policy or solver substitution was invented. Native sklearn1.4.1.post1 KMeans settings: n_init10, k-means++, Lloyd, max300, tol1e-4 and copy_x=True. Both PCA estimators keep svd_solver=auto and native defaults.

**FASA.** Call the external univariate FASA_I_I function on N,L,1 arrays. Keep random partition, zero centers, max100/stable assignments, native empty-cluster recovery, FFT/SBD alignment and aligned-mean centroid calculation. Membership indices are scattered to canonical rows and checked for exact uniqueness/coverage; missing samples are never silently assigned0 or omitted.

Scoped NumPy seeding is independent of the native itr output-filename parameter. Native zero-norm conventions (den=inf, centroid zscore/nan_to_num) remain; warning/zero-center metadata is recorded. No epsilon, alternative center, interpolation or scientific repair was added. Nonfinite returned centroids are reported as numerical failure rather than repaired. Native FASA exposes members/centroids, not an out-of-sample predict API: its membership extraction is timed and replayed without inventing another classifier.

**Resolved adapter bug.** Shortening the external derived-source directory from64 SHA characters to16 changes filesystem identity only; the full content hash, source text, source commit and scientific settings remain. This was an infrastructure correction, not a post-score scientific decision.

## Environment and Locks

New isolated shared CPU environment:

`G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/envs/wave1b-cpu-py39-v1`

**Python3.9.13**, NumPy1.26.4, SciPy1.13.1, scikit-learn1.4.1.post1, Numba0.60.0, llvmlite0.43.0, threadpoolctl3.5.0, psutil6.1.1, tqdm4.67.1 and pytest8.3.5. The complete **20-distribution** lock includes packaging tools: [requirements-wave1b.lock.txt](../../experiments_new/baselines2026/configs/requirements-wave1b.lock.txt).

- Lock SHA256: **6c1f18b31a205d3a8c04c2bff1e61ea609c759a37ab643316d5700cd9fa56308**.
- Environment fingerprint: **e31cf122e4cd92edc3b4c4d17fae4f34475dce9e57195f677606f0c0586ead8a**.
- Parent code commit: **5ff48377deac2b5934b450951eb41b1e3b08e7c9**.
- FASA executed adapter tree: **d61b119802efa5bf0dfb8ddf8a7b22a4af7e643ee9492b756084852d565015c2**.
- R executed/final adapter tree after path correction: **5b684201940171e0a855ffdab9ca1392b07e25ffc2fa50022e16921e4748026f**.

Both runs report adapter_commit=null and source_committed=false for the extension, with the parent SHA, full per-file/tree hashes and archived exact adapter source. The parent contains preserved Wave1, not the new extension. FASA's earlier valid source bundle was not rewritten to the later R tree hash.

Narrow requirements were inspected before installation. Core FASA imports NumPy/SciPy/tqdm; its broad UTS requirement list includes unrelated algorithms and was not installed wholesale. Canonical arrays bypass HDF5/aeon archive loaders; unused notebook pandas/result-writing cells do not require installation. Original Wave1/LoSTer environments and global Python packages were not altered. pip check passed. The install had one transient connection-reset retry and then succeeded; trusted Windows CA validation remained enabled.

Numba's independent RNG/compiled seed requirement was verified against [Numba0.60 documentation](https://numba.readthedocs.io/en/0.60.0/reference/numpysupported.html#randomstate-and-legacy-random-number-generation) and tested on actual fitted biases, not just host random draws.

## Common Data and Label Firewall

Reuse the unchanged common headerless official UCR reader. Only **ECG200** was used for new real fitting:

`G:/Articoli/Articoli da Completare/LoSTer 2026/data/ucr/ECG200`

**N=200, L=96, k=2; TRAIN100 then TEST100**. IDs are ECG200/TRAIN/0..99 then ECG200/TEST/0..99, with no missing or duplicated samples.

| Input | SHA256 |
| --- | --- |
| TRAIN | bd08b37147ebf205231ad43e02c42dd5c14bc348f4ccc7614f85634c8916bf21 |
| TEST | c8ac02caf98be23c47c044fde026219301ceaf0b2f97bc107bd0cd96330b867c |
| Common normalized array, both methods | fcf338c04c4e7fb727a27b4f60c58bf37592a5524f825f155a2e2f8842881aef |

Common per-series population z-score uses float64 with no imputation/deletion or undefined-variance repair. R then uses its native float32 representation; FASA retains native centroid ddof1 transformations. The trusted reader transiently derives permitted oracle k; TrainingInput contains no membership labels. Fit APIs and selected notebook scientific code receive X,k,seed only. Evaluation labels are loaded by the independent evaluator after immutable prediction export. No supervised model/checkpoint/hyperparameter selection.

No other Development8 or Final36 task was fitted. Existing historical scores were not recomputed. Debug seed1 was used only for tiny stochastic replay, not for ECG200 or tuning.

## Unit and Regression Tests

| Validation attempt | Wave1B + common tests | Unchanged Wave1 regressions | Failures / errors / skips |
| --- | ---: | ---: | --- |
| validation-001, initial layout | 71 passed,31.00s | 57 passed,36.95s | 0 / 0 / 0 |
| validation-002, path correction + regression | **72 passed,20.67s** | **57 passed,24.63s** | **0 / 0 / 0** |

Final gate: **129 passing executions**. Across the two attempts:257 passing executions. Counts are not unique-case counts because common contracts and unchanged regressions are intentionally repeated in their respective locked environments. No failing unit test was concealed; the initial filesystem bug appeared only in the deeper real-dispatch layout despite passing the shorter validation path.

Coverage includes source commit/origin/blob/cleanliness, exact notebook definitions and scientific AST, native full-state feature/PCA/KMeans equivalence, 420 features, native dtype/kernel/dilation behavior, independent compiled RNG and actual bias changes under fixed host permutation, PCA zero/boundary/nonfinite-ratio examples, FASA native center/membership equivalence, index ordering, duplicate/missing/out-of-range/noninteger members, seeded native empty-cluster recovery, actual initial partition draws at0/0/1, zero-norm distance and cancelling-centroid warnings, environment lock integrity, and shortened-path/full-hash preservation.

The reused 40 common tests exercise canonical sample order/headerless reading, raw and metadata hashes, malformed submissions, integer boundaries, immutable export, collapse acceptance, normalization agreement, independent known metric examples and exact frozen public metric agreement. All57 original Wave1 tests were rerun without changing its source/lock; outputs/caches/temp paths were new G: paths, not prior runs or environment directories.

## Native Edge Cases and Seed Verdicts

**R feature/PCA/Numba verdict: PASS WITH RETAINED DOMAIN CAVEAT.** Native bias arrays replay exactly with compiled seeds; altering only the compiled seed while holding host permutation fixed changes biases. Interpreted NumPy seeding alone does not reset Numba's stream. The actual kernel table/default dilation logic is used unchanged. The native PCA rule returns0 when no ratio meets the threshold, when the first component meets it, or for all-NaN variance ratios. A repeated-series valid-input fixture triggers native invalid-variance warnings and classified PROTOCOL INCOMPATIBILITY; the test passes because no clamp or rescue is performed. This is an expected domain-failure test, not an omitted real score.

**FASA ordering/seed verdict: PASS WITH RETAINED ZERO-NORM CAVEAT.** Out-of-order member lists scatter correctly to original rows. Invalid membership fails explicitly. Spying on the unchanged native randint call confirmed identical actual initialization at0/0 and a different initialization at1; itr does not seed. Native empty-cluster recovery replays exactly. Zero norm yields the native zero-correlation convention; an exactly cancelling centroid produces a native invalid-divide warning and native zero conversion. No new repair. Actual tiny/ECG200 runs had zero warnings and no zero final centroids.

Fixed-seed replay is demonstrated on this exact CPU/numeric environment, not promised across future platforms/backends. Changed seeds need not change final quality or an equivalent optimal partition.

## Standalone Synthetic Smoke Tests

Both validation attempts ran three full-native tiny fits per method: **N48,L64,k2**, fixture generator seed19, fit seeds **0,0,1**. Thus six dedicated tiny fits per gate,12 across both gates, in addition to unit native-parity/negative fixtures. No real-data fit occurs in these suites.

| Method | Same seed labels / centers / stochastic state | Changed-seed native state | Checkpoint/export/replay | Warnings / occupied clusters |
| --- | --- | --- | --- | --- |
| R-Clustering | Exact / exact / exact | Bias hash changed | PASS | 0 / 2 of2 |
| FASA | Exact / exact / exact | Actual initial partition hash changed | PASS | 0 / 2 of2 |

After immutable export, synthetic diagnostic ARI/NMI/RI/ACC were all1.0. These were plumbing checks, not scientific parameter selection or competitiveness evidence. Native recipe/settings were unchanged between initial and corrected gates.

## ECG200 Integration Smoke

Exactly one completed native fit per method at seed0. One additional R setup dispatch failed before any kernel-bias/feature/scaler/PCA/KMeans fit. It is retained separately, not called a completed fit or assigned an accuracy. R's successful completion used the same native source/recipe/input/environment/root seed after renewed validation of the path-only fix. FASA was not rerun.

| Method | Native result | Pipeline seconds | Inference/extraction | Sampled peak process-tree RSS MiB | Prediction/eval/replay/resume |
| --- | --- | ---: | --- | ---: | --- |
| R-Clustering | 420 features,12 PCs; native PCA auto chose full; selected KMeans6 iterations among10 restarts | **15.9667679** | Stored transform/PCA/KMeans full-pool inference **0.2517667s** | **264.809** | PASS / PASS / exact / no fit |
| FASA | Native final memberships,2 occupied clusters,0 zero centroids | **0.6834042** | Native member extraction **0.0001288s**; no independent native predict API | **144.629** | PASS / PASS / exact / no fit |

R ECG200 dilations: **[1,3,6,11]**, per-dilation allocations **[2,1,1,1]**, totaling5 per each of84 kernels. Bias SHA256 **bf949902f84d9a8cd88cc36bc963e6f13809dfd204bb82c5498f7c59fc5d56ce**; feature SHA256 **a99e47946fd2a46dfe566928b79d3cfc23357476865c1363831621bb58192be4**. The second PCA fit, not reused first-fit components, is preserved.

Scores are **integration evidence only**:

| Method | ARI | NMI arithmetic | RI | ACC |
| --- | ---: | ---: | ---: | ---: |
| R-Clustering | 0.2630289489 | 0.1808968871 | 0.6333668342 | 0.7600000000 |
| FASA | 0.2358804474 | 0.1859296237 | 0.6181407035 | 0.7450000000 |

No tuning, competitiveness claim, seed substitution, significance test or scientific decision was made from these scores. Both successful runs had zero warnings and no collapse. Recorded completion times were **2026-10-09 11:02:21 local (FASA)** and **11:08:35 local (R)**.

## Independent Evaluator and Artifact Replay

Unchanged version **baselines2026.metrics.v1**, SHA256 **e34c448fa192adf5608809a555a52d24ab128b5cf6c362f56321819f5a28ded7**. Separate processes reload frozen JSON predictions, canonical labels and matching dataset hashes, then compute primary ARI, explicit arithmetic NMI, integer pair RI and rectangular Hungarian ACC. The evaluator never selects a checkpoint or changes assignments.

Both versioned submissions contain200 finite integer predictions and unique IDs in exact TRAIN/TEST order, with source/config/environment/input/adapter hashes and SUCCESS status. Utilization is diagnostic, not an all-k requirement. Self-produced trusted pickle checkpoints store R kernels/biases/scaler/both PCA states/KMeans/labels, or FASA centers/member lists/labels. Fresh processes reproduce assignments without fitting. Identical-identity runner calls report REUSED WITHOUT FIT and preserve prediction/model hashes.

External evidence root:

`G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/wave1b`

| Evidence | Location / run ID |
| --- | --- |
| Original and corrected gates/tests/synthetic | validation-001/ and validation-002/ |
| Failed pre-fit R setup and successful FASA | development-smoke-001/; original mixed summary retained |
| FASA successful result | runs/fasa/**18dd57ad80a54aa5bbcecf5f6497426b674dd6508eae69142b1efcef4b5bd0f4**, under development-smoke-001 |
| R-only successful completion, evaluation/replay/resume | development-smoke-002/ |
| R successful result | runs/r_clustering/**7c37a77ed155289ad658220a2f1e01500650f017988dd094f6ff33541beb2eb2**, under development-smoke-002 |
| Diagnosed path failure / Windows probe | r-cache-path-failure-assessment.json and path-probe/ |

Manifests and exact36-file adapter bundles identify both executed trees. No prior source bundle or initial failed summary was rewritten to make the first mixed run appear successful.

## Timing and Resource Contract

CPU only, Intel Core i7-8700K,6 physical/12 logical CPUs,16,977,186,816 host RAM bytes. BLAS/OpenMP/Numba thread environment and threadpoolctl are1. GPU/batch/VRAM fields are not applicable. Monotonic pipeline stages include common normalization, native initialization/feature fitting/transform/scaling/both PCA fits/all KMeans restarts or all FASA iterations and ordered extraction. Cold eager R JIT is included: **15.1553681s** in native definitions/JIT. This illustrates why timing only final KMeans would be invalid.

Source preparation/verification overhead inside the adapter is included and disclosed. Raw decoding, checkpoint/export/reload and central evaluation are excluded. R uses a new real-run cache; its failed setup cache was not reused. FASA inference_seconds=null honestly reflects its native membership-only interface; member extraction has its own stage. CPU process-tree RSS is sampled20ms and may miss short peaks or count shared pages across processes. These are smoke measurements, not publication-quality efficiency. Campaign warmup/repeat/cache/timeout/admission gates remain unfulfilled.

## Failures and Historical Infrastructure Ledger

| Event | Actual outcome / classification | Resolution and scientific consequence |
| --- | --- | --- |
| Historical Wave1 package install | TLS certificate failure, DEPENDENCY ISSUE | Trusted Windows CA bundle; no TLS bypass or scientific change; retained in Audit09 |
| Historical Wave1 validation-001 | 56 pass/1 fail, ADAPTER BUG | Source egg-info was counted as installed environment; fixed site-packages enumeration, separate source provenance |
| Historical Wave1 validation-002 | 56 pass/1 fail, diagnostic bytecode ADAPTER BUG | Correct clean-source rejection; archived bytecode, disabled bytecode writes; source unchanged |
| Current new-env installation | One connection-reset automatic pip retry, then successful install/check | No failed scientific fit or dependency/recipe substitution |
| Current draft serialization | Automatic approval review could not complete due service usage limit; action not executed;4500-byte partial external draft retained | Normal approval path resumed on user's continue instruction; no approval bypass |
| Initial R ECG200 dispatch | failure.json initially UNKNOWN; traceback/local probe diagnoses **ADAPTER BUG**, Windows MAX_PATH | Failed cache filename270 characters,parent existed; standard long-path probe failed while extended path worked. Shorten only external source directory, retain full SHA/source/recipe; revalidate before remaining actual fit |
| Native zero-PC/cancelling-centroid tests | Expected PROTOCOL INCOMPATIBILITY / native numerical warning fixtures | Negative tests passed; no clamp/alternative centroid/imputation; retain domain caveats |
| Completed R/FASA ECG200 fits | SUCCESS,zero warnings,no collapse | No new source/scientific blocker or score-driven recovery |

The initial R failure stopped in common_normalization/native_definitions_and_JIT (8.5848882s in failed native-JIT stage), before compiled_rng_seed/kernel_bias_fitting or any representation/cluster training. Thus one completed native fit per method remains the exact total. All failed-dispatch logs and metadata are preserved. The failed JSON's fallback UNKNOWN classification remains historical; the separate assessment and this report supply the diagnosis.

## Readiness and Implementation Status

The new [implementation-status registry](../baselines/implementation-status-2026.json) is separate from the immutable historical registry. It records historical readiness alongside current implementation status, evidence reports, sources, environment/adapter hashes, run IDs and remaining conditions.

| Method | Verdict | Remaining condition |
| --- | --- | --- |
| R-Clustering | **READY WITH WARNINGS** | Native500-to420 discrepancy remains disclosed; valid inputs can select zero PCs and must fail without scientific rescue; full-panel resource/domain coverage is not established |
| FASA | **READY WITH WARNINGS** | Source license NOT STATED; private local execution only, no permission to redistribute claimed; native zero-norm caveat remains; full-panel coverage not established |

Current11-method counts: **3 READY FOR DEVELOPMENT BENCHMARK** (Euclidean,k-Shape,KASBA), **2 READY WITH WARNINGS** (R,FASA), **3 CONDITIONAL_NOT_IMPLEMENTED** (TS2Vec,FCACC,TFMCC), **3 BLOCKED** (CKM,DTCC-2023,CDCC). All five implemented methods have one-task integration evidence; none gains CORE44 authorization.

## Frozen/Protected Safety

LoSTer remains **LoSTer-Legacy-Clean + MonotoneWarp**, frozen **cab7b6ffecab92bfcf6fe46e73e32dc0c040ee59 / loster-2026-frozen**. Final verification compares the41 frozen scientific source/configuration files to Git checkout-filtered tag bytes, checks95 original protected hashes and84 current historical-file hashes, and verifies all321 previous Wave1 output files unchanged. Historical audits/registry/manuscripts/legacy sources/results are untouched.

Protected-path Git diff and all preexisting tracked-file diffs are empty. External source commits/working trees remain clean. Only14 additive framework files plus this report and the new status registry are untracked; no staging, Wave1B commit or push. HEAD and origin/revision-2026 remain the preservation commit5ff48377deac2b5934b450951eb41b1e3b08e7c9. Final evidence is saved externally as final-safety-and-artifact-checks.json. No ignored generated repository output is introduced.

## Remaining Blockers and Next Wave

No new scientific source blocker was introduced. CKM still lacks official code/complete faithful recipe; DTCC retains TF2-vs-TF1 dependency, disconnected cluster gradient, label-best output/extraction and correspondence concerns; CDCC retains augmentation-selector and one-step temporal-axis/source behavior concerns. TFMCC's dependency/global resource recipe is unresolved. None was trained or promoted.

Recommend Wave2 **official TS2Vec+k-means, then FCACC**, with source/label-free checkpoint/extraction/environment/head-seed contracts frozen before fitting; **TFMCC after dependency and global resource configuration verification**. Retain CKM/DTCC/CDCC in source-resolution work. Further datasets, final seeds, publication efficiency repeats and campaign execution require new authorization. **CORE-44 authorization: NO.**
