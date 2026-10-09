# LoSTer 2026 Baseline Wave 2A

## Executive Summary

Wave 1B was preserved as commit 4b66880a3face2d26e07dbb89e16e9f950567ce3 and pushed only to origin/revision-2026. Git was clean before Wave 2A. Both new external source trees match every tracked Git blob at their authorized pins. The adapters, tests, locks and this report remain uncommitted.

TS2Vec verdict: READY WITH WARNINGS. FCACC verdict: BLOCKED. A newly demonstrated source/manuscript discrepancy prevents FCACC's explicit clustering-distance term from supplying encoder gradients. Its relevant scientific test remains failed, so the authorized SyntheticControl integration fit was withheld under the stop policy. No loss repair, reduced budget or favorable-score rescue was made. CORE-44 authorization: NO.

## Frozen Source Identities

TS2Vec: [official frozen tree](https://github.com/zhihanyue/ts2vec/tree/b0088e14a99706c05451316dc6db8d3da9351163), SHA b0088e14a99706c05451316dc6db8d3da9351163, MIT.
FCACC: [official frozen tree](https://github.com/Du-Team/FCACC/tree/78b5e5a138ed8c83fea64e5b6668a5cded79d2dc), SHA 78b5e5a138ed8c83fea64e5b6668a5cded79d2dc, license NOT STATED. All source remains in private external G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/src. No redistribution permission is asserted and no scientific source is vendored.

Frozen LoSTer remains LoSTer-Legacy-Clean + MonotoneWarp at loster-2026-frozen, cab7b6ffecab92bfcf6fe46e73e32dc0c040ee59. Pre-fit checks proved 41 scientific files byte-equal to the tag's checkout-filtered bytes, 370 protected files, 36 prior framework files and all 697 prior Wave1/Wave1B artifact files unchanged. Final verification is recorded separately below.

Executed adapter parent: 4b66880a3face2d26e07dbb89e16e9f950567ce3; uncommitted adapter tree: 7a2d78f05166dc718292c65bf1204fe1d7a1db04430c06bb00566831d798757f. The TS run archives every framework file and its per-file SHA; adapter_commit=null truthfully describes execution.

## Environment Reproducibility

No interpreter was installed. Existing Python3.7.9/3.9.13 created isolated environments on G:; default Python, LoSTer venv and locked Wave1/Wave1B environments were not installed into.

| Method | New environment | Core scientific versions | Complete lock |
|---|---|---|---|
| TS2Vec | wave2a-ts2vec-py37-v1; Python3.7.9 | torch1.8.1+cu111, NumPy1.19.2, SciPy1.6.1, sklearn0.24.2, pandas1.0.1 | 23 packages; requirements-wave2a-ts2vec.lock.txt |
| FCACC | wave2a-fcacc-py39-v1; Python3.9.13 | torch1.10.0+cu113, NumPy1.21.4, SciPy1.10.1, sklearn1.3.2, pandas2.0.3, matplotlib3.5.0, Pillow8.4.0 | 31 packages; requirements-wave2a-fcacc.lock.txt |

TS lock SHA74c0b01411f906497b6785f771e083e3e20512fab8eb5f48a36be9e5ef7a3641; environment SHA8eca15e6debe4a870994892db271a766430a1c1c9c622e49afb7ee6d9a802913.
FC lock SHA3e9d70904773f833229ba43ca1ff4eaabb4c8db09142e5cf0a098cd71c0626f6; environment SHAe85651315ac15fa4ea1981bac9afadc372f5fce929d3ef681100a2598233534b. Full package lists are in the tracked-candidate locks; exact installed-distribution comparison and pip check passed for both.

TS author core pins were preserved using existing Python3.7; no modern PyTorch substitution. Python3.7 requires importlib-metadata6.7.0 as a packaging shim for the unchanged common provenance module. Its worker needs Anaconda Library/bin on process PATH for SSL DLLs. Forecast-only Bottleneck/statsmodels are outside the encoder import closure. The isolated interpreter has no system site packages.

FCACC's frozen README actually supplies recommended core versions, contrary to Audit08's unpinned-framework summary. It still is not a full author lock. Core recommended pins were adopted before native execution; the existing Python3.9 substitutes for absent recommended3.8.10. Unused torchvision, notebook, image and visualization extras are excluded. This packaging adaptation does not repair the scientific blocker.

Actual CUDA matmul passed in both environments on GTX1080, compute capability(6,1). Runtime/cuDNN: TS CUDA11.1/cuDNN8005; FC CUDA11.3/cuDNN8200. Driver552.22, 8GiB VRAM. Native CPU encoder forwards/backward passed. FCACC's full class is CUDA-specific (hardcoded CUDA tensors), so a full CPU fallback was not claimed.

## TS2Vec Scientific Correspondence

Native encoder, crop overlap, NumPy masks, dropout, hierarchical instance/temporal contrast, AdamW, SWA updates and architecture are imported directly. Use CLI B8, output320, hidden64, depth10, lr.001, max_train_length3000 and temporal_unit0. Preserve native min(B,N), drop_last and size-based budget:200 updates if input.size<=100000, otherwise600. ECG200 uses200; no budget tuning.

The official init_dl_program seed offsets are retained: root0 -> Python0, NumPy1, TorchCPU2, CUDA3. cuDNN deterministic=True is an explicit reproducibility setting; benchmark=False and TF32=False follow the official helper. A cuda:0 device string is required by this old helper. Scoped state restores caller RNG/backend flags. Tiny direct-native fit equivalence verified exact averaged weights, full-series vectors and ordered predictions. Native contrastive objective gradients passed.

## TS2Vec Full-Series Clustering Adaptation

The common clustering task uses pooled TRAIN+TEST per-series population z-normalized float64 inputs, converted to native float32 [N,L,1]. This is an explicitly declared adaptation from native TRAIN-only SSL/supervised TEST classification. Neither tasks/classification.py nor the supervised head/search runs.

Call official encode(X,encoding_window='full_series') with native batch8, no sliding, no multiscale, no concatenated pooling. One ordered vector per original sample. Fit one independently derived kmeans-phase head, known k, k-means++/ten minimum-inertia restarts/max300/tol1e-4/Lloyd. sklearn0.24.2 spells the Lloyd algorithm full; this API-name mapping preserves its native Lloyd implementation and is disclosed, rather than upgrading sklearn/NumPy.

The root seed identifies one complete pipeline, not five heads selected by target ARI. Representations, deployed averaged state, n_averaged, raw encoder, extraction config, head/centers, seeds/input hashes and ordered predictions are persisted with artifact hashes.

## FCACC Scientific Correspondence

The native released recipe was verified: B8, d64/h64/depth10, pretrain100 at AdamW.001, joint30 at hardcoded.0001, m1.5, w_c.2/hard_w.2, T1=2, confidence.95, gamma.5, scaling_rate.8. Class d32/MaxIter100 defaults are not mixed with runner overrides. Native source is preserved.

Two Audit08 descriptions need qualification before any real fit. The README supplies recommended core pins. Also DataTransform actually invokes additive jitter(sigma.8), although Audit08 calls the third view scaled. The adapter preserves released jitter, not a prose-driven scaling substitution.

New correspondence blocker: the paper's [clustering and total losses, equations8 and11](https://dumingjing.github.io/files/paper-21_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series/2026_PR_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series.pdf) describe joint representation/clustering optimization. Released Finetuning obtains distance-term features through encode_with_pooling, which runs the averaged network under torch.no_grad and returns detached CPU/GPU representations. Centers are ordinary non-gradient tensors and memberships are detached. The computed clustering term has requires_grad=False and no path to the optimized raw encoder. Synthetic tensor verification failed the explicit published-objective gradient check; the source was not repaired.

This is a source/manuscript inconsistency, with author intent unresolved. It does NOT mean all cluster awareness is absent: model-generated pseudo-labels still affect the trainable cluster-aware contrastive term. The concern is the separately weighted fuzzy distance term. Reconnecting it would change encoder/loss behavior and requires a reopened scientific protocol/source gate.

Further source-mode observation: Kmeans_model_evaluation sets the raw encoder to eval and does not restore train; Pretraining sets train only before its epoch loop. Thus later pretraining epochs run in eval mode as released. This conclusion comes from inspected control flow, not an unexecuted100-epoch measurement. The adapter intentionally retains the transition; it does not silently enable dropout/masking in subsequent epochs.

## FCACC Fuzzy-Clustering Protocol

The implementation retains native fuzzy membership/center equations, every cluster loop/forward, dilated encoder, raw/SWA distinction, augmentation/masking, AdamW/loss scheduler and final assignment argmax. Native center initialization random_state0 remains independent of root RNG; pinned sklearn1.3.2 n_init='warn' resolves to10. Warnings remain visible. No label-best epoch or per-dataset choice is introduced.

Per-epoch native diagnostic loader/forward/KMeans/mode/RNG effects remain; only true-label metrics, metric prints and feature-file exports are filtered from their verified AST. Constant dummy target tensors satisfy the native tuple API. Synthetic comparison with original diagnostic methods (metric stubs returningNone) proved exact SWA, centers, representation and prediction state, confirming that removing label diagnostics did not remove computational/RNG effects.

Final inference retains native unused N,L,d float64 allocation and extra raw-encoder forwards. No memory optimization was adopted. Predictions scatter by canonical row indices; duplicates, out-of-range indices or incomplete coverage fail. Save active averaged net and centers plus raw encoder/modes. Do not confuse native raw-only pretraining/final files with a complete deployed checkpoint.

The scientific gradient test remains failed. Final benchmark readiness is BLOCKED; a passed environment/checkpoint/order test does not override source correspondence.

## Ground-Truth Label Isolation

The unchanged common training capability supplies only raw sequences, canonical IDs, hashes and known k. Class memberships are absent from adapters and native training/model-selection calls. FCACC receives constant API placeholders, never real target membership. No target metric enters optimization, scheduling, checkpoint/seed/hyperparameter selection or retries.

The central evaluator separately reopens true labels only after immutable export; ARI primary, explicit arithmetic NMI secondary, pair-count RI and rectangular Hungarian ACC descriptive. Its unchanged source hash is e34c448fa192adf5608809a555a52d24ab128b5cf6c362f56321819f5a28ded7. Synthetic metric fixtures do not authorize real-data label diagnostics or selection.

## Synthetic/Regression Tests

Final same-tree gates: TS validation-ts-005, 56 passed/0 failed/errors/skips. FC validation-fcacc-002,55 passed/1 scientific correspondence failure/0 errors/skips. Unchanged Wave1 suite in its locked original environment:57 passed/0 failures/errors/skips. Final executions total168 passed and1 failed; common40 contracts repeat across environments, so these are execution counts rather than169 distinct tests.

Tests cover source/Git identity and imports; exact installed locks; CPU tensors/backward; actual CUDA; native training/objective; shapes; active SWA/counter versus raw state; strict checkpoint reload and corrupted/missing state rejection; full_series equivalence; known-k/no-y API; seed0 replay and seed1 change; head seed; FC shuffled scatter and coverage failures; native diagnostic execution equivalence; common predictions/metrics/data identity; synchronized timing and CPU/GPU telemetry.

Generated tiny N16,L16,k2 tests use explicit TS three-update or FC one-pretrain/two-joint budgets only. Full real-data budgets were not reduced. Repeated seeds0,0,1 replay active states/representations/predictions exactly; changed seed changes learned state/representations without requiring a changed clustering partition. Native FC center seed0 is not randomized.

Historical validation outcomes and all raw logs are retained externally: 

| Attempt | Tests | Passed | Failures | Errors | Skips |
|---|---:|---:|---:|---:|---:|
| validation-fcacc-001 | 55 | 54 | 1 | 0 | 0 |
| validation-fcacc-002 | 56 | 55 | 1 | 0 | 0 |
| validation-ts-001 | 54 | 53 | 1 | 0 | 0 |
| validation-ts-002 | 55 | 54 | 1 | 0 | 0 |
| validation-ts-003 | 55 | 45 | 0 | 10 | 0 |
| validation-ts-004 | 55 | 55 | 0 | 0 | 0 |
| validation-ts-005 | 56 | 56 | 0 | 0 | 0 |

Resolved historical failures include oracle backend/batching assumptions and a device-index adapter error; the published-gradient failure remains a real scientific gate failure, not xfail/skip. See Failures and Warnings.

## ECG200 TS2Vec Smoke

SUCCESS, one full native ECG200 seed0 fit: TRAIN100+TEST100, N200,L96,k2;200 updates/eight complete epochs, SWA n_averaged201. All200 losses/representations finite; two occupied clusters. Root stream seeds Python0/NumPy1/TorchCPU2/CUDA3, head seed525343541.

Integration evidence only: ARI=0.2586500049186616, arithmetic NMI=0.1649530914481815, RI=0.6333668341708543, ACC=0.76. No competitiveness, selection or statistical-comparison claim.

Raw TRAIN SHAbd08b37147ebf205231ad43e02c42dd5c14bc348f4ccc7614f85634c8916bf21; TEST SHAc8ac02caf98be23c47c044fde026219301ceaf0b2f97bc107bd0cd96330b867c. Normalized float64 SHAfcf338c04c4e7fb727a27b4f60c58bf37592a5524f825f155a2e2f8842881aef; native float32 input SHAb6f15c3970dbc0d1235e67388fca6a3edf9b2d3553609a549a89abdd0382727c.

Run ID ec532757cbe6d28620a5fe9948e2b43ab95ce317286c011cf52e3d7b02f32a85; config SHAb9a8e84ef037816b38c001ad96d9b00226675aa6e0e662a3fdafc768b892394a. Artifact folder development-smoke-001/runs/ts2vec_kmeans/ec532757cbe6d28620a5fe9948e2b43ab95ce317286c011cf52e3d7b02f32a85. No second fit or other real dataset was used.

## SyntheticControl FCACC Smoke

NOT RUN. Canonical data preflight verified N600,L60,k6, TRAIN300+TEST300, and expected raw hashes. The full100+30 fit was withheld because the relevant gradient gate failed before real-data training, as required by the stop policy. There is no SyntheticControl FCACC prediction, metric, full-fit timing, GPU peak or real checkpoint result. No arbitrary score or time is imputed. There was no FCACC OOM or real fit retry.

Synthetic tests executed the released training behavior unchanged and characterize the adapter; they do not establish paper-faithful scientific eligibility or full SyntheticControl integration completion.

## Prediction and Checkpoint Validation

TS exported200 integer predictions against200 unique canonical IDs in exact TRAIN-then-TEST order, plus200x320 full-series representations. Active SWA/raw states and head centers/config/hash/seed identity saved; in-process and fresh-process reload reproduced representations and predictions bit-for-bit. No-fit resume verified every model/prediction/source-bundle hash and reused without training. The failed FC gate also actively refused real dispatch before loading/fitting a model; no FC real-fit ledger exists.

Central evaluator ran independently in the unchanged wave1-cpu-py39-v1 lock, using the same frozen source as earlier waves. Prediction SHA1e29a8521d4b834818e42dffc1a2219df241de65a2a9ab586b7d632f0bf0d6cb; evaluation SHA388b05d2962d8c31df7305061044432f8009709014443e0efe4156817046227c. These are tied by the evaluator's submission hash.

FC synthetic checkpoints restore active SWA/raw encoders, n_averaged/modes and fuzzy centers; all tiny representations and labels replay exactly. Missing states, raw-for-SWA substitution, changed weights and wrong input identities are rejected. The deliberate shuffled-index oracle compares exact native results under identical batches/conditions; canonical coverage is strict. No full real-data FC checkpoint exists.

All artifacts stay outside Git, under G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/wave2a. Complete locked identities, archived adapter bytes and output hashes support independent replay. Successful identical run reuse performs no fit; an exclusive authorization ledger prevents a new TS fit in a different output root.

## GPU/CPU Resource Measurements

TS real full pipeline 497.8291443s.

| TS real stage | Seconds |
|---|---:|
| canonical_prediction_validation | 0.0008214 |
| common_preprocessing | 0.0006216 |
| downstream_kmeans | 0.7378589 |
| full_series_extraction | 32.1972464 |
| initialization | 0.0169717 |
| representation_training | 462.3901343 |

Source verification/helper overhead between timed stages is included in total. GPU peak allocated60,545,024bytes (57.740MiB), reserved67,108,864bytes (64.000MiB); sampled CPU tree RSS2,150,084,608bytes (2.002GiB). nvidia-smi 461 samples: device utilization min0/mean13.84/max86%; device-used memory range2780-3173MiB. Background allocations are included in device-used figures.

FC tiny synthetic pipelines at seeds0,0,1 in the final suite took56.2623444s,65.6015591s,48.1975820s. Peak allocated13,784,576/18,227,712/22,104,064bytes; reserved14,680,064/18,874,368/23,068,672bytes. Peak process-tree RSS2,465,624,064/2,495,614,976/2,484,224,000bytes. These are tiny tests with prior live model/checkpoint states retained in the process; they are not isolated full SyntheticControl fit estimates. No full FC pipeline/resource value is available.

GPU jobs were sequential; no concurrent TS/FC training. GPU peaks use synchronized torch allocator counters. nvidia-smi samples report utilization and device-total memory including unrelated WDDM/background activity; they are not process-exclusive GPU bytes. Host process-tree RSS is sampled20ms and may miss brief peaks. Full pipeline includes cold native source/initialization and all scientific stages; interpreter/package imports/raw decoding and checkpoint/export/evaluation/replay are separately excluded as integration scope.

Both methods have crop-length-quadratic temporal similarity. FC additionally repeats encoding/crops by k and allocates unused N,L,d float64 arrays. Long sequences/many clusters remain unmeasured risks; there is no precise44-task forecast from these tiny/single-task timings. No ShapesAll or StarLightCurves fit occurred.

## Original vs Adapter Differences

Shared task changes: canonical headerless TRAIN+TEST pooling, per-series population z-score, known-k firewall and independent evaluator, as predeclared. No sample deletion, imputation or method-capacity change. TS's downstream head is the declared clustering-task adaptation from its classification paper.

Engineering changes: isolated package/import closure; Python3.7 metadata backport/process DLL path; cuda:0 selection; explicit scoped root/backend seeds; old-sklearn Lloyd API name; source/provenance checks; active-state checkpoints; no-fit identity/replay; prediction scatter; finite/coverage guards; synchronized timing/telemetry and external output paths. FC label diagnostics are removed while preserving their computational effects, proven on tiny inputs. Large unused inference arrays remain.

No loss, gradient, centroid update, augmentation distribution, encoder depth/dimension, contrastive batch/negatives, precision, crop policy or real training budget was repaired. The detected FC graph/mode behavior is disclosed and blocks paper-faithful readiness. No scientific choice was made after smoke scores.

## Failures and Warnings

Preserved infrastructure history:
1. First TS Python3.7 pip attempt could not import SSL because Anaconda DLLs were not on its venv process PATH. Adding only that worker DLL path recovered SSL, using the existing trusted Windows CA bundle; no TLS bypass or base installation mutation.
2. Initial numeric installs emitted old-pip/PowerShell stderr status artifacts even when packages installed. Installed-distribution verification, actual imports and pip check distinguish successful installs from real failures.
3. FC preliminary environment used candidate modern numerics/torch1.8.1 download before the frozen README was fully extracted. The download was interrupted before native use; native README torch1.10.0+cu113 and core numeric pins replaced candidates before any model execution. A leftover contourpy NumPy incompatibility was removed; matplotlib's packaging dependency setuptools-scm6.4.2 restored. No scientific score informed these packaging choices.
4. Native Torch download had a connection-reset retry; final torch import/CUDA and locks verified success.
5. TS validation001:53 pass/1 oracle failure. Validation002:54 pass/1 continuing oracle failure. Direct oracle differed in batch/layout/backend policy; ultimately use native DataLoader and identical deterministic cuDNN settings, retaining exact equality. Adapter native-equivalence and checkpoint replay had already passed.
6. TS validation003:45 pass/10 shared-fixture errors from official helper set_device(cuda) lacking an index in torch1.8.1. Corrected adapter device string to cuda:0. No model training occurred in that failed fixture. Validation004:55 pass; renewed final005 adds the scientific-gradient test and passes56.
7. FC validation001:54 pass/1 ordering-oracle representation equality failure under different batch/backend conditions; predictions and seeded checkpoint replay matched. Identical-batch native scatter oracle remains exact. Final002:55 pass/1 published-gradient failure. No assertion was relaxed to tolerance and no failing science test was suppressed.
8. Final FC warning set includes native sklearn future n_init warning (effective10 retained) and Matplotlib/pyparsing deprecations. These are distinct from the unresolved gradient correspondence blocker.
9. Prior Wave1/Wave1B historical failure logs, including R's resolved Windows Numba cache MAX_PATH infrastructure failure, remain unchanged; Audit09/10 were not rewritten.

Two recovered report-builder failures are preserved as report-build-001-failure.json and report-build-002-failure.json: a successful PowerShell replay log was initially decoded as UTF8 instead of UTF16LE, then the new decoder referenced an unimported Path name. BOM-aware decoding and the imported pathlib.Path corrected the infrastructure checks with assertions retained. An initial Audit04 read used a nonexistent filename; the actual 04-method-and-experiment-plan.md was then read. These failures changed no scientific output.

No real-data failed fit, OOM, capacity rescue, seed replacement or accuracy-based tuning occurred in Wave2A. FC's failure is a source/scientific gate, not an observed SyntheticControl numerical failure.

Additional recovered infrastructure attempt: the first independent evaluator CLI used --submission instead of the existing --predictions argument; argparse refused it before evaluation. The retained failure log precedes the successful separate evaluation-002 log. No fit, checkpoint, labels or metric definition was changed. The final failed-FC-gate dispatch is an intentional negative admission check, refused before real fitting.

## Updated Baseline Status

The separate implementation-status registry preserves all other nine method records and all historical Audit08 findings. It records each new method's source/environment/integration/resource/readiness evidence independently. FC's environment/adapter contracts can pass while scientific readiness remains blocked.

Final counts: {"READY FOR DEVELOPMENT BENCHMARK": 3, "READY WITH WARNINGS": 3, "CONDITIONAL_NOT_IMPLEMENTED": 1, "BLOCKED": 4}. Audit08's source registry and reports04/08/09/10 remain immutable. No method was dropped from the eleven-baseline roster. Readiness is for a future development benchmark, not all44 feasibility or CORE-44 authorization.

## Implications for TFMCC

Do not implement TFMCC here. A next wave needs a verified import closure and complete isolated lock, a real GTX1080 CUDA operation for that exact candidate, source/paper optimizer correspondence, active averaged-state persistence and ordered inference, synthetic objective/gradient tests and a globally frozen resource policy before target scores. Today's legacy-wheel CUDA success does not prove an unrelated modern TFMCC wheel supports sm61.

Native B256, d320/depth10, length512 flatten/head256 and1400 epochs remain a large8GB risk. Smaller batches change negatives/BatchNorm; no resource ladder or capacity change is authorized here. The FC gradient finding makes objective-to-encoder connectivity a necessary early TFMCC check.

## Remaining CKM/DTCC/CDCC Blockers

Unchanged: CKM lacks an official executable source/complete recipe; DTCC-2023 has conflicting TF dependency declaration, disconnected cluster-contrast gradient and label-selected outputs; CDCC has released selector/time-axis/evaluation randomness/small-batch concerns. No implementation or repair of those methods was attempted. A corrected authoritative artifact or separately reviewed scientific reconstruction is required to reopen their source gate.

## CORE-44 Authorization Gates

CORE-44 AUTHORIZATION: NO. No Final36, final seeds100..104, other Development8 real fit, ShapesAll or StarLightCurves was executed. Frozen LoSTer bytes, old source/config/manuscripts/audits and prior run outputs remain protected. Final pre-report check:41 frozen scientific files match checkout-filtered tag bytes;370 protected files,36 original framework files and all697 prior Wave1/Wave1B artifacts are unchanged. Both old environment fingerprints still equal their locked values (2273ecc22bd24e18b79d2775f6920f8d9af3ba0ab76b96671d13172f5c35cad5 / e31cf122e4cd92edc3b4c4d17fae4f34475dce9e57195f677606f0c0586ead8a). Protected-path Git diff is empty. HEAD remains4b66880; no Wave2A commit/push.

Additional benchmark authorization requires scientific source gates (now including FCACC), common frozen global recipes, full environment/provenance/export/reload and resource/failure policy readiness. One integration score is not eligibility proof for all44.

## Next Recommended Step

Request authoritative FCACC clarification/corrected source for the explicit clustering-gradient and pretraining-mode discrepancies; do not reconnect gradients or toggle training modes silently. Separately authorize a TFMCC source/environment/gradient/resource preflight wave. Preserve these results and failures; do not start broader development/final benchmarking or CORE-44 from this report.
