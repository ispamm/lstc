# Time-Warp Validity Resolution

Report date: 2026-10-08. Stages C-J predeclared before harness implementation or W1 training. Data-only and all 40 W1 fits complete; terminal paired analysis applied. CORE-44 is not authorized by this document.

## Executive Decision

The method family is frozen as LoSTer-Legacy-Clean. NoResoftmax and AlignedInit remain DO NOT ADOPT. This task considers only the interpolation validity of the historical warp. M1 (positive smooth speed followed by cumulative integration) is the single selected candidate, named MonotoneWarp. Training validation is required: the already available pilot tables show common series-level violations, not merely one exceptional path in each snapshot. Final outcome: **MONOTONEWARP ADOPTED**. The fixed rule below determines adoption; no alternate method search was opened.

## Historical Implementation Provenance

Source-code provenance: AST comparison of the complete `time_warp` function in `models/LoSTer/data/augmentation.py` against the official repository at commit `656318d18c0945080c969ce2bf839760880b8d30` is exactly equal, ignoring whitespace/comments. This establishes an identical published implementation and is strong evidence of copying/common source; it cannot prove the direction or exact historical copying commit without a contemporaneous attribution record. The 2026 univariate harness implements the same spline, endpoint rescaling, clipping and `np.interp` mapping. Its segment-index adaptation preserves the legacy distribution and RNG draws.

Official source: [Iwana-Uchida augmentation code](https://github.com/uchidalab/time_series_augmentation/blob/656318d18c0945080c969ce2bf839760880b8d30/utils/augmentation.py). Companion [documentation](https://github.com/uchidalab/time_series_augmentation/blob/656318d18c0945080c969ce2bf839760880b8d30/docs/AugmentationMethods.md) calls this smooth time warping and cites Um et al. The [associated survey](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841), section 2.2.3, describes smooth temporal stretching/contraction. Code equivalence and intended concept are separate findings.

The historical mapping is `c(t)=CubicSpline(u, u*m)(t)`, for six equally spaced knots, `m ~ Normal(1,0.2)`, followed by `xp=clip((L-1)*c/c[-1],0,L-1)` and interpolation of the pre-warp samples at integer output times. Neither independent coordinate perturbations, an unconstrained cubic spline, nor clipping guarantees strict order. Clipping adds equal boundary coordinates and cannot repair negative interior increments.

Um et al.'s [actual published example](https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/e5b28e6c884ba5b1fc656f39ba452390cd786b8c/Example_DataAugmentation_TimeseriesData.py), `DistortTimesteps`, cumulatively sums spline-generated local intervals and normalizes endpoints. It is mathematically different from LoSTer's direct perturbed-coordinate spline. Gaussian intervals and their spline have no universal positivity guarantee, but that is a separate potential limitation; this study does not establish the same observed foldback pathology or incidence in Um's algorithm. Do not attribute LoSTer's coordinate defect to Um et al.

## NumPy Interpolation Requirement

The [NumPy 1.22 manual](https://numpy.org/doc/1.22/reference/generated/numpy.interp.html) requires increasing `xp` when `period` is unspecified and gives `all(diff(xp)>0)` as its strict-increase check. The requirement is not enforced by a runtime exception. Finite returned values do not make a violating coordinate sequence a valid temporal interpolation. Both duplicates (`diff=0`) and reversals (`diff<0`) violate the documented precondition. The pilot environment is exactly NumPy 1.22.4; its [version-pinned source docstring](https://github.com/numpy/numpy/blob/v1.22.4/numpy/lib/function_base.py) independently confirms the same precondition and strict-increase check.

## Pilot Snapshot Incidence

Independent verification read every saved array from all 80 snapshots, independently recomputed non-increasing/reversed/duplicate counts, clipping, minimum increments, paired MSE and historical output hashes, and agreed with all 140,580 per-series records. Lightweight tables: [all snapshot severities and speed quantiles](warp/snapshot-severity.csv), [dataset aggregates](warp/severity-by-dataset.csv), [seed aggregates](warp/severity-by-seed.csv), [snapshot-kind aggregates](warp/severity-by-snapshot.csv), [length aggregates](warp/severity-by-length.csv). All denominator/count/fraction fields are explicit.


Reconstruct all 8 datasets x 5 seeds x 2 sequential snapshots using global NumPy RandomState seeding and the unchanged historical augmentation. Capture the exact arrays passed to `np.interp`, its pre-warp input, Gaussian draws and the spline path before clipping. Require exact snapshot SHA256 agreement with both pilot tables and saved run manifests. No labels enter candidate selection or signal diagnostics. No retraining is performed in this stage.

## Series-Level Severity

All 80 snapshots exactly match both frozen hash sources. There are 14,058 original series and 140,580 generated paths. Invalid historical paths: **101,722/140,580 (72.3588%)**; reversed paths: **62,537/140,580 (44.4850%)**. Non-increasing intervals: **23,210,645/140,075,200 (16.5701%)**. Minimum increment: **-6.31952 samples**. Maximum consecutive non-increasing region: **1,298 intervals**. MonotoneWarp: **0/140,580 invalid**, minimum increment **0.353020 samples**. All historical endpoint pairs remain ordered; that does not validate the folded interiors.

| Dataset | Generated paths | Invalid fraction | Reversal fraction | Non-increasing interval fraction | Historical/Monotone output correlation | Paired MSE |
|---|---:|---:|---:|---:|---:|---:|
| SyntheticControl | 6000 | 0.7155 | 0.4435 | 0.1643 | 0.4451 | 0.9573 |
| Beef | 600 | 0.7367 | 0.4417 | 0.1680 | 0.3942 | 1.2151 |
| ECG200 | 2000 | 0.7265 | 0.4260 | 0.1639 | 0.5112 | 0.9838 |
| OSULeaf | 4420 | 0.7305 | 0.4362 | 0.1666 | 0.2619 | 1.5032 |
| ShapesAll | 12000 | 0.7231 | 0.4451 | 0.1665 | 0.3903 | 1.2537 |
| SemgHandMovementCh2 | 9000 | 0.7217 | 0.4176 | 0.1646 | 0.2118 | 1.2586 |
| CinCECGTorso | 14200 | 0.7294 | 0.4331 | 0.1668 | 0.1313 | 1.7705 |
| StarLightCurves | 92360 | 0.7230 | 0.4502 | 0.1655 | 0.5374 | 0.9779 |


Record every row's counts of non-increasing, negative and duplicated intervals; minimum increment; longest non-increasing run in intervals; number of strictly out-of-bound coordinates before clipping; boundary fractions including endpoints; and endpoint ordering. Aggregate by dataset, seed, snapshot and length. A series repeated across seeds/snapshots is a separate generated path, not an independent original subject; denominators distinguish generated paths from unique original series. Endpoint ordering alone does not imply internal monotonicity.

## Signal-Level Consequences

All historical and monotone values are finite. Finite values coexist with invalid interpolation coordinates. Across generated paths, historical mean variance 0.908004, first-difference energy 0.045677, exact adjacent repeat fraction 0.004195, and large-jump fraction 0.005692; MonotoneWarp gives 0.975517, 0.062891, 0.002131, 0.002679 respectively. Historical duplicate-coordinate fraction 0.101961, clipped-coordinate fraction 0.102005, mean lower/upper boundary fractions 0.014358/0.090948; MonotoneWarp has no clipping and exactly its two boundary endpoints. Original first-difference energy 0.115893; pre-warp energy 0.120375. The changes therefore include substantive removal of foldbacks/clipping and different temporal speeds, without asserting biological label preservation.


Compare normalized original, sign/permutation pre-warp input, HistoricalWarp and MonotoneWarp outputs. Per-series statistics: finite fraction, population variance, mean squared first difference, total variation, adjacent exact-repeat fraction, longest exact-flat run, and large discontinuity fraction. A large discontinuity exceeds five times the original row's RMS first difference; this fixed descriptive threshold is neither a label-preservation claim nor an acceptance rule. Undefined correlations for constant signals remain missing. Comparing original to final augmentation includes sign/permutation effects; paired pre-warp comparison isolates time warp.

[Deterministic diagnostic examples](warp/diagnostic-examples.csv) contain original/pre-warp/historical/monotone signal summaries. Representative examples are the fixed indices 0, floor(N/2), N-1, at seed 0/A_train in each dataset. Worst-case diagnostics are selected separately across all ten snapshots of each dataset by minimum historical increment (ties: seed, snapshot name, earliest row). These examples are mathematical worst cases, not representative figures. The earlier seed-0-only example export remains in the external root as preliminary evidence; the report links the complete worst-case selection. They are labeled as worst case. No visual or label selection determines the repair.

## Candidate Monotone Repairs

| Strategy | Correctness and conceptual fidelity | Historical closeness | Complexity/constants | Precedent |
|---|---|---|---|---|
| M1: positive smooth speed, cumulative integration | Positive increments give strict temporal order and smooth local stretch/compression | Preserves draw count, scale parameter, knots and pre-warp input, but changes the coordinate law | Cubic log-speed, exponential, trapezoidal sum; no epsilon, sorting or projection tuning | Cumulative-interval structure in Um's published example; positivity is an explicit correction, not a verbatim reproduction |
| M2: positive anchor increments and PCHIP | Ordered anchors and monotone interpolation preserve order; check finite sampled strict differences | Changes anchor construction and interpolation family | Requires a chosen positive-increment distribution; PCHIP is shape preserving | [SciPy PCHIP](https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.interpolate.PchipInterpolator.html) |
| M3: isotonic/minimal projection of historical path | Ordinary isotonic projection is only nondecreasing, requiring an extra strict-increase constraint | Closest in a chosen norm but can flatten/fix substantial folded regions | Requires norm, endpoint constraints, minimum slope and potentially smoothing | Isotonic regression is a projection tool, not evidence that projected foldbacks are intended temporal deformations |

Choose M1 for mathematical validity and intended local-speed interpretation, before examining any W1 clustering score. M2/M3 will not be trained or tuned in this task.

## Selected MonotoneWarp Definition

For the same paired Gaussian draw `m_j ~ Normal(1,sigma)` at `knot+2=6` uniformly spaced knots on `[0,L-1]`, define `z(t)=CubicSpline(u,m-1)(t)`. Let `v_t=exp(z(t)-max_t z(t))`. The subtracted constant cancels after normalization and only improves numerical representability. Set `c_0=0`, `c_t=sum_{r=1}^t (v_{r-1}+v_r)/2`, and `xp_t=(L-1)*c_t/c_{L-1}`. Set the final endpoint exactly to `L-1` and assert every sampled increment is strictly positive. Return `np.interp(arange(L),xp,prewarp)` at the same length, cast to float32 as in the pilot. Fail explicitly for nonfinite/unrepresentable paths; never reseed, clip, sort or resample. The sampled time map is piecewise linear with smooth varying local increments derived from a cubic log-speed. This is a discrete trapezoidal integration definition, not a claim of exact continuous integration.

`sigma=0.2` now controls log-speed variability rather than perturbed-coordinate multipliers: this semantic difference must be disclosed. As sigma approaches zero the warp approaches identity. Label data are absent from the augmentation API. Sign inversion, segment permutation, snapshot ordering and draw count stay identical. The same NumPy state after augmentation and the same Torch RNG/model construction and loader RNG preserve pairing; augmented model weights and later gradients are expected to differ.

Manuscript counterpart identified before changes: `paper/revision_2026/elsarticle-template-num-revised.tex`, augmentation paragraph describing Gaussian scale factors at six control points and cubic temporal-axis distortion. Audit 4 section on warp validity classifies a monotone repair as Category C. This is a methodological validity correction with a changed augmentation distribution, not an engineering-only fix. The historical implementation violates the interpolation precondition; the manuscript's intended smooth temporal deformation is not guaranteed by its code. No optional model refactoring is authorized.

## Data-Only Comparison

The speed table defines forward deformation speed as `diff(xp)` (new-time samples per original sample). For MonotoneWarp, effective inverse sampling speed is `diff(interp(arange(L),xp,arange(L)))`; it has a valid monotone inverse. The corresponding HistoricalWarp column is explicitly an **operational interpolation-index step**, not a valid inverse of a folded map. Quantiles {0,1,5,50,95,99,100}% are provided for every snapshot. Normalized displacement distributions and original/pre-warp/augmented signal statistics are inspectable in [dataset signal summary](warp-dataset-summary.csv) and the external per-series CSV.


**SUBSTANTIAL AUGMENTATION CHANGE.** Series-weighted mean paired output correlation 0.449381, median correlation 0.477228, mean paired MSE 1.116186. Dataset-macro mean correlation 0.360402 and MSE 1.239995. Mean absolute paired coordinate displacement is 0.0949 of path length. Thus it is not a minimal behavioral repair even though the implementation is compact. Candidate selection remains based on mathematical validity, not these descriptive thresholds or clustering results.


Use identical pre-warp inputs and the same Gaussian arrays on all 80 snapshots. Persist both paths and outputs. Report sampled local speeds, normalized displacement from identity and between paths, output correlation and MSE, first differences and clipping. Classify as MINIMAL if mean normalized paired MSE <=0.01 and median paired correlation >=0.99; MODERATE if MSE <=0.25 and correlation >=0.90; otherwise SUBSTANTIAL. MSE is normalized by original per-row variance (one after normalization). These descriptive bands are declared before aggregation and never govern candidate selection or clustering-score interpretation. Training is required regardless because historical violations are already common.

## Validation Protocol

Exactly W0 = LoSTer-Legacy-Clean + HistoricalWarp versus W1 = LoSTer-Legacy-Clean + MonotoneWarp, eight fixed pilot datasets and seeds 0..4. Reuse all forty frozen W0 runs after matching full pilot configuration, original source hash, environment, input hashes, snapshot hashes and checkpoint/manifest/result provenance. W1 uses an isolated output root `G:/Articoli/Articoli da Completare/LoSTer 2026/warp-validation`; no W0 fit is rerun. Fifty pretraining epochs per view, same architecture, independent centers, losses, batch sizes, stopping, GPU, metrics, diagnostics and exact checkpoint reload protocol. Sequential fresh GPU child process per fit, same single-thread CPU limits and four loader workers. Only augmentation mode/order-name and its implementation differ in scientific config. Record parent Git SHA plus an archived exact dirty-source inventory because this task explicitly leaves new work uncommitted. Do not inspect partial W1 quality scores to change execution. No third arm, score-dependent tuning or automatic favorable rerun.

## Validation Results

**Cost outcome:** recorded mean full-fit estimates increase from 229.98s (W0) to 251.01s (W1), ratio 1.0914 (**+9.14%**), with increases on all eight datasets. This unfavorable operating observation is retained. Runs occurred in separate campaigns, and these are stage-summed operational estimates with substantial Windows worker/process overhead; the comparison does not isolate causal warp-computation overhead or establish benchmark efficiency. Adoption was predeclared on validity, safety, quality comparability and stability, without a cost-winning requirement.

| Dataset | W0 full-fit estimate, s | W1 full-fit estimate, s | Occupied centers O/A, W1 |
|---|---:|---:|---:|
| SyntheticControl | 207.4 | 223.7 | 6 / 6 |
| Beef | 204.1 | 213.3 | 5 / 5 |
| ECG200 | 205.3 | 237.0 | 2 / 2 |
| OSULeaf | 209.7 | 225.7 | 6 / 6 |
| ShapesAll | 222.7 | 245.8 | 60 / 60 |
| SemgHandMovementCh2 | 227.3 | 248.7 | 6 / 6 |
| CinCECGTorso | 221.9 | 241.4 | 4 / 4 |
| StarLightCurves | 341.5 | 372.4 | 3 / 3 |

All W0 and W1 runs stop by the registered assignment-change rule, rather than the epoch cap. Mean ARI SD across datasets changes from 0.025115 to 0.025371. Per-dataset utilization, stopping and stage-summed fit cost are retained in the paired tables; complete convergence/loss/occupancy histories remain external.


Implementation gate: all **58/58** standard-library unit/regression tests pass, with zero skips. Six new tests cover strict monotonicity/endpoints/finite lengths, seed replay and paired RNG, sigma-zero identity and small-sigma limit, label-free signatures, configuration/cache isolation, and explicit rejection of invalid paths. The first test invocation caught UTF-8 BOMs introduced by PowerShell writing; removed from newly authored Python files before the successful full suite. No scientific definition changed to resolve this encoding issue. The historical `augment` function is retained unchanged; only `two_snapshots` selects the explicit mode. Config/checkpoint parsing defaults old records to HistoricalWarp; W1 mode is isolated in its initialization hash.


**40/40 complete, zero failures.** All forty runs passed exact checkpoint reload, bitwise original-view preparation, RNG boundary pairing and exact data-only snapshot agreement. Forty W0 runs were reused; zero W0 retrains.

| Dataset | W0 ARI | W1 ARI | Paired delta ARI | Paired delta NMI | ARI SD increase | Negative seeds | W0/W1 epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| SyntheticControl | 0.5749 | 0.5749 | -0.000010 | +0.000905 | +0.000762 | 2/5 | 3.2 / 2.8 |
| Beef | 0.1073 | 0.1073 | +0.000000 | +0.000000 | +0.000000 | 0/5 | 2.2 / 2.2 |
| ECG200 | 0.2145 | 0.2145 | +0.000000 | +0.000000 | +0.000000 | 0/5 | 2.0 / 2.0 |
| OSULeaf | 0.1529 | 0.1528 | -0.000095 | -0.001186 | +0.000551 | 2/5 | 3.6 / 4.8 |
| ShapesAll | 0.3784 | 0.3784 | +0.000007 | -0.000080 | -0.000311 | 3/5 | 5.2 / 6.0 |
| SemgHandMovementCh2 | 0.1095 | 0.1096 | +0.000104 | +0.001136 | +0.003802 | 2/5 | 9.4 / 6.6 |
| CinCECGTorso | 0.1706 | 0.1701 | -0.000568 | +0.000010 | +0.000749 | 3/5 | 3.2 / 3.0 |
| StarLightCurves | 0.5053 | 0.5073 | +0.002033 | -0.000908 | -0.003507 | 2/5 | 5.2 / 4.8 |

Macro paired delta ARI **+0.000184**; arithmetic NMI **-0.000015**. Positive/negative dataset means: 3/3. Mean ARI SD increase **+0.000256**. W1 final complete collapse 0/40, near-collapse 0/40; any-epoch complete collapse 0/40. Estimated standalone full-fit cost means W0 230.0s, W1 251.0s; median costs 216.9s/236.1s. These are instrumented stage-summed estimates, not benchmark-quality repeated timing. See lightweight [paired runs](warp/paired-runs.csv) and [dataset effects](warp/paired-dataset-effects.csv).

Analyze only after all 40 terminal outcomes. Report per-dataset paired ARI and arithmetic NMI effects, sample seed SD (ddof=1), collapse/utilization, final epoch/stopping, convergence and stage-summed full fit costs. No large inferential significance exercise. Failures remain explicit.

## Predeclared Decision Rule

Default scientific preference is MonotoneWarp because it supplies valid coordinates; no positive ARI improvement is required. ADOPT only if: all paths strictly increase with valid endpoints and finite outputs; all forty W1 fits complete with exact checkpoint reload; no new numerical failure or paired final complete-collapse/near-collapse event; every dataset paired mean ARI >= -0.03; equal-weight macro paired ARI >= -0.01 and macro arithmetic NMI >= -0.01; and mean across-dataset increase in seed ARI SD <= 0.01. These tolerances use Audit 4's pilot-scale 0.03 dataset and 0.01 macro/stability bands without its positive-gain selection requirements. Report occupied centers and epoch trajectories separately; ordinary empty centers are not complete collapse.

Predeclare catastrophic degradation as a dataset mean paired ARI < -0.03 with negative effects on at least 4/5 seeds, or macro ARI < -0.01 with negative dataset means on at least 6/8 tasks and macro NMI < -0.01. Any new numerical/complete-collapse failure is a safety regression independent of ARI. If catastrophic degradation or another comparability/safety gate fails, do not automatically restore invalid HistoricalWarp: classify TIME-WARP COMPONENT REQUIRES RECONSIDERATION, disclose whether the reason is catastrophic, instability, safety or unresolved comparability, and keep final freeze blocked. A separate predeclared removal decision may then be considered; this task authorizes no third training arm. Training cost is descriptive, not an adoption competition.

## Final Warp Decision

**MONOTONEWARP ADOPTED**.

- complete_40: PASS
- valid_all_paths: PASS
- new_collapse: PASS
- new_near_collapse: PASS
- dataset_ARI_band: PASS
- macro_ARI_band: PASS
- macro_NMI_band: PASS
- stability_band: PASS

Catastrophic datasets: none. Systematic catastrophic flag: False. No threshold, seed, arm or hyperparameter was changed after looking at W1 scores.


## Consequences for LoSTer Definition

Final name: **LoSTer**. Historical reference: **LoSTer-Legacy-Clean**. Final 2026 definition: **LoSTer-Legacy-Clean with MonotoneWarp**, using sign inversion, equal-segment permutation, positive cubic log-speed integration with sigma=0.2/knot=4, and the same two fixed snapshots. NoResoftmax and AlignedInit remain DO NOT ADOPT. The rule passed; [frozen method specification](frozen-method-2026.json) defines the final method, exact source and protocol. The warp-method scientific gate is resolved; separate baseline prerequisites and launch authorization remain. Other method-selection decisions remain closed.

## Consequences for CORE-44

**Warp-method scientific gate: PASS. Overall CORE-44 authorization: NOT GRANTED.** The final LoSTer method is frozen for the next planning stage, but the separate baseline source/protocol prerequisites remain and no final campaign is launched.


Do not launch CORE-44. Passing this warp gate resolves this methodological prerequisite only; Audit 4's separate baseline source/protocol gates and explicit campaign authorization still apply. Failure blocks a scientific final freeze.

## Manuscript Disclosure

Outcome A: **MONOTONEWARP ADOPTED**. Suggested precise description: historical LoSTer used the published Iwana-Uchida direct-coordinate spline implementation. The 2026 LoSTer uses a positive cubic log-speed curve, trapezoidal cumulative integration and endpoint normalization to ensure strictly ordered interpolation coordinates. This is an explicit mathematical augmentation change, paired-validated on eight development datasets and five seeds, with near-zero ARI/NMI effects. It is a reproducibility/validity correction, not evidence of superior clustering or universally label-preserving transformations. Um et al. inspired the cumulative-time concept but their code must not be conflated with LoSTer's historical mapping.


Required disclosure for the eventual manuscript revision (no manuscript edit made): historical code used the published Iwana-Uchida time-warp implementation, whose interpolation coordinates can violate monotonicity. The 2026 method uses an explicitly positive cumulative-speed formulation and documents the mathematical augmentation change and paired validation. Do not claim bitwise historical reproduction or universally label-preserving augmentation. No manuscript file is edited in this task.

## Artifact Provenance

Training used the same GTX 1080 and driver 552.22, Python/package environment, one-thread CPU settings and four loader workers as the pilot. Forty fresh GPU child processes executed only W1. Original-view preparation was bitwise identical for all 40 pairs. The surprisingly small clustering effect despite substantial signal changes is an empirical outcome, not a reason to conceal the new augmentation definition. Beef and ECG200 have exact seed-wise ARI/NMI ties; other dataset mean ARI changes lie between -0.000568 and +0.002033.


Pilot completion preserved in commit `596a163`, annotated `pilot-complete-2026`, pushed only to origin/revision-2026 and origin tag. Original training SHA is `a247dffdea50e236b171392b8c9336ecd4cc5727`. Primary-source pinned code is archived separately under the new external root. Historical function AST equality is verified independently of numeric diagnostics. Original pilot files are read-only throughout the new study. New per-series paths, raw signal statistics, source snapshots and training outputs stay outside Git; lightweight tables and this report remain uncommitted and unpushed. Verify protected-path diff and original artifact hashes at completion.

Terminal preservation: **1374 original pilot files unchanged in SHA256, size and mtime**, with identical file membership. New W1 source hash `fcd966198dbf1e85bb5c6f2038fa19172eec0e1ea8a64ceb88340dffa9a7df16`; exact dirty sources archived with parent commit `596a16391c4cc03b8bb1c71f1cd8aa473773a28a`. Training source did not change during the campaign.

Final checks: **58/58 unit/regression tests**, **40/40 fresh W1 fits**, **40 unchanged W0 references reused**, **80/80 saved-array validations**. No final or any-epoch collapse, no near-collapse, and full final-center utilization on both views for every run. All 80 W0/W1 runs stop by the registered assignment-change condition. No third arm, CORE-44 fit, manuscript edit, Report 07 commit or Report 07 push occurred.

Primary-source byte hashes and terminal analysis script hashes: [artifact provenance](warp/artifact-provenance.json). Exact replay precedes harness modification in the numbered protocol: to reproduce historical data preparation use the `pilot-ready-2026` source/config revision, not the later changed package hash, and a fresh output root. The archived W1 source closure and manifests fully identify the uncommitted MonotoneWarp extension.
