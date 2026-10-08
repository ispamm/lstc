# LoSTer 2026 Pilot Results

Report date: 2026-10-06. Method-selection pilot; not the final publication benchmark.

## Protocol

Audit 4 is definitive. Eight fixed datasets × three named variants × seeds {0,1,2,3,4} = 120 planned runs. The full OSULeaf / Legacy-Clean / seed 0 preflight was retained, validated and skipped on campaign resume; it was not retrained.

Canonical TRAIN then TEST, all records, per-series z-normalization (ddof=0), known k; labels are evaluation-only during fitting. Independent 3-block encoder/3-block decoder AEs, width256, dropout0.1, sigma1. Two distinct fixed sign/segment/cubic-warp augmentation snapshots and paired preparation/RNG replay. Fifty Adam0.001 epochs per view; KMeans k-means++, n_init1, Lloyd, seed-matched. SGD0.01, no momentum/weight decay; StepLR5/gamma0.1; tau=max(10×0.65^e,0.01); alpha1 and original reductions. B=min(128,N), shuffle/drop_last=True, four loader workers. At most100 joint epochs; strict full-pool original assignment-change fraction<0.001 starting after epoch2. Final stopped checkpoint, never best-label epoch.

Variants isolate S0/C0 (Legacy-Clean), S1/C0 (NoResoftmax with entropy log floor1e-8 and column norm floor1), and S0/C2-once (AlignedInit label-free matching before optimizer). No combined arm or tuning. All settings are in the frozen campaign JSONs.

## Environment and Provenance

| Item | Value |
|---|---|
| Git SHA | a247dffdea50e236b171392b8c9336ecd4cc5727 |
| Annotated tag | pilot-ready-2026 |
| Environment lock SHA256 | d8a5d88b6cb2b5f85a13c0abefeb1f75a4eaeab24ea9a1b5ebfce6acfd239e14 |
| Pilot manifest SHA256 | 47d5d7710035aabe14dd9147f4983c43acfcc25fc000e86be83967539ba4f819 |
| Audit 4 SHA256 | e549bf5a46ca9a4b721d46bfcafa6de2810f30366377620733162371e5fc6c26 |
| Python | 3.9.13, 64-bit |
| PyTorch / CUDA runtime | 1.13.1+cu117 / 11.7 |
| NumPy / SciPy / pandas / sklearn | 1.22.4 / 1.13.0 / 1.5.3 / 1.4.1.post1 |
| GPU / driver | NVIDIA GeForce GTX1080, 8GiB / 552.22 |
| CPU threads | OMP/MKL/OpenBLAS/NumExpr1; Torch intra/inter-op1; loader workers4 |
| Campaign start UTC | 2026-10-05T22:11:35.604955+00:00 |
| Campaign end UTC | 2026-10-06T01:29:12.367241+00:00 |

The controlled 2026 environment preserves audited library semantics; it does not reconstruct the original historical runtime. One GPU child process per run, with fresh process exit before the next fit. All completed artifacts match Git SHA, full config, input hashes, source hash and environment, and pass exact checkpoint prediction reload.

## Dataset Verification

| Dataset | TRAIN | TEST | N | L | k | Verification |
|---|---|---|---|---|---|---|
| SyntheticControl | 300 | 300 | 600 | 60 | 6 | PASS |
| Beef | 30 | 30 | 60 | 470 | 5 | PASS |
| ECG200 | 100 | 100 | 200 | 96 | 2 | PASS |
| OSULeaf | 200 | 242 | 442 | 427 | 6 | PASS |
| ShapesAll | 600 | 600 | 1200 | 512 | 60 | PASS |
| SemgHandMovementCh2 | 450 | 450 | 900 | 1500 | 6 | PASS |
| CinCECGTorso | 40 | 1380 | 1420 | 1639 | 4 | PASS |
| StarLightCurves | 1000 | 8236 | 9236 | 1024 | 3 | PASS |

All 16 canonical headerless TSV files retain their first records; no NaNs, infinities or zero-variance series. Official split/shape/class metadata matched the frozen manifest. Raw SHA256 values are listed in raw-file-hashes.csv and all run provenance.

Source: [official UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and [official summary](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv). The inspected individual official OSULeaf package lacked TSVs. The authorized whole-archive exception was used to preserve original canonical TSV bytes; only the eight pilot pairs were extracted/used. No CORE-44 datasets were extracted or trained. Archive SHA256: a7163f6edd2b6876d195ab0ee5bcce5ec09873cba75512d35bc64a8a493d9d4a.

## Execution Summary

| Measure | Result |
|---|---|
| Complete / planned | 120 / 120 |
| Failed | 0 |
| Blocked | 0 |
| Preflight wall time | 0h 03m 40.2s |
| Total elapsed pilot wall time | 3h 17m 36.8s |
| Prepared dataset/seed pairs | 40 |

Elapsed time includes process startup, validation, cached-arm execution and orchestration gaps. It is distinct from estimated standalone method-fit time. Execution ledger retains PLANNED/RUNNING/COMPLETE/FAILED/BLOCKED states and attempt logs. No score aggregation or winner selection occurred while the campaign was active.

## Results by Dataset

Each entry is five-seed mean ± sample SD (ddof=1), on the complete canonical pooled cohort. Incomplete cells, if any, are explicitly labeled instead of using favorable surviving seeds. ARI is primary; arithmetic NMI secondary; RI and Hungarian ACC descriptive.

NMI_arithmetic

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 0.7322 ± 0.0553 | 0.6865 ± 0.0737 | 0.7324 ± 0.0583 |
| Beef | 0.2928 ± 0.0197 | 0.2805 ± 0.0178 | 0.2928 ± 0.0197 |
| ECG200 | 0.1264 ± 0.0352 | 0.1300 ± 0.0376 | 0.1264 ± 0.0352 |
| OSULeaf | 0.2165 ± 0.0206 | 0.2118 ± 0.0194 | 0.2140 ± 0.0214 |
| ShapesAll | 0.7190 ± 0.0054 | 0.7047 ± 0.0082 | 0.7191 ± 0.0054 |
| SemgHandMovementCh2 | 0.1945 ± 0.0161 | 0.1865 ± 0.0292 | 0.1966 ± 0.0213 |
| CinCECGTorso | 0.2747 ± 0.0182 | 0.2746 ± 0.0153 | 0.2754 ± 0.0200 |
| StarLightCurves | 0.6047 ± 0.0025 | 0.6081 ± 0.0038 | 0.6049 ± 0.0021 |


ARI

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 0.5749 ± 0.0542 | 0.5143 ± 0.0777 | 0.5743 ± 0.0557 |
| Beef | 0.1073 ± 0.0209 | 0.0982 ± 0.0222 | 0.1073 ± 0.0209 |
| ECG200 | 0.2145 ± 0.0487 | 0.2190 ± 0.0516 | 0.2145 ± 0.0487 |
| OSULeaf | 0.1529 ± 0.0165 | 0.1506 ± 0.0152 | 0.1518 ± 0.0176 |
| ShapesAll | 0.3784 ± 0.0166 | 0.3741 ± 0.0212 | 0.3787 ± 0.0167 |
| SemgHandMovementCh2 | 0.1095 ± 0.0130 | 0.1099 ± 0.0255 | 0.1100 ± 0.0179 |
| CinCECGTorso | 0.1706 ± 0.0270 | 0.1760 ± 0.0183 | 0.1706 ± 0.0277 |
| StarLightCurves | 0.5053 ± 0.0040 | 0.5002 ± 0.0055 | 0.5050 ± 0.0033 |


RI

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 0.8702 ± 0.0165 | 0.8573 ± 0.0177 | 0.8701 ± 0.0165 |
| Beef | 0.6797 ± 0.0289 | 0.6831 ± 0.0244 | 0.6797 ± 0.0289 |
| ECG200 | 0.6169 ± 0.0200 | 0.6190 ± 0.0213 | 0.6169 ± 0.0200 |
| OSULeaf | 0.7534 ± 0.0042 | 0.7536 ± 0.0041 | 0.7530 ± 0.0046 |
| ShapesAll | 0.9761 ± 0.0015 | 0.9775 ± 0.0016 | 0.9761 ± 0.0015 |
| SemgHandMovementCh2 | 0.7360 ± 0.0122 | 0.7407 ± 0.0103 | 0.7359 ± 0.0135 |
| CinCECGTorso | 0.6796 ± 0.0176 | 0.6837 ± 0.0121 | 0.6795 ± 0.0179 |
| StarLightCurves | 0.7634 ± 0.0017 | 0.7611 ± 0.0025 | 0.7633 ± 0.0014 |


ACC

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 0.6907 ± 0.0986 | 0.6457 ± 0.0822 | 0.6903 ± 0.0965 |
| Beef | 0.3733 ± 0.0384 | 0.3700 ± 0.0361 | 0.3733 ± 0.0384 |
| ECG200 | 0.7430 ± 0.0217 | 0.7450 ± 0.0229 | 0.7430 ± 0.0217 |
| OSULeaf | 0.3846 ± 0.0298 | 0.3814 ± 0.0286 | 0.3837 ± 0.0316 |
| ShapesAll | 0.5182 ± 0.0126 | 0.5102 ± 0.0145 | 0.5185 ± 0.0130 |
| SemgHandMovementCh2 | 0.3491 ± 0.0138 | 0.3460 ± 0.0265 | 0.3476 ± 0.0175 |
| CinCECGTorso | 0.4459 ± 0.0091 | 0.4500 ± 0.0117 | 0.4452 ± 0.0092 |
| StarLightCurves | 0.7478 ± 0.0039 | 0.7425 ± 0.0060 | 0.7476 ± 0.0032 |


## Results by Variant

Macro averages below are secondary, equal-weight descriptive summaries across eight dataset means, not pooled sample statistics or a selection rule.

| Variant | ARI | NMI_arithmetic | RI | ACC |
|---|---|---|---|---|
| Legacy-Clean | 0.2767 | 0.3951 | 0.7594 | 0.5316 |
| NoResoftmax | 0.2678 | 0.3853 | 0.7595 | 0.5238 |
| AlignedInit | 0.2765 | 0.3952 | 0.7593 | 0.5312 |


## Stability Across Seeds

All seed scores are retained in run-summary.csv. Per-dataset ARI SD and paired SD reductions are in dataset-variant-summary.csv and paired-dataset-effects.csv. No individual samples or 40 seed fits are treated as independent cross-dataset statistical units.

| Dataset | Legacy-Clean ARI SD | NoResoftmax ARI SD | AlignedInit ARI SD |
|---|---|---|---|
| SyntheticControl | 0.0542 | 0.0777 | 0.0557 |
| Beef | 0.0209 | 0.0222 | 0.0209 |
| ECG200 | 0.0487 | 0.0516 | 0.0487 |
| OSULeaf | 0.0165 | 0.0152 | 0.0176 |
| ShapesAll | 0.0166 | 0.0212 | 0.0167 |
| SemgHandMovementCh2 | 0.0130 | 0.0255 | 0.0179 |
| CinCECGTorso | 0.0270 | 0.0183 | 0.0277 |
| StarLightCurves | 0.0040 | 0.0055 | 0.0033 |


## Cluster Utilization / Collapse Diagnostics

Complete collapse means one occupied final center on either view (k>1); near-collapse is maximum fraction≥0.98. Minibatch zeros and ordinary class imbalance are not failure gates. Both views, all centers, hard entropy in nats, assignment changes, dead-center use and component gradients remain in the saved diagnostics.

| Variant | Fits with any complete collapse | Original collapse | Augmented collapse | Fits with any near-collapse flag |
|---|---|---|---|---|
| Legacy-Clean | 0 | 0 | 0 | 0 |
| NoResoftmax | 0 | 0 | 0 | 0 |
| AlignedInit | 0 | 0 | 0 | 0 |


| Dataset | Legacy-Clean occupied O/A | NoResoftmax occupied O/A | AlignedInit occupied O/A |
|---|---|---|---|
| SyntheticControl | 6.0 / 6.0 | 6.0 / 6.0 | 6.0 / 6.0 |
| Beef | 5.0 / 5.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| ECG200 | 2.0 / 2.0 | 2.0 / 2.0 | 2.0 / 2.0 |
| OSULeaf | 6.0 / 6.0 | 6.0 / 6.0 | 6.0 / 6.0 |
| ShapesAll | 60.0 / 60.0 | 60.0 / 60.0 | 60.0 / 60.0 |
| SemgHandMovementCh2 | 6.0 / 6.0 | 6.0 / 6.0 | 6.0 / 6.0 |
| CinCECGTorso | 4.0 / 4.0 | 4.0 / 4.0 | 4.0 / 4.0 |
| StarLightCurves | 3.0 / 3.0 | 3.0 / 3.0 | 3.0 / 3.0 |


Warp validity is monitored without repairing the preserved historical augmentation. Across the 80 distinct snapshots, 80 have at least one non-increasing coordinate step; 80 have at least one series with a negative coordinate difference. These are known augmentation limitations, not silent preprocessing changes. A separately declared validity sensitivity is required before retaining a claim of valid temporal deformation; this pilot isolates only the S1/C2 questions.

## Efficiency

Full-fit values below are stage-summed estimates including both pretrains and KMeans, even when paired arms reuse preparation. They are not fresh standalone observed runtimes or benchmark-quality measurements. Operational joint/inference times and peaks are separately retained. Cached preparation peak memory is reused in full-pipeline estimates; observed joint GPU peaks are also explicit.

Estimated standalone fit seconds, mean ± sample SD:

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 207.4 ± 5.7 | 220.0 ± 6.3 | 206.4 ± 2.8 |
| Beef | 204.1 ± 2.9 | 208.6 ± 7.7 | 204.2 ± 2.8 |
| ECG200 | 205.3 ± 4.4 | 206.9 ± 3.9 | 205.2 ± 4.3 |
| OSULeaf | 209.7 ± 4.4 | 213.6 ± 4.0 | 207.0 ± 2.3 |
| ShapesAll | 222.7 ± 3.9 | 243.2 ± 4.9 | 222.5 ± 4.8 |
| SemgHandMovementCh2 | 227.3 ± 6.9 | 232.2 ± 2.5 | 225.2 ± 5.0 |
| CinCECGTorso | 221.9 ± 4.3 | 229.8 ± 0.8 | 223.2 ± 4.9 |
| StarLightCurves | 341.5 ± 13.7 | 347.8 ± 18.5 | 346.2 ± 6.9 |


Inference seconds, mean ± sample SD:

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 0.0088 ± 0.0017 | 0.0080 ± 0.0004 | 0.0078 ± 0.0002 |
| Beef | 0.0023 ± 0.0003 | 0.0029 ± 0.0009 | 0.0028 ± 0.0010 |
| ECG200 | 0.0059 ± 0.0016 | 0.0035 ± 0.0001 | 0.0035 ± 0.0001 |
| OSULeaf | 0.0086 ± 0.0031 | 0.0083 ± 0.0019 | 0.0080 ± 0.0033 |
| ShapesAll | 0.0189 ± 0.0031 | 0.0181 ± 0.0013 | 0.0178 ± 0.0035 |
| SemgHandMovementCh2 | 0.0134 ± 0.0002 | 0.0131 ± 0.0002 | 0.0140 ± 0.0014 |
| CinCECGTorso | 0.0192 ± 0.0002 | 0.0202 ± 0.0019 | 0.0200 ± 0.0010 |
| StarLightCurves | 0.1215 ± 0.0100 | 0.1102 ± 0.0021 | 0.1161 ± 0.0090 |


Peak allocated GPU MiB, mean ± sample SD (including cached preparation estimate):

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 34.0 ± 0.0 | 33.9 ± 0.0 | 34.0 ± 0.0 |
| Beef | 41.1 ± 0.0 | 41.1 ± 0.0 | 41.1 ± 0.0 |
| ECG200 | 33.5 ± 0.0 | 33.5 ± 0.0 | 33.5 ± 0.0 |
| OSULeaf | 44.6 ± 0.0 | 44.4 ± 0.0 | 44.6 ± 0.0 |
| ShapesAll | 167.5 ± 0.0 | 167.5 ± 0.0 | 167.5 ± 0.0 |
| SemgHandMovementCh2 | 78.4 ± 0.0 | 78.3 ± 0.0 | 78.4 ± 0.0 |
| CinCECGTorso | 81.3 ± 0.0 | 81.2 ± 0.0 | 81.3 ± 0.0 |
| StarLightCurves | 106.6 ± 0.0 | 106.6 ± 0.0 | 106.6 ± 0.0 |


Joint epochs, mean ± sample SD:

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 3.2 ± 1.8 | 9.0 ± 3.9 | 2.6 ± 0.9 |
| Beef | 2.2 ± 0.4 | 4.4 ± 2.6 | 2.2 ± 0.4 |
| ECG200 | 2.0 ± 0.0 | 2.8 ± 1.3 | 2.0 ± 0.0 |
| OSULeaf | 3.6 ± 2.2 | 5.4 ± 1.3 | 2.2 ± 0.4 |
| ShapesAll | 5.2 ± 2.2 | 14.4 ± 2.3 | 5.4 ± 1.9 |
| SemgHandMovementCh2 | 9.4 ± 3.1 | 11.6 ± 0.9 | 8.4 ± 2.5 |
| CinCECGTorso | 3.2 ± 1.8 | 6.4 ± 0.5 | 3.6 ± 2.2 |
| StarLightCurves | 5.2 ± 2.2 | 6.6 ± 3.6 | 6.2 ± 0.4 |

Observed joint-training seconds, mean ± sample SD (preparation excluded):

| Dataset | Legacy-Clean | NoResoftmax | AlignedInit |
|---|---|---|---|
| SyntheticControl | 6.8 ± 3.8 | 19.4 ± 8.4 | 5.8 ± 2.0 |
| Beef | 4.5 ± 0.8 | 9.1 ± 5.2 | 4.7 ± 0.9 |
| ECG200 | 4.4 ± 0.3 | 5.9 ± 2.7 | 4.3 ± 0.0 |
| OSULeaf | 7.7 ± 4.5 | 11.6 ± 2.9 | 4.9 ± 0.9 |
| ShapesAll | 13.0 ± 4.1 | 33.5 ± 5.2 | 12.8 ± 4.5 |
| SemgHandMovementCh2 | 21.1 ± 7.0 | 26.1 ± 1.9 | 19.0 ± 5.4 |
| CinCECGTorso | 7.7 ± 4.2 | 15.6 ± 1.2 | 9.0 ± 5.3 |
| StarLightCurves | 25.6 ± 10.7 | 31.9 ± 17.4 | 30.4 ± 2.2 |

Parameter inventory (identical across the three variants for each dataset):

| Dataset | Training parameters, both views | Retained original AE + centers | Inference encoder + centers |
|---|---|---|---|
| SyntheticControl | 1974512 | 987256 | 494848 |
| Beef | 2815320 | 1407660 | 704512 |
| ECG200 | 2046336 | 1023168 | 512256 |
| OSULeaf | 2727596 | 1363798 | 682752 |
| ShapesAll | 2929664 | 1464832 | 740096 |
| SemgHandMovementCh2 | 4929392 | 2464696 | 1232128 |
| CinCECGTorso | 5213596 | 2606798 | 1302784 |
| StarLightCurves | 3951104 | 1975552 | 987648 |

The frozen resource field `deployed_original_parameters` counts the original autoencoder including its decoder plus centers. It therefore describes the retained original-view artifact. The separately derived inference count above includes only the original encoder and centers, using the frozen architecture; no scientific code or saved resource values were changed.

Registered standalone complete-fit cost and seed variability (secondary summaries):

| Variant | Median full-fit estimate, s | Mean per-dataset ARI SD |
|---|---|---|
| Legacy-Clean | 216.936 | 0.025115 |
| NoResoftmax | 226.243 | 0.029668 |
| AlignedInit | 213.863 | 0.026057 |

## Legacy-Clean vs NoResoftmax

Seed-paired differences are candidate minus Legacy-Clean, with equal-weight dataset means. Positive values favor the candidate for quality. These exploratory eight-dataset effects are descriptive; no final CORE-44 Friedman/Holm analysis or significance-based method reselection is performed.

| Dataset | Paired ARI mean | Paired ARI SD | Paired NMI mean | ARI SD reduction |
|---|---|---|---|---|
| SyntheticControl | -0.0607 | 0.0381 | -0.0457 | -0.0235 |
| Beef | -0.0091 | 0.0097 | -0.0124 | -0.0013 |
| ECG200 | 0.0044 | 0.0099 | 0.0035 | -0.0029 |
| OSULeaf | -0.0023 | 0.0019 | -0.0048 | 0.0013 |
| ShapesAll | -0.0042 | 0.0077 | -0.0142 | -0.0047 |
| SemgHandMovementCh2 | 0.0004 | 0.0188 | -0.0081 | -0.0125 |
| CinCECGTorso | 0.0054 | 0.0121 | -0.0001 | 0.0087 |
| StarLightCurves | -0.0051 | 0.0066 | 0.0033 | -0.0015 |


## Legacy-Clean vs AlignedInit

Seed-paired differences are candidate minus Legacy-Clean, with equal-weight dataset means. Positive values favor the candidate for quality. These exploratory eight-dataset effects are descriptive; no final CORE-44 Friedman/Holm analysis or significance-based method reselection is performed.

| Dataset | Paired ARI mean | Paired ARI SD | Paired NMI mean | ARI SD reduction |
|---|---|---|---|---|
| SyntheticControl | -0.0006 | 0.0029 | 0.0002 | -0.0015 |
| Beef | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ECG200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OSULeaf | -0.0011 | 0.0024 | -0.0026 | -0.0011 |
| ShapesAll | 0.0004 | 0.0010 | 0.0001 | -0.0002 |
| SemgHandMovementCh2 | 0.0005 | 0.0066 | 0.0020 | -0.0048 |
| CinCECGTorso | -0.0001 | 0.0009 | 0.0007 | -0.0007 |
| StarLightCurves | -0.0002 | 0.0030 | 0.0002 | 0.0008 |


## Predeclared Selection Rule

The existing aggregation CLI was run once after terminal completion; its output is retained locally. The frozen summarize() applies eligibility/safety, quality-or-stability, domination and the fixed tie-break sequence. Failed records, if present, are explicit unusable placeholders, never removed to manufacture complete pairs. The numerical recommendation is checked against implementation identity and the named hypothesis; no combined variant is introduced.

| Candidate | Eligible | Safety/consistency | Quality route | Stability route | Dominated |
|---|---|---|---|---|---|
| NoResoftmax | False | False | False | False | True |
| AlignedInit | False | True | False | False | False |


Registered thresholds: no new paired complete-collapse event; minimum dataset ARI difference≥−0.03; macro NMI difference≥−0.01. Quality requires macro ARI≥+0.01, >0 on≥6 tasks and ≥+0.01 on≥3. Stability requires macro ARI≥−0.01 with SD improvement on≥6 tasks averaging≥0.01, or elimination of≥2 collapse events across≥2 tasks. Reject legacy domination. If both qualify, use collapse count, ARI band0.01, SD band0.01, positive-task count, time ratio1.05, then simpler NoResoftmax.

| Candidate | Mean ΔARI | Median ΔARI | Positive tasks | Exact ties | Negative tasks | Mean≥0 | Mean≥.005 | Mean≥.02 |
|---|---|---|---|---|---|---|---|---|
| NoResoftmax | -0.0089 | -0.0033 | 3 | 0 | 5 | False | False | False |
| AlignedInit | -0.0001 | -0.0000 | 2 | 2 | 4 | False | False | False |


The 0/.005/.02 columns describe mean-effect sensitivity only. They do not change the registered +.01 decision or any other gate. Source/config hashes and exclusive paired preparation validate the candidates as the named S1 removal and C2-once initialization experiment, without an undocumented workaround.

## Decision

NoResoftmax: **DO NOT ADOPT**. AlignedInit: **DO NOT ADOPT**. Recommended final method: **Legacy-Clean**.

NoResoftmax fails safety because SyntheticControl paired mean ARI decreases by 0.060685, beyond the allowed 0.03 regression. Its macro ARI change is -0.008895; only 3/8 datasets improve, none by 0.01, and ARI SD improves on only 2/8 datasets with a mean reduction of -0.004553. With no collapse to eliminate, neither substantive route passes. Legacy also dominates it under the registered ARI, SD, collapse and median-cost comparison.

AlignedInit passes safety but provides no qualifying improvement: macro ARI change -0.000150, 2/8 positive dataset effects, no gain of 0.01, and SD improvement on only 1/8 tasks with mean reduction -0.000941. There is no collapse to eliminate. Both Beef and ECG200 are exactly tied to Legacy-Clean across seeds. It passes neither quality nor stability, even though its median estimated fit cost is slightly lower.

All 120 fits satisfy the finite-artifact and named-implementation eligibility checks. The conceptual review retains the exact S1/C0 and S0/C2-once hypotheses; no result is attributed to a hidden workaround. Neither alternative is eligible, so the registered default retains Legacy-Clean. The final simplicity tie-break between eligible alternatives is not reached.

An ADOPT decision identifies the single selected alternative; another eligible alternative can still be DO NOT ADOPT because the fixed candidate-choice rule selected the other. When neither qualifies, Legacy-Clean remains the default. Missing legacy evidence makes the decision INCONCLUSIVE. This is pilot method selection, not publication-level superiority evidence.

## Consequences for CORE-44

The execution and S1/C2 method-selection gate is complete, with Legacy-Clean recommended. The project is not yet ready for an unconditional CORE-44/EXTENDED-44 final freeze: Audit 4 requires an explicit resolution of the observed warp-validity limitation before final freeze. If valid temporal-deformation claims are retained, predeclare the required focused validity sensitivity; otherwise explicitly bound the scientific claims and document the decision. Do not silently repair augmentation or reinterpret this pilot as that sensitivity. Tier-1 baseline source/protocol gates also remain. After resolving these prerequisites, freeze the recommended method before examining final tasks; final seeds remain {100,101,102,103,104}. Analyze non-pilot36 as primary and all44 as development-inclusive. Legacy-Clean remains the reference method; no method-change branch is selected. The final campaign must follow that declared warp-validity/claim decision and complete the separate baseline prerequisites.

No CORE-44 fit was run in this task.

## Failures / Deviations

No protocol deviations.

No execution, NaN/Inf, deterministic-reload or OOM failure occurred. Automatic approval review hit a usage limit during an external report-guard edit; that edit was not executed then. The existing supervisor continued to completion, and the same edit was later approved before aggregation. No training configuration, seed or artifact changed because of this interruption. The resource-count label clarification above affects all 120 retained-artifact counts only; fitting, metrics and selection remain valid and no pilot restart is required.

The whole-archive download is the explicitly authorized acquisition exception. CPU thread limits and external ledger/prediction exports are operational choices; the scientific source/config/tests stayed identical to pilot-ready-2026. Complete collapse, underutilization and non-monotone warp coordinates are reported as outcomes/limitations and never rescued or retried for favorable scores.

## Artifact Locations

Local root: `G:\Articoli\Articoli da Completare\LoSTer 2026`. Raw original files: `data/ucr/<dataset>/*_{TRAIN,TEST}.tsv`; downloads/metadata: `data/downloads`; verification/hashes: `data/verification.json`.

Pilot: `pilot/artifacts/<dataset>/<variant>/seed-<n>/complete.pt` and manifest.json; paired preparation under `pilot/artifacts/initialization`; epoch logs under `pilot/logs`; original metrics under `pilot/results`; predictions/contingencies/status under `pilot/runs`; checkpoint hash index under `pilot/checkpoints`; execution ledger/process logs under `pilot/execution`; aggregation evidence under `pilot/aggregates`. Failed attempts, if any, are preserved locally.

Lightweight inspectable tables: [dataset/variant summary](pilot/dataset-variant-summary.csv), [all runs](pilot/run-summary.csv), [paired seed differences](pilot/paired-seed-differences.csv), [paired dataset effects](pilot/paired-dataset-effects.csv), [warp diagnostics](pilot/warp-diagnostics.csv), [registered selection](pilot/registered-selection.json), [raw hashes](pilot/raw-file-hashes.csv). No large checkpoint, raw dataset or epoch log is copied into Git.

Protected legacy code/manuscript/result files and frozen harness source/config/tests remain unchanged. This report and tables are left uncommitted and unpushed.
