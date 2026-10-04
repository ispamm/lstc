# LoSTer Provenance and Artifact-Recovery Audit

Audit date: 2026-10-05. Scope: read-only provenance investigation after preserving Audit 1. Audited branch: `revision-2026`, HEAD `f25f9194568e074827fd8063947b3bbf747a7aab`. Historical upstream code: `b594b650f68ebed80436b8de7313fa1536d3d95c`.

## Executive Summary

- **Published UCR numbers:** a real parser defect is confirmed and its standard-input sample-count effect is quantified: one TRAIN sample and one TEST sample are omitted for each of the 17 datasets. Eight actual official TSVs were checked. This establishes a mismatch with the advertised full dataset when standard files are used; it does **not** establish that the reported scalar scores were calculated incorrectly, nor prove which input files generated the historical tables. Historical data checksums, predictions and run configurations are absent. F11 classification: **CONFIRMED PARSER BUG, IMPACT QUANTIFIED**; score/ranking impact remains unknown.
- **Published M5 numbers:** no demonstrated numerical invalidity. The filter includes string `store_id` and omits `d_1913`; the stored notebook proves a 1,150-series, 29-class historical selection. Corrected membership and score impact cannot be determined without the missing daily sales data. The corrected set can only remove rows, under the stored schema.
- **NMI convention:** all 15 supplied model implementations route clustering evaluation to one unchanged sklearn function, with no normalization override. The scikit-learn 1.4.1.post1 pin implies **arithmetic NMI**, making option **A** the most plausible table convention. The printed equation is geometric. Exact convention for the February 2025 published means remains unverified because the July code/environment upload postdates those means.
- **Five-run variance and ARI:** not recoverable from the files available here. Thirteen named convergence runs provide RI/NMI trajectories for two datasets, without seed/config mappings or ARI. They cannot reconstruct the five values behind any table mean. External W&B histories or original per-seed metric CSVs could resolve this without training.
- **Exact trained LoSTer models:** no checkpoints or active trained centroid states were found locally or in any reachable historical tree. The save routine also omits the separately optimized loss-module centroids. A newly trained model cannot be claimed to reconstruct the exact historical state.
- **Store Sales:** not reproducible from this repository. Its reported shape, scores and timing appear in the original February 2025 PDF, but no extraction/filtering/label pipeline, raw arrays, launch recipe, predictions or run artifacts appear in the complete reachable history.
- **Reruns:** new training is necessary to produce complete new states/predictions or five-run statistics if original artifacts cannot be supplied, and to evaluate corrected UCR inputs if the historical runs used the defective parser. M5 clustering reruns are conditional on a membership change and the chosen revised protocol. New measurements are needed if timing claims must be independently substantiated and original benchmark records cannot be recovered. No training is needed to preserve tables, recover existing curves/M5 selection indices, correct plot epoch labels, document arithmetic NMI once provenance is confirmed, or recompute metrics from subsequently recovered predictions/contingency tables.

**Stage A completed:** Audit 1 was the sole untracked file, committed as `f25f9194568e074827fd8063947b3bbf747a7aab` with subject `Add initial LoSTer paper-code audit`, and pushed only to `origin revision-2026`. The tree was clean afterward. Audit 2 is deliberately uncommitted and unpushed.

### Scope, evidence conventions and limits

`M:n` means line n of `paper/original_r1/elsarticle-template-num-revised.tex`; its revision copy is identical. Notebook cell numbers are zero-based positions in saved JSON, rather than execution counts. Source paths/line references refer to the audited HEAD unless a commit is stated. HIGH confidence concerns an observed artifact or source behavior; it never substitutes for missing run provenance.

Inspection covered all **34 reachable commits**, all recursive Git trees (**296 distinct historical paths**, **243 distinct blobs**, 295 paths at HEAD), commit statistics, deleted paths, source text, notebook source/outputs, public branch/tag advertisements, and every non-Git local project file, including ignored files and immutable Drafts. The repository is not shallow. All textual historical blobs were searched for artifact serialization, metrics, Store Sales, timing and benchmark evidence; notebooks were inspected separately for embedded outputs. Symlinks were resolved from Git mode 120000 and target text. There are 77 tracked symbolic links; this Windows checkout materializes them as ordinary target-text files, so filesystem searches alone can miss shared implementations.

The searches covered results/seed/prediction/assignment/checkpoint/ckpt/centroid/store_sales/storesales/Kaggle/timing/benchmark/speed/inference/W&B/CSV/NPY/PKL/PT/PTH names and relevant content. Broad hits such as generic transformer time features, M5 `store_id`, and CSV digits were inspected rather than treated as Store Sales evidence. No scans outside this project subtree were made. Git configuration, setup manifests and hashes under `.git/` are administrative evidence, not trained scientific artifacts. Unreachable objects/deleted remote branches, other computers, remote W&B accounts and external author archives are outside this evidence set.

Only Python standard-library inspection, Git, PDF text/metadata tools and public documentation/data reads were used. No scientific module was executed, no package was installed, and no training, clustering, metric recomputation, benchmark or manuscript compilation was performed. Small data checks ran entirely in memory, so no temporary files required cleanup.

## Git History Timeline

All dates below are author dates, also matching committer dates; times are Europe/Rome UTC+02. This is the complete reachable chronology, including administrative commits.

| Commit | Date/time | Introduction or change established by tree/diff |
|---|---|---|
| `50f203a` | 2025-07-18 10:23 | Dataset ignore placeholder and requirements; sklearn 1.4.1.post1 and pandas 1.5.3 already pinned |
| `92a2df7` | 2025-07-18 12:26 | LoSTer, five seeds 0–4, UCR loader, evaluation, checkpoint/output paths, result aggregation |
| `6669c95` | 2025-07-18 15:30 | CKM implementation and launch/aggregation scripts |
| `a1fcaa6` | 2025-07-18 15:55 | Remove augmented branch from CKM |
| `d2942e1` | 2025-07-18 16:07 | CKM learning-rate change to 0.001 |
| `aa28567` | 2025-07-18 16:18 | DEC implementation |
| `fb90e1c` | 2025-07-18 16:49 | DNFCS implementation plus baseline edits; subject is misleadingly “CKM” |
| `5a3616b` | 2025-07-18 17:41 | DTC implementation |
| `3a2cc72` | 2025-07-18 18:27 | IDEC implementation |
| `5324803` | 2025-07-18 19:03 | DTCR and related baseline edits |
| `7cac262` | 2025-07-18 19:31 | DTCC implementation |
| `ab21b18` | 2025-07-18 20:33 | LoSTer_DRNN ablation |
| `226bd24` | 2025-07-18 20:57 | LoSTer_MLP; DRNN experiment becomes a link to LoSTer experiment |
| `52cdc06` | 2025-07-18 21:55 | LoSTer_F and related argument-default changes |
| `6c0accf` | 2025-07-18 22:48 | LoSTer_KL |
| `bb4259a` | 2025-07-18 23:30 | PathFormer comparison implementation |
| `523d65e` | 2025-07-18 23:54 | Preformer comparison, supplied batch size 32 |
| `2037845` | 2025-07-19 00:15 | iTransformer comparison |
| `ee0a197` | 2025-07-21 16:14 | M5 preprocessing script, executed notebook, instructions |
| `f46d389` | 2025-07-21 16:42 | M5 loader/scripts for standard baselines |
| `80bb70c` | 2025-07-21 17:23 | M5 loader/scripts for LoSTer family and transformer wrappers |
| `f1d852a` | 2025-07-21 18:06:09 | README with UCR/M5 setup and launch/aggregation instructions |
| `99df4aa` | 2025-07-21 18:06:42 | Delete `M5_instructions.txt`; recipe retained in README |
| `2f2dd27` | 2025-07-22 00:51 | Add wandb 0.21.0 requirement |
| `d8b3746` | 2025-07-22 07:43 | LoSTer W&B logging, including seed/config and per-epoch ARI |
| `96b76a0` | 2025-07-22 10:55 | W&B logging across other model implementations |
| `6ad7595` | 2025-07-22 11:17 | Augmentation permutation-split data-type fix |
| `3096275` | 2025-07-24 13:23 | Set pretraining optimizer LR to 0.001 in LoSTer family/transformers |
| `a322caa` | 2025-07-24 13:36 | LoSTer joint LR/script/default becomes 0.01; related ablation changes |
| `efb74ff` | 2025-07-24 21:47 | Four convergence CSVs, two PDFs and plotting notebook |
| `9f52cb3` | 2025-07-24 23:32 | Add Preformer to UWave exports/plot; refresh both PDFs/notebook |
| `b594b65` | 2025-08-01 10:50 | Remove two decoder argument declarations in three configs; no network/forward change |
| `0c83687` | 2026-10-04 23:42 | Revision workspace, manuscript snapshots and safeguards |
| `f25f919` | 2026-10-05 00:26 | Preserve Audit 1; no scientific implementation changes |

Five-seed launching and final-per-seed metric serialization were present at LoSTer's first commit; there is no later commit introducing the claimed five-run protocol. W&B logging is a July 22 addition. No reachable code commit introduces Store Sales preprocessing or speed measurements. Store Sales tables/timing enter this Git history only through the manuscript import at `0c83687`, which is an archival import, not an experiment.

`git ls-remote` for fork and upstream matched cached heads. Upstream advertises only `main` at `b594b65`, with no tags; the fork advertises main, revision-2026 and `original-code`. Both public [upstream releases API](https://api.github.com/repos/jrtaloma/lstc/releases) and [fork releases API](https://api.github.com/repos/ispamm/lstc/releases) returned empty lists at inspection. The annotated `original-code` tag is a 2026 fork archival tag, not evidence of a 2025 upstream release. Its tag object is `3a44012eaeafc8a18d80d03b7c9a661ce8c6f048` and target is `b594b65`.

### Association with submission versions

| Submission/version | Direct dated/content evidence | Commit association and confidence |
|---|---|---|
| PR-D-25-00861 original | Local original PDF creation/modification metadata: 2025-02-12. Contains transformer comparisons, M5/Store Sales results, and the same 2.85/1531/550/3.66-second timing table. It lacks the later empirical convergence figures/section. Several original UCR means, including LoSTer 0.9737 RI and 0.7399 NMI for fetal ECG 1, match the imported manuscript. | **No generating commit established.** All reachable commits start July 18, after this artifact. `92a2df7` is the earliest public source snapshot of the method, not the February execution version. HIGH confidence in chronology/content; UNKNOWN generating commit. |
| PR-D-25-02649 resubmission | No separately dated manuscript or code snapshot for this exact un-suffixed version is present. | UNKNOWN. Assigning July uploads to this submission would be speculation. |
| PR-D-25-02649R1 | Local correspondence is dated 2025-09-09. Imported revised source contains July convergence figures. The ZIP has uniform entry timestamps 2026-05-29, and correspondence-PDF export metadata is also May 29, 2026; these are export dates. | `9f52cb3` is established for the final convergence artifacts; `b594b65` is the most complete available pre-decision code snapshot, a **LOW-confidence candidate** for R1 code association. Neither proves the generating version of table runs. |

The preserved Figure 4 PDF was created August 2, 2025 using Chromium/Skia, yet equivalent timing numbers already appear in the February original PDF. It is therefore a later rendering of older reported results, not evidence of August benchmarking. Do not equate PDF creation dates, Git upload dates or ZIP export dates with run dates.

## Recoverable Historical Artifacts

No local or reachable historical `.pth`, `.pt`, `.ckpt`, prediction serialization, dataset `.npy/.npz`, per-seed final-metric CSV or confusion matrix was found. Checkpoint/log/dataset directories contain ignore placeholders. Ignore rules explain why runs could have existed outside version control; they do not prove the artifacts still exist.

### Inventory of useful local artifacts

Sizes are bytes. All paths are relative to the project root. “Reusable” means an existing record can be inspected/replotted/documented; it does not certify the underlying run.

| Path | Type/size | Git status / provenance | Dataset, method, seed evidence | Scientific reuse |
|---|---|---|---|---|
| `csv_logs/NonInvasiveFetalECGThorax1_NMI.csv` | W&B-style CSV; 33,340 | Tracked; `efb74ff`, unchanged | Six named model trajectories; seed unknown | Recover NMI curves |
| `csv_logs/NonInvasiveFetalECGThorax1_RI.csv` | W&B-style CSV; 33,295 | Tracked; `efb74ff`, unchanged | Same six runs; seed unknown | Recover RI curves |
| `csv_logs/UWaveGestureLibraryY_NMI.csv` | W&B-style CSV; 42,570 | Tracked; `efb74ff`, amended `9f52cb3` | Seven named trajectories; seed unknown | Recover NMI curves |
| `csv_logs/UWaveGestureLibraryY_RI.csv` | W&B-style CSV; 41,697 | Tracked; `efb74ff`, amended `9f52cb3` | Same seven runs; seed unknown | Recover RI curves |
| `M5_EDA_cat_store_id.ipynb` | Notebook; 136,261 | Tracked; single version `ee0a197` | M5 selection/preprocessing, no model/seed | Full 1,150 positional indices, 29 label counts, shapes and recipe |
| `plots_training.ipynb` | Notebook; 1,904,175 | Tracked; `efb74ff`, `9f52cb3` | Two datasets; 13 named run mappings | Recover plotting recipe and embedded plots/frames |
| `M5_EDA.py` | Python preprocessing source; 2,647 | Tracked; `ee0a197` | M5 raw price/calendar construction | Recipe; no arrays or model results |
| `figures/plots_NonInvasiveFetalECGThorax1.pdf` | PDF; 23,017 | Tracked; `efb74ff`, refreshed `9f52cb3` | Six curves; no seed metadata | Archival/replot comparison |
| `figures/plots_UWaveGestureLibraryY.pdf` | PDF; 24,828 | Tracked; `efb74ff`, refreshed `9f52cb3` | Seven curves; no seed metadata | Archival/replot comparison |
| `paper/original_r1/Figure_1.pdf`; corresponding revision copy | PDF; 324,603 each | Tracked archival import `0c83687` | Model diagram, no run | Diagram only |
| `paper/original_r1/Figure_2.pdf`; corresponding revision copy | PDF; 23,017 each | Tracked `0c83687`; exact bytes of current fetal-ECG plot | As above | Figure/CSV lineage confirmed |
| `paper/original_r1/Figure_3.pdf`; corresponding revision copy | PDF; 24,828 each | Tracked `0c83687`; exact bytes of current UWave plot | As above | Figure/CSV lineage confirmed |
| `paper/original_r1/Figure_4.pdf`; corresponding revision copy | PDF; 12,233 each | Tracked `0c83687`, original ZIP entry | Store Sales seven timing bars | Published values only; no measurement samples |
| `paper/original_r1/elsarticle-template-num-revised.tex`; revision copy | TeX; 76,921 each | Tracked `0c83687`, original ZIP entry | All reported means and timing table | Preserve exact reported numbers |
| `Drafts/PR-D-25-00861.pdf` | Original submission PDF; 2,538,574 | Ignored/untracked immutable local artifact | February 2025 methods/tables/timings | Dated content comparison; no per-run data |
| `Drafts/Your Submission PR-D-25-02649R1.pdf` | Correspondence PDF; 120,999 | Ignored/untracked immutable local artifact | September 2025 correspondence; exported May 2026 | Submission-stage chronology only |
| `Drafts/Pattern_Recognition___Concrete_Dense_Network_for_Long_Sequence_Time_Series_Clustering.zip` | Manuscript ZIP; 419,576 | Ignored/untracked immutable local artifact | Twelve exact source/figure entries | Recover source snapshot; not an experiment archive |

Notebook metadata records kernel `ts_clustering` and Python 3.9.23, but no installed sklearn/pandas runtime or seed provenance. The convergence PDFs record Matplotlib 3.8.4 and creation times July 24, 2025 at 23:25:27/28 Europe/Rome, matching their later commit and the requirements pin.

### Launch and aggregation artifacts

All following tracked launch scripts encode seeds **0 1 2 3 4**. Each model also supplies `models/<directory>/results.py`, which reads per-seed scalar CSVs and prints mean/std/ARI; no such input CSVs are present. Source availability is reproducible procedure evidence, not recovered execution evidence.

| Launch path | Bytes | First commit | Current batch size | Reusable scope |
|---|---:|---|---:|---|
| `models/LoSTer/run.sh` | 1,183 | `92a2df7` | 128 | UCR 17 + M5; configs/commands only |
| `models/CKM/run.sh` | 954 | `6669c95` | 128 | UCR 17 + M5 |
| `models/DEC/run.sh` | 889 | `aa28567` | 128 | Additional baseline, not in main paper tables |
| `models/DNFCS/run.sh` | 975 | `fb90e1c` | 128 | UCR 17 + M5 |
| `models/DTC/run.sh` | 904 | `5a3616b` | 128 | UCR 17 + M5 |
| `models/IDEC/run.sh` | 924 | `3a2cc72` | 128 | UCR 17 + M5 |
| `models/DTCR/run.sh` | 824 | `5324803` | 128 | UCR 17 + M5 |
| `models/DTCC/run.sh` | 866 | `7cac262` | 128 | UCR 17 + M5 |
| `models/LoSTer_DRNN/run.sh` | 1,002 | `ab21b18` | 128 | Backbone ablation |
| `models/LoSTer_MLP/run.sh` | 1,020 | `226bd24` | 128 | Backbone ablation |
| `models/LoSTer_F/run.sh` | 1,116 | `52cdc06` | 128 | Spectral/F variant; plotted as LoSTer_F |
| `models/LoSTer_KL/run.sh` | 1,156 | `6c0accf` | 128 | KL ablation |
| `models/PathFormer/run.sh` | 1,119 | `bb4259a` | 128 | Transformer wrapper; length truncation in loader |
| `models/Preformer/run.sh` | 1,146 | `523d65e` | 32 | Transformer wrapper |
| `models/iTransformer/run.sh` | 1,080 | `2037845` | 128 | Transformer wrapper |

No stored shell-session history, seed-resolved invocation log or executed benchmark command was recovered. README contains generic setup/training commands; its destructive shell examples were read as historical text and were not executed.

### Files recoverable only from history

The sole historical path absent from HEAD is `M5_instructions.txt`, added `ee0a197` and deleted `99df4aa`. It points to the M5 Kaggle competition and instructs running the root script then notebook; README contains the same recipe. Recovery is possible with `git show f1d852a:M5_instructions.txt`, with no training. It contains no Store Sales pipeline, model states or run results.

The four CSV paths and two figure paths are the only committed result-like files in upstream history. The older `efb74ff` UWave CSVs have 37 columns before Preformer's six columns were added; all old column values match the current exports exactly. Older PDFs (23,064 fetal-ECG bytes; 23,041 UWave bytes) and notebook revision also remain in Git. No vanished checkpoint/array/seed-results path was found. Embedded notebook output is included in this negative search; displayed M5 head/tail samples and curve frames are not full raw arrays or cluster assignments.

## UCR Header Investigation

**F11: CONFIRMED PARSER BUG, IMPACT QUANTIFIED.** This label quantifies the standard-input row loss; it does not claim to quantify metric changes.

The repository expects official `datasets/UCRArchive_2018/<name>/<name>_TRAIN.tsv` and TEST files. README directs extraction of the official archive without a header-adding conversion. `models/LoSTer/data/load_data.py:21–23` calls `pd.read_csv(..., sep='\\t')` for each split and concatenates their `.values`. The same behavior exists in CKM's independent loader and PathFormer's loader; other directories share these data paths via Git links. The UCR read calls were introduced `92a2df7` and retained through the M5-only expansion `80bb70c`; no historical header repair was found.

The [official UCR briefing, pp. 3–4](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/BriefingDocument2018.pdf) defines one labeled numeric exemplar per row. The [pandas 1.5.3 API](https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.read_csv.html) specifies that, without names, default header inference consumes the first row as column names. Thus a standard numeric first row is lost independently from each split. Concatenating `.values` avoids an additional alignment issue from differing numeric column names, but cannot restore the omitted observations. Training and full-pool evaluation both use the reduced pool; this is not just a missing displayed dataset count.

### Official-data check and all 17 counts

The official [DataSummary.csv](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv) (11,624 bytes, SHA-256 `7fe2428588c5d513b88df2c480d71e25e4f113e8ad695e21ee2057dfac53c3ac`) agrees with the manuscript's train/test counts, lengths and class counts for all 17 datasets.

A bounded HTTP-range reader inspected the central directory and **only eight small TSV members** of the [official 2018 ZIP](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip), using the publicly documented archive password. Transfer for that reader: **815,512 bytes**, under a hard 2.5 MB cap; the whole 316,175,400-byte archive was not downloaded. An earlier 65,536-byte range probe verified server support. A 552,537-byte official SyntheticControl toolkit ZIP was inspected separately (then fetched again for label counts); it contains numeric whitespace TXT and metadata-bearing TS/ARFF representations, distinct from the exact UCR TSVs. No files were saved.

| Dataset | Official TRAIN/TEST | Loader TRAIN/TEST on standard TSV | Pooled official → loader | Official k | Direct exact TSV check / k after omission |
|---|---|---|---|---:|---|
| CinCECGTorso | 40 / 1380 | 39 / 1379 | 1420 → 1418 | 4 | Metadata/format deduction; k not directly checked |
| NonInvasiveFetalECGThorax1 | 1800 / 1965 | 1799 / 1964 | 3765 → 3763 | 42 | Metadata/format deduction; k not directly checked |
| NonInvasiveFetalECGThorax2 | 1800 / 1965 | 1799 / 1964 | 3765 → 3763 | 42 | Metadata/format deduction; k not directly checked |
| StarLightCurves | 1000 / 8236 | 999 / 8235 | 9236 → 9234 | 3 | Metadata/format deduction; k not directly checked |
| UWaveGestureLibraryX | 896 / 3582 | 895 / 3581 | 4478 → 4476 | 8 | Metadata/format deduction; k not directly checked |
| UWaveGestureLibraryY | 896 / 3582 | 895 / 3581 | 4478 → 4476 | 8 | Metadata/format deduction; k not directly checked |
| MixedShapesRegularTrain | 500 / 2425 | 499 / 2424 | 2925 → 2923 | 5 | Metadata/format deduction; k not directly checked |
| MixedShapesSmallTrain | 100 / 2425 | 99 / 2424 | 2525 → 2523 | 5 | Metadata/format deduction; k not directly checked |
| EOGVerticalSignal | 362 / 362 | 361 / 361 | 724 → 722 | 12 | Metadata/format deduction; k not directly checked |
| SemgHandMovementCh2 | 450 / 450 | 449 / 449 | 900 → 898 | 6 | Metadata/format deduction; k not directly checked |
| ECG5000 | 500 / 4500 | 499 / 4499 | 5000 → 4998 | 5 | Metadata/format deduction; k not directly checked |
| OSULeaf | 200 / 242 | 199 / 241 | 442 → 440 | 6 | Metadata/format deduction; k not directly checked |
| Symbols | 25 / 995 | 24 / 994 | 1020 → 1018 | 6 | Metadata/format deduction; k not directly checked |
| MiddlePhalanxOutlineAgeGroup | 400 / 154 | 399 / 153 | 554 → 552 | 3 | Both exact TSVs checked; k stays 3 |
| ProximalPhalanxOutlineAgeGroup | 400 / 205 | 399 / 204 | 605 → 603 | 3 | Both exact TSVs checked; k stays 3 |
| ProximalPhalanxTW | 400 / 205 | 399 / 204 | 605 → 603 | 6 | Both exact TSVs checked; k stays 6 |
| SyntheticControl | 300 / 300 | 299 / 299 | 600 → 598 | 6 | Both exact TSVs checked; k stays 6 |

For the 13 unchecked raw datasets, the loader counts are exact consequences of the official schema/counts, not empirical executions on the historical input files. No model or pandas loader was run.

### Exact TSV evidence and reproducible identification

Each inspected first row was fully numeric, with label in field 1 and L observations after it. Representative first fields and SHA-256 identities:

| Dataset/split | Raw rows / fields | First four fields | SHA-256 |
|---|---|---|---|
| SyntheticControl TRAIN | 300 / 61 | 1, -0.37693558, 1.2248643, 0.34387438 | `58af636cbd5592bdda45ba638629e40f17c08354aa545a61fd41b92582bdd176` |
| SyntheticControl TEST | 300 / 61 | 1, -1.4139728, -1.1620647, -0.62417387 | `cd44464b200820913f928a952af78d512ee3924f6243053d28a6d45c0aacdb76` |
| MiddlePhalanxOutlineAgeGroup TRAIN | 400 / 81 | 2, -0.53790081, -0.47113745, -0.3241874 | `1cec6f15c6f114198967d847cf0566611bbe72b7121f11537b078f82c3c207c2` |
| MiddlePhalanxOutlineAgeGroup TEST | 154 / 81 | 2, -0.78752559, -0.72126207, -0.52368794 | `86ca182529fe0eb4af864300b8ea1b992eba7e46181313ada9ae67d532a25450` |
| ProximalPhalanxOutlineAgeGroup TRAIN | 400 / 81 | 2, -0.70673216, -0.65213669, -0.45675415 | `d78ca88846204d29e349f30b1319d7b8388e4e01e126731f0f047561c0127d12` |
| ProximalPhalanxOutlineAgeGroup TEST | 205 / 81 | 3, -0.54020196, -0.45485724, -0.29946496 | `45031c90f22581591156713c466e14fdd407b0025cdc107031f7c7ac4b42eabf` |
| ProximalPhalanxTW TRAIN | 400 / 81 | 3, -0.70129052, -0.608801, -0.43552144 | `96e310bd1a8eb8b31e8765ab576cf9b2f7909ca39eb69ba2d29f7bf6ccc5cb73` |
| ProximalPhalanxTW TEST | 205 / 81 | 8, -0.65756949, -0.57131126, -0.34851172 | `6ad75ed55cabb6b04c21c1a3ac84f4663a7623ca4a69cbde74849dc31abfc4fe` |

### Could k change?

The loader derives k from unique **retained pooled labels** (`load_data.py:33`), rather than fixing official k. A class vanishes only if every member of that class is among the two removed rows. It cannot increase k. No class vanishes in the four directly checked datasets: SyntheticControl has 100 members per pooled class, MiddlePhalanx minimum pooled count 92, ProximalPhalanxOutlineAgeGroup minimum 89, and ProximalPhalanxTW minimum 18.

Official N/k metadata alone cannot exclude singleton/two-member classes in the other 13 datasets. Their first-row labels plus pooled class-frequency vectors would suffice, without acquiring all time-series values; these vectors are not in DataSummary.csv. Exact historical run files or checksum-matched labels are still needed to establish historical k. Do not infer unchanged k merely because N is large.

### Impact and repair decision

The code/README protocol is affected on canonical data. Historical runs may have used the same code on standard files, an older uncommitted implementation, or locally header-added files; evidence does not distinguish them. The February original means predate the public implementation snapshot, and W&B exports contain no data hashes or sample counts.

Removing two samples can change pair counts, initialization, training, early stopping and rankings; small fractional row loss does not bound model sensitivity. Corrected full-pool training cannot generally be obtained by recomputing two scalar entries. Recover data/run provenance first. If the historical parser was defective, evaluate the corrected cohort for all affected methods under a separately approved rerun plan. No parser or result was changed in this audit.


## Historical NMI Convention

**F23 verdict: implementation convention established; historical table convention most plausibly A (arithmetic), not conclusively traced.**

The common evaluator is `models/LoSTer/utils/metrics.py:18–22`, introduced once at `92a2df7` and never changed. It calls `sklearn.metrics.normalized_mutual_info_score(label, prediction)` with no `average_method`. All other model metric paths are Git symlinks to that exact file. Each independent experiment imports `evaluation` from `utils.metrics` and evaluates its own assignments against labels; DRNN shares LoSTer's experiment through a link. These are common evaluation semantics within separate training scripts, not a saved universal evaluation job that can be matched to table runs.

The only requirements change after `50f203a` adds W&B. Scikit-learn is 1.4.1.post1 throughout; pandas is 1.5.3, NumPy 1.22.4, Torch 1.13.1+cu117, Matplotlib 3.8.4 and statsmodels 0.14.1. The [official sklearn 1.4 NMI API](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.normalized_mutual_info_score.html) documents arithmetic default since 0.22. This yields `2 MI/(H_true + H_pred)`; M:508–511 prints `MI/sqrt(H_true H_pred)`. The two generally differ. No explicit geometric/min/max normalization call exists in historical model code.

| Model / paper use | Supplied evaluation path | Explicit average_method | Pinned / actual historical version | Code convention / table inference |
|---|---|---|---|---|
| LoSTer; main, transformer, ablation and retail tables | Own evaluator; experiment:74 | None | 1.4.1.post1 / historical runtime unknown | Arithmetic / A most plausible |
| CKM; main and retail | Linked common evaluator; experiment:58 | None | Same | Arithmetic / A most plausible |
| IDEC; main and retail | Linked common evaluator; experiment:56 | None | Same | Arithmetic / A most plausible |
| DNFCS; main and retail | Linked common evaluator; experiment:61 | None | Same | Arithmetic / A most plausible |
| DTC; main and retail | Linked common evaluator; experiment:73 | None | Same | Arithmetic / A most plausible |
| DTCR; main and retail | Linked common evaluator; experiment:92 | None | Same | Arithmetic / A most plausible |
| DTCC; main and retail | Linked common evaluator; experiment:68 | None | Same | Arithmetic / A most plausible |
| Preformer; transformer table/curve | Linked common evaluator; experiment:72 | None | Same | Arithmetic / A most plausible |
| PathFormer; transformer table | Linked common evaluator; experiment:72 | None | Same | Arithmetic / A most plausible |
| iTransformer; transformer table | Linked common evaluator; experiment:72 | None | Same | Arithmetic / A most plausible |
| LoSTer_MLP; MLP backbone ablation | Linked common evaluator; experiment:74 | None | Same | Arithmetic / A most plausible |
| LoSTer_DRNN; RNN backbone ablation | Linked metric and shared LoSTer experiment | None | Same | Arithmetic / A most plausible |
| LoSTer_F; convergence/spectral comparison | Linked common evaluator; experiment:68 | None | Same | Arithmetic / A most plausible; do not automatically equate this directory with the soft-k-means table column |
| LoSTer_KL; KL ablation | Linked common evaluator; experiment:56 | None | Same | Arithmetic / A most plausible |
| DEC; supplied additional baseline, absent main tables | Linked common evaluator; experiment:48 | None | Same | Arithmetic; no main table provenance to infer |
| Soft k-means / w/o CL table columns | No seed-resolved invocation/config recovered for these named conditions | Not recoverable for actual runs | Runtime unknown | Related implementation can use common evaluator, but exact table experiment mapping remains unestablished |

PathFormer's `net/utils/metrics.py` contains forecasting-error utilities, not an alternative NMI evaluator. The installed-requirements entries torchmetrics/tslearn are not evidence that their metrics generated these tables. No external score import script or historical normalization change was found.

Main NMI table M:563–579, transformer NMI table M:637–648, and KL/backbone ablation NMI table M:715–731 are most plausibly common arithmetic evaluation. Retail table M:778/781 has weaker provenance, especially Store Sales with no dataset pipeline. There is **no positive evidence for B (geometric actual evaluation) or C (mixed conventions)**, but the unchanged source alone cannot exclude uncommitted runtimes/external values. If requiring a demonstrated convention for the actual historical numbers, rather than the best-supported explanation, those executions remain unverified (D in the sense of not recoverable conclusively from available execution records).

The February original PDF already contains the table means, whereas the pinned public environment was added in July and includes packages released during 2025. It must not be presented as a proven February environment. Numeric agreement between rounded published means and source-controlled scalar tables does not identify a normalization convention.

**Repair without training:** if original run configuration/environment establishes arithmetic NMI, correct the manuscript equation/definition and state the version explicitly; retain historical numbers. If geometric evaluation is chosen for the revision, original predictions plus aligned labels, or full contingency matrices, allow recomputation without retraining. Neither is available here. RI/NMI scalars and cluster-size summaries alone do not determine the joint contingency table. No metrics were recomputed.

## Checkpoint and Prediction Recovery

**F10 verdict: no active historical centroid or exact-model artifact is recoverable from the inspected files/history.**

The first LoSTer version already separates wrapper `model.centroids` from active `KMeansLoss.centroids`. Current `main.py:100–113` initializes both copies from KMeans; `main.py:116` optimizes model parameters, both autoencoders and both loss-module centroid parameters together. The wrapper `net/model.py:201–205` stores its centroid parameter but its forward returns only reconstruction and embedding. Assignments in `experiment.py` use the loss-module centroids. Saving `model.state_dict()` at `experiment.py:135` therefore stores the wrapper's unused initialization copy, not the trained active original-view centroids. Every reachable save version follows the model-only pattern; no historical loss-state serialization repairs this.

`main.py:28–49` constructs these expected paths, identical across seeds:

- `models/LoSTer/ckpts/UCRArchive_2018/<dataset>/<experiment>/model.pth`;
- `.../autoencoder.pth` and `.../autoencoder_augmented.pth`;
- `.../model_augmented.pth` is named but not supplied to the final save routine;
- the `UCRArchive_2018` path prefix is also used for M5 output naming, so searches covered generic paths as well as obvious dataset names.

None of these checkpoints exists in the checkout or any historical tree. Seed-specific checkpoint names are absent; repeated launches would overwrite shared checkpoint paths. Final metric filenames do encode seed, but have not been preserved. The expected per-seed result path is `models/LoSTer/logs/UCRArchive_2018/<dataset>/<experiment>/metrics_LoSTer_<seed>.csv`, with columns Dataset, Model, Seed, Epoch, RI, ARI, NMI, Silhouette (`main.py:124–136`). The same output concept exists across baselines.

`main.py:141–150` constructs assignment lists in memory but prints/writes only each cluster's size. It does not save the observation indices or prediction vector. Initial KMeans can be rerun from recovered pretrained embeddings, but that cannot reconstruct the separately trained final centers. Final model weights plus the stale wrapper centers are insufficient for historical predictions.

W&B initialization (`experiment.py:116`) names project **Concrete Dense Network for Long-Sequence Time Series Clustering** and supplies `config=args`. Epoch logging (`:99–110`) records RI, ARI, NMI and losses. Neither the entity, actual immutable run ID nor exported config appears in the CSVs. Named histories may still exist remotely, but were not accessed and cannot be assumed recoverable. No local W&B run directory, artifact manifest or W&B model-upload call was found.

| Purpose | Available evidence | Can recover now? | Rerun requirement |
|---|---|---|---|
| Preserve published scalar values | Manuscript means, original PDF, convergence CSVs | Yes, as reported values and separate trajectories; cannot independently validate means | No for transcription/archive. Original final CSVs/W&B records could validate without training; otherwise new runs are needed for independent replication |
| Reconstruct published per-observation predictions | No predictions or contingency arrays; no active centers | No | Recover original assignments or complete state externally; otherwise training produces new predictions, not guaranteed historical assignments |
| Reconstruct exact trained models | No checkpoints; incomplete historical save design | No | Complete original states would be needed for exact recovery. A rerun supplies a new complete model, not proof of the exact old state |
| Resume exact historical training | No original/augmented active states, optimizer/scheduler, epoch/RNG/data order or previous stopping assignments | No | Exact continuation cannot be recovered from this evidence. A restart/new run is the available alternative if external training-state bundles are absent |

The save bug **does not invalidate metrics computed in memory**: training evaluates with the active centroids before saving and returns those scores/assignments (`experiment.py:123,140`). It is a persistence/reproducibility defect, not proof of an erroneous historical score. Full continuation would additionally require original and augmented network/criterion states, optimizer/scheduler state, random-generator states, augmentation/data identity and previous assignments used for the tolerance test. No such bundle was found.

## M5 Filter Impact

**F19 verdict: exact slice defect and historical selected indices recovered; membership impact remains unknown.**

The single historical notebook version (`ee0a197`) stores enough schema to prove the error:

1. Cell 10 groups by item_id, cat_id, store_id, resets the index, and displays 30,490 rows with columns `item_id, cat_id, store_id, d_1, ..., d_1913`.
2. Cell 14 merges price columns after the daily sales, preserving those first columns.
3. Cell 16 takes `sales_items.values[:,2:2+1913]`, namely string store_id plus d_1 through d_1912. Store names such as CA_1/TX_1 are nonzero strings; they contribute no zero count. d_1913 is missing.
4. It retains positional indices with the count <400. The saved `flags` output is a complete list of **1,150 distinct integer row positions**, not a boolean mask. First ten: 104,105,106,160,161,162,163,165,167,168. Last ten: 25272,25275,25276,25291,25292,26932,28822,29412,29592,30422. SHA-256 of `json.dumps(flags).encode()`: `876a0b20dfa0f0dea8f42af04e125a092e1862a1f74a4ba276a5d1a01196cc0a`.
5. Cell 17 records FOODS 858, HOUSEHOLD 256, HOBBIES 36; cell 18 records 29 cat-store labels. Cell 20 records sales/price shapes (1150,1913) and label vectors (1150,). Its correct post-label sales slice is `[4:4+1913]`, followed by STL(period=7) seasonal+trend. The erroneous filter does not itself shift the final daily-series slice.
6. The notebook's preprocessing progress reports 44 seconds; this is neither a clustering-training timing nor a replicate of the paper benchmark.

Let H be the historical zero count among d_1…d_1912 and z be 1 if d_1913 is zero, else 0. The intended count is H+z. Historic retention is H<400; corrected retention is H+z<400. A changed row must have **H=399 and d_1913=0**. Only removals are possible under the observed string-store schema; corrected membership is a subset of historical membership.

| Requested quantity | Established outcome |
|---|---|
| Historical subset size | 1,150, supported by full saved index list and notebook shapes; matches M:789 |
| Corrected subset size | Unknown; 1150 minus the number of retained rows satisfying H=399 and d_1913=0 |
| Symmetric membership difference | Unknown; exactly that removed-row count, with no additions under stored schema |
| Corrected label/class count | Unknown; cannot exceed 29, and decreases if all retained rows of a label are removed |
| Changed IDs | Not established; positional selection indices are known but boundary values are absent |
| Demonstrated published-score invalidity | None; neither changed membership nor resulting model-score differences have been demonstrated |

The recovered label-frequency vector is:

| Label | Count | Label | Count | Label | Count |
|---|---:|---|---:|---|---:|
| FOODS_CA_1 | 112 | FOODS_CA_2 | 66 | FOODS_CA_3 | 172 |
| FOODS_CA_4 | 64 | FOODS_TX_1 | 78 | FOODS_TX_2 | 98 |
| FOODS_TX_3 | 83 | FOODS_WI_1 | 53 | FOODS_WI_2 | 59 |
| FOODS_WI_3 | 73 | HOUSEHOLD_CA_1 | 23 | HOUSEHOLD_CA_2 | 24 |
| HOUSEHOLD_CA_3 | 60 | HOUSEHOLD_CA_4 | 1 | HOUSEHOLD_TX_1 | 28 |
| HOUSEHOLD_TX_2 | 30 | HOUSEHOLD_TX_3 | 28 | HOUSEHOLD_WI_1 | 18 |
| HOUSEHOLD_WI_2 | 25 | HOUSEHOLD_WI_3 | 19 | HOBBIES_CA_1 | 9 |
| HOBBIES_CA_2 | 2 | HOBBIES_CA_3 | 12 | HOBBIES_CA_4 | 3 |
| HOBBIES_TX_1 | 1 | HOBBIES_TX_2 | 2 | HOBBIES_TX_3 | 2 |
| HOBBIES_WI_1 | 3 | HOBBIES_WI_3 | 2 | HOBBIES_WI_2 | 0 (absent) |

The 29 positive counts sum to 1,150. Singleton HOUSEHOLD_CA_4 and HOBBIES_TX_1 demonstrate that a boundary removal could affect k; their actual boundary status is unknown.

### Minimal later data required

The smallest sufficient author-supplied record is the **1,150 historical selected rows**, each with stable item/store key, cat_store label, H (zeros in first 1,912 days), and d_1913 zero/nonzero status, together with the grouped-row ordering/checksum connecting it to saved flags. That determines removed keys, corrected size and surviving labels; rejected rows cannot be added by this correction.

If obtaining official data instead, only the keys and d_1…d_1913 columns of **sales_train_validation.csv** from the [official M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) are needed for the membership comparison. Calendar, sell_prices, sample_submission and evaluation/test sales are unnecessary for this narrow question. Full preparation of historical STL inputs is a separate task. No M5 data was downloaded here.

Correcting the filter requires reviewing the hard-coded `range(1150)` in cell 20 if the subset size changes; merely changing the slice can break downstream preprocessing. If membership is unchanged, this particular repair needs no clustering rerun. If membership changes, new corrected-subset experiments are conditional on adopting that cohort; a revised protocol must apply the same cohort to all compared methods. The original arrays/run provenance must still be checked before asserting that notebook execution generated the reported scores.

## Store Sales Provenance

**F21 verdict: no recoverable pipeline evidence in the complete reachable history or local subtree.**

Searches included Store Sales/store_sales/storesales/Favorita, 1729, 1684, family/store_nbr, Kaggle/date/sales, 33-class configuration and batch-32/timing code. No deleted or non-obviously named pipeline was recovered. Generic transformer calendar-feature helpers and M5 identifiers are not Store Sales preparation. Every supplied run script covers UCR/M5, and dataset dispatch recognizes only M5 specially; any other dataset name is sent to the UCR TSV route. A hypothetical conversion to TSV is possible, but no such conversion/shape/class manifest is supplied.

Direct evidence is confined to manuscripts/figures:

- Original PR-D-25-00861 PDF already reports 1729 x 1684 and 33 families, LoSTer RI 0.9523/NMI 0.4844, and the batch-32 timing comparison.
- Revised M:775–781 contains mean RI/NMI for LoSTer, IDEC, DNFCS, CKM, DTC, DTCR and DTCC; M:791 specifies the shape/classes; M:824 states the timing setup.
- `biblio.bib:376–380` cites `huy2023store`, a 2023 thesis titled *Store sales--time series forecasting in Ecuador using kaggle*. It provides neither a dataset URL/identifier nor extraction parameters.
- The plausible primary source is [Kaggle Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), whose official description includes store_nbr, family and sales and identifies Corporación Favorita. This is a **source identification inference**, not proof of the exact downloaded version, pivot, date interval or selected 1,729 series. It must not be conflated with other Favorita competitions.

No evidence establishes whether the table numbers were manually copied from original training logs, generated by uncommitted scripts, or imported from another source. They are literal TeX cells, but that observation does not prove external-score provenance. No historical pipeline deletion exists in advertised reachable Git history.

To reproduce the experiment requires the original data snapshot/checksum, date grid yielding length 1684, grouping key, the selection yielding 1729 rows, missing/zero treatment, transformations/normalization, family-label mapping, seed-resolved launch configs, output metrics and timing procedure. None is recoverable here. This is a reproducibility gap; it does not establish that the experiment never happened or that its scores are mathematically wrong.

## Convergence Provenance

**F22 convergence: curves and plot lineage recoverable; seed/config/five-run linkage unestablished.**

Both datasets have RI and NMI files with a common Epoch column 1–100. Fetal ECG files have 37 columns; UWave files have 43. Every run contributes six columns: _step, _step__MIN, _step__MAX, metric, metric__MIN, metric__MAX. Across all four exports, metric min/max equal the main metric whenever populated; these are not recovered five-seed intervals.

The notebook explicitly renames named columns to model names (cells 5/10 for fetal ECG, 20/25 for UWave). The mappings below are therefore author-recorded plotting mappings, not inferred from scores. “Column” is a **zero-based metric column position in both that dataset's RI and NMI files**; full header is `<display name> - RI score` or `<display name> - NMI score`. These random W&B-style display names are not immutable API run IDs. Seeds are **unknown for every row**.

| Dataset | Plotted method | Column | Run display name | Populated epochs RI/NMI | _step first → last | Last recorded RI / NMI |
|---|---|---:|---|---|---|---|
| NonInvasiveFetalECGThorax1 | LoSTer_F | 4 | devoted-sound-306 | 1–100 (100/100) | 29 → 2999 | 0.9649542405 / 0.6559626624 |
| NonInvasiveFetalECGThorax1 | IDEC | 10 | copper-waterfall-309 | 1–6 (6/6) | 29 → 179 | 0.9676046307 / 0.6840101070 |
| NonInvasiveFetalECGThorax1 | DTCC | 16 | amber-rain-344 | 1–100 (100/100) | 29 → 2999 | 0.9604235708 / 0.6229839410 |
| NonInvasiveFetalECGThorax1 | LoSTer | 22 | worthy-yogurt-341 | 1–11 (11/11) | 29 → 329 | 0.9688587626 / 0.6858052298 |
| NonInvasiveFetalECGThorax1 | CKM | 28 | driven-waterfall-297 | 1–6 (6/6) | 29 → 179 | 0.9674370741 / 0.6847918640 |
| NonInvasiveFetalECGThorax1 | DTCR | 34 | fancy-waterfall-343 | 1–100 (100/100) | 29 → 2999 | 0.9554995809 / 0.6239759666 |
| UWaveGestureLibraryY | Preformer | 4 | fearless-sun-353 | 1–100 (100/100) | 139 → 13999 | 0.6539821569 / 0.1664474289 |
| UWaveGestureLibraryY | LoSTer_F | 10 | silvery-fog-321 | 1–100 (100/100) | 34 → 3499 | 0.8393455849 / 0.4025614452 |
| UWaveGestureLibraryY | IDEC | 16 | honest-feather-317 | 1–3 (3/3) | 34 → 104 | 0.8412542124 / 0.4224063638 |
| UWaveGestureLibraryY | LoSTer | 22 | rare-tree-337 | 1–11 (11/11) | 34 → 384 | 0.8441842028 / 0.4300894745 |
| UWaveGestureLibraryY | DTCC | 28 | pleasant-grass-346 | 1–100 (100/100) | 34 → 3499 | 0.7731070739 / 0.1969464678 |
| UWaveGestureLibraryY | DTCR | 34 | brisk-valley-351 | 1–100 (100/100) | 34 → 3499 | 0.7950714175 / 0.2862063516 |
| UWaveGestureLibraryY | CKM | 40 | brisk-moon-313 | 1–6 (6/6) | 34 → 209 | 0.8408034907 / 0.4207474753 |

These are **single named exported histories**, most plausibly individual runs; no aggregation configuration or list of contributing runs proves otherwise. Identical min/max and one run name per series provide no basis to describe them as five-run means.

### Eleven LoSTer epochs and stopping evidence

The 100 CSV rows are the shared union of displayed epochs across methods that reached 100. LoSTer has actual values only at epochs 1–11, followed by 89 blanks; no later LoSTer measurements are recoverable. IDEC/CKM likewise end early. Source logs a metric once per joint epoch and allows early stopping when the changed-assignment fraction falls below tolerance (`LoSTer/experiment.py:127–132`). This explains the observed pattern plausibly, but without logs/config/stop-reason metadata, early stopping cannot be distinguished conclusively from interruption/export truncation.

Step increments support the supplied configurations: fetal ECG named histories increase by 30 per epoch, UWave non-Preformer by 35, and Preformer by 140. With batch logging plus epoch logging, these are consistent with floor(N/128)+1 and floor(N/32)+1 respectively. They do not establish full versus two-row-reduced data, because both candidate N values have identical batch floors. They also do not identify seed or immutable code commit.

Convergence generation is associated with code that had W&B logging (`d8b3746/96b76a0`) and the July 24 learning-rate changes (`3096275/a322caa`). Exact command/config is absent. `efb74ff` is the artifact introduction commit, not proven training commit; `9f52cb3` adds Preformer without changing any existing UWave CSV entries. Later `b594b65` only deletes argument declarations and does not prove a different convergence architecture.

### Epoch alignment and published figures

The CSVs align named metrics by explicit Epoch; RI and NMI populated spans agree for each run. W&B _step differs by method and is not used as the epoch axis. Notebook cells 4/9 select [0,4,10,16,22,28,34]; UWave cells 19/24 add column 40. No within-CSV epoch mismatch was found.

However the final plotting helpers (cells 14 and 29) call `ax.plot(np.arange(0,len(df)), df.values[:,i])`, discarding the 1-based Epoch index. Thus saved epoch 1 is plotted at x=0 and saved epoch 11 at x=10. This is a confirmed **one-epoch axis-label offset**, repairable without training. Ordinary exploratory `df.plot()` cells use the index, but the final PDF helpers do not. NaN tails leave curves ending early; no 100-epoch LoSTer curve is present.

SHA-256 proves exact Figure 2/3 lineage:

- `figures/plots_NonInvasiveFetalECGThorax1.pdf` equals both manuscript Figure_2 copies: `4c97e5fcd5cc48aa926427861e485565b1830299b7beae3e2e8e1e15a6ff7c91`.
- `figures/plots_UWaveGestureLibraryY.pdf` equals both Figure_3 copies: `367fa6f7286d97ad05fd8b0c72c9afe3cbfb97da40902b2515df3443182385a8`.

The LoSTer fetal-ECG curve ends at RI 0.9688587626/NMI 0.6858052298, while M:528/564 reports five-run means 0.9737/0.7399. UWave ends at 0.8441842028/0.4300894745 versus table means 0.8483/0.4388. Distinct values are compatible with one run versus a mean, or different runs/configs; they do not validate that these histories belong to the five-run table cohorts.

## Timing Provenance

**F22 timing: published scalar times and broad hardware recovered; measurement procedure and raw timings unavailable.**

Figure_4's text contains exact Store Sales bars: LoSTer 1.648, IDEC 0.69, DNFCS 0.795, CKM 0.699, DTC 76, DTCR 263, DTCC 165 seconds. M:824 gives the same baseline values and describes LoSTer as under 2 seconds. M:813–816 records StarLightCurves train/inference times: LoSTer 2.85/1.81, Preformer 1531/270, Pathformer 550/60, iTransformer 3.66/2.03 seconds. These were transcribed, **not generated or remeasured**.

| Provenance component | Store Sales | StarLightCurves transformer comparison |
|---|---|---|
| Hardware | M:824: single NVIDIA A4000 GPU, 24-core Intel Xeon at 2.40 GHz | “All running times” sentence suggests same hardware; no independent hardware log |
| Batch size | M:824 explicitly 32 for each timed method | No benchmark-specific batch manifest; supplied scripts use LoSTer/PathFormer/iTransformer 128 and Preformer 32 |
| Training definition | Average time for one epoch in paper | One training data pass in M:826 |
| Inference definition | No inference timings reported here | One inference data pass; which views/encoder/decoder/assignments included is unspecified |
| Warm-up / GPU synchronization | Not reported; no benchmark script | Same |
| I/O, preprocessing, validation, logging included? | Unknown | Unknown |
| Pretraining versus joint phase; loss/evaluation work | Unknown beyond “epoch” | Unknown beyond “training data pass” |
| Repeat count, selected epochs, variance, outlier handling | “Average” claimed; samples/count unavailable | Repeat scheme not reported |
| Framework/environment/code commit | No executed config; no runtime record | Same |
| Generation script / raw measurements | None in local files or reachable history | None |
| Current batch-32 code hit | Preformer launch script, unrelated to Store Sales | Supports a possible transformer config, not measured execution provenance |

Complete historical source search found no `time.time`, `perf_counter`, `timeit`, CUDA event timing or synchronization benchmark implementation producing these results. Notebook progress bars do not establish timing methodology. Generic filenames such as timefeatures and transformer helper metrics contain no such benchmark. Figure_4 is a rendered PDF with Chromium/Skia metadata (created August 2, 2025), not raw measurement output; its SHA-256 is `bbfac65aa77cb026386b47f51ca51c7766497f10663473a8109a796fd6c34641`.

The original February PDF says Preformer's batch size was reduced to 16 for some memory-heavy accuracy experiments, whereas its supplied July script uses 32. This is another reason not to infer historical benchmark/accuracy settings solely from current scripts. It does not prove that timing runs used 16 or that either comparison is invalid.

Existing timing values can be archived without retraining. To substantiate speed claims, recover original benchmark script/config/raw repeats/hardware record. If unavailable, a separately authorized benchmark must define synchronized warm-up, scope of training/inference passes, preprocessing/I/O inclusion, method-specific batches and repeated measurements. Training full five-seed experiments is not automatically required for a benchmark, but representative trained state/initialization must be specified.


## Per-Seed Results and Statistics Recovery

No method/table reaches **FULL PER-SEED RESULTS RECOVERABLE**. Published means are rounded summaries, not sufficient statistics for standard deviation or ARI. RI/NMI trajectories describe changes during one named history, not variation between five independent seeds. The duplicate __MIN/__MAX columns are not confidence intervals, and a 100-row CSV is not 100 independent experiments.

The following classification is deliberately scoped to the named table/result. Where related curves exist, their **PARTIAL RESULTS RECOVERABLE** status is listed separately and does not imply that one of the five table seeds has been identified.

| Method / condition | Table or dataset scope | Recovery classification | Related recoverable evidence / limitation |
|---|---|---|---|
| LoSTer | Main UCR RI/NMI tables; 17 datasets | **ONLY PUBLISHED MEAN RECOVERABLE** | Two separate named curves recoverable; no evidence mapping them to the five table runs |
| CKM | Main UCR RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Two named convergence histories, seeds unknown |
| IDEC | Main UCR RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Two named convergence histories, seeds unknown |
| DNFCS | Main UCR RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Code/launch/aggregation only; no saved trajectories or final runs |
| DTC | Main UCR RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Same |
| DTCR | Main UCR RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Two named convergence histories, seeds unknown |
| DTCC | Main UCR RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Two named convergence histories, seeds unknown |
| LoSTer | Transformer comparison tables; 12 UCR datasets | **ONLY PUBLISHED MEAN RECOVERABLE** | Shared reported means; no recoverable five-run set |
| Preformer | Transformer comparison tables | **ONLY PUBLISHED MEAN RECOVERABLE** | One UWave curve, seed/config unverified |
| PathFormer | Transformer comparison tables | **ONLY PUBLISHED MEAN RECOVERABLE** | No saved run trajectories or final values behind mean |
| iTransformer | Transformer comparison tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Same |
| LoSTer_MLP / LoSTer_DRNN / LoSTer_KL | Ablation RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Named directories and launch recipes; no per-seed outputs |
| Soft k-means / w/o CL conditions | Ablation RI/NMI tables | **ONLY PUBLISHED MEAN RECOVERABLE** | Exact condition-to-invocation mapping **NOT ESTABLISHED** |
| LoSTer / LoSTer_F / CKM / IDEC / DTCR / DTCC | Fetal ECG 1 convergence Figure 2 | **PARTIAL RESULTS RECOVERABLE** | Six named RI/NMI histories, no ARI, seed or config export |
| Same six methods plus Preformer | UWave Y convergence Figure 3 | **PARTIAL RESULTS RECOVERABLE** | Seven named RI/NMI histories, same limitations |
| LoSTer / CKM / IDEC / DNFCS / DTC / DTCR / DTCC | M5 retail table | **ONLY PUBLISHED MEAN RECOVERABLE** | Historical preprocessing selection available; no model per-run scores |
| Same seven methods | Store Sales retail table | **ONLY PUBLISHED MEAN RECOVERABLE** | Shape/means only; pipeline and runs missing |
| Same seven methods | Store Sales timing Figure 4 | **ONLY PUBLISHED MEAN RECOVERABLE** | Reported bars/claimed averages; raw measurement count/samples unavailable |
| LoSTer / Preformer / PathFormer / iTransformer | StarLightCurves timing table | **PARTIAL RESULTS RECOVERABLE** | Scalar timings recovered; whether each is a repeated-measurement mean **NOT ESTABLISHED** |
| DEC | No main-paper table | **NOT ESTABLISHED** | Implementation exists, but no reported table cell or historical result artifact to recover |

All supplied `results.py` scripts calculate mean/std of per-seed RI, ARI, NMI and epoch, with `np.std` default **ddof=0**. All use the 17-UCR-dataset list; none includes M5/Store Sales. Their `path_results='./results.csv'` variable does not yield a written results file: they print their DataFrame and overall means. No historical stdout or aggregate CSV was found. M5 launches could generate seed-specific metrics under the generic output layout, but retail aggregation is not provided.

**ARI:** the common evaluator computes ARI and W&B logs it; final per-seed CSV schema includes it. The available exports preserve only RI/NMI. ARI cannot generally be deduced from RI/NMI scalars without marginals/contingency structure. No saved predictions, contingency matrices or actual ARI values were found.

**Variance/std:** a rounded five-run mean permits many different five-tuples and variances. Even recovering one seed would leave the variance unidentified. Different epochs from a curve must not be used as seed replicates. If original per-seed CSVs or W&B scalar histories/configs are recovered, ARI and std can be recovered without training; the revised reporting plan must distinguish ddof=0 historical std from any newly selected sample-std convention. No statistics were calculated in this audit.

## Provenance Matrix

“Raw data available?” means present in the local inspected project; the four small official-data checks were in memory and are not a retained historical-input archive. “Per-seed metrics” means identified final seed records, not a named curve with an unknown seed. The commit field distinguishes **artifact introduction** from **run generation**. Confidence applies only to the claim actually supported in Notes. Rerun decisions concern recovering/validating the stated result; original values remain archivable without experiments.

| Artifact / Result | Dataset | Method | Paper location | Current source code available? | Raw data available? | Per-seed metrics available? | Predictions available? | Exact model state available? | Generation script available? | Likely generating commit | Confidence | Rerun required? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Main accuracy means | UCR 17 | LoSTer | M:527–543,563–579 | Yes | No locally | No | No | No | Launch/aggregate yes; actual command no | Run UNKNOWN; source from 92a2df7 | LOW | CONDITIONAL | Corrected-input evaluation needs new runs if historic parsing was defective; original logs could validate scalar provenance |
| Main accuracy means | UCR 17 | CKM | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 6669c95 | LOW | CONDITIONAL | Common evaluator; separate curve artifacts do not identify table seeds |
| Main accuracy means | UCR 17 | IDEC | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 3a2cc72 | LOW | CONDITIONAL | Same |
| Main accuracy means | UCR 17 | DNFCS | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from fb90e1c | LOW | CONDITIONAL | No independent saved result data |
| Main accuracy means | UCR 17 | DTC | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 5a3616b | LOW | CONDITIONAL | Same |
| Main accuracy means | UCR 17 | DTCR | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 5324803 | LOW | CONDITIONAL | Same |
| Main accuracy means | UCR 17 | DTCC | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 7cac262 | LOW | CONDITIONAL | Same |
| Transformer accuracy means | UCR 12 | LoSTer | M:606–617,637–648 | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN | LOW | CONDITIONAL | Means reused across table contexts do not recover run identity |
| Transformer accuracy means | UCR 12 | Preformer | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 523d65e | LOW | CONDITIONAL | UWave named curve separate; batch/config mismatch across snapshots |
| Transformer accuracy means | UCR 12 | PathFormer | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from bb4259a | LOW | CONDITIONAL | Wrapper loader truncates sequence length; actual run config missing |
| Transformer accuracy means | UCR 12 | iTransformer | Same tables | Yes | No locally | No | No | No | Launch/aggregate yes | Run UNKNOWN; source from 2037845 | LOW | CONDITIONAL | Common evaluator, no seed records |
| Backbone/KL ablation means | UCR 17 | MLP, DRNN, KL | M:679–695,715–731 | Yes | No locally | No | No | No | Named launch scripts yes | Run UNKNOWN; July 18 source introductions | LOW | CONDITIONAL | Seed-resolved table lineage absent |
| Other ablation means | UCR 17 | Soft k-means; w/o CL | Same ablation tables | Related code; exact mapping unknown | No locally | No | No | No | Exact invocation no | UNKNOWN | UNKNOWN | UNKNOWN | Must recover condition definitions before rerun design |
| Fetal ECG CSV histories | Fetal ECG 1 | Six mapped models | Figure 2; M:745–764 | Yes | No locally | No; named runs only | No | No | Plot yes; W&B producer source yes | Artifact efb74ff; run UNKNOWN | HIGH | NO | Existing RI/NMI curves exactly recoverable; no five-seed validation |
| UWave CSV histories | UWave Y | Seven mapped models | Figure 3; M:745–764 | Yes | No locally | No; named runs only | No | No | Plot yes; W&B producer source yes | Artifact efb74ff/9f52cb3; run UNKNOWN | HIGH | NO | Old entries preserved when Preformer added |
| Figure 2/3 PDFs | Same two UCR datasets | Same mapped models | Figure 2/3 | Yes | No locally | No | No | No | plots_training.ipynb | Final rendering 9f52cb3 | HIGH | NO | Exact PDF hashes match both manuscript trees; epoch offset repair needs only replot |
| Historical filter indices/labels | M5 | Preprocessing | M:789 | Yes | No full daily arrays | Not applicable | Not cluster predictions | Not applicable | Executed notebook source yes | Artifact ee0a197; execution commit not encoded | HIGH | NO | 1150 positions and 29 counts recoverable from outputs |
| Corrected filter membership | M5 | Preprocessing | Revised cohort not yet chosen | Historical recipe yes; repair not made | No necessary boundary values | Not applicable | Not applicable | Not applicable | Historical notebook; intended comparison defined here | UNKNOWN | UNKNOWN | CONDITIONAL | Data-only calculation resolves; no training required |
| Retail accuracy means | M5 | LoSTer | M:777–778 | Yes | No sales.npy/labels | No | No | No | Preprocessing/launch yes; retail aggregate no | Run UNKNOWN | LOW | CONDITIONAL | If corrected membership changes and adopted, rerun; active state unavailable |
| Retail accuracy means | M5 | Six standard baselines | M:777–778 | Yes | No | No | No | No | Preprocessing/launch yes; retail aggregate no | Run UNKNOWN | LOW | CONDITIONAL | Apply common chosen subset; original per-seed metrics could validate historic table |
| Retail accuracy means | Store Sales | LoSTer | M:780–781 | Model yes; pipeline no | No | No | No | No | No dataset generation/launch recipe | UNKNOWN; result existed by February PDF | UNKNOWN | CONDITIONAL | New reproducible experiment requires recipe and runs if original artifacts unavailable |
| Retail accuracy means | Store Sales | Six standard baselines | M:780–781 | Models yes; pipeline no | No | No | No | No | No dataset generation/launch recipe | UNKNOWN | UNKNOWN | CONDITIONAL | Same |
| Timing bars | Store Sales | Seven methods | Figure 4; M:824 | Models yes; benchmark no | No | No raw repeated timing | No | No | No benchmark script | Run UNKNOWN; PDF created 2025-08-02 | LOW | CONDITIONAL | Original measurements could substantiate; otherwise new benchmark needed |
| Timing table | StarLightCurves | LoSTer + 3 transformers | M:813–816 | Models yes; benchmark no | No locally | No raw repeats | No | No | No benchmark script | UNKNOWN; numbers present February PDF | LOW | CONDITIONAL | Current scripts do not identify warm-up/scope/batch settings |
| Actual five-run ARI/std | All table datasets | All reported methods | Not reported as raw values | Computation source yes | No full historical inputs | No | No | No | UCR aggregate yes; retail aggregate no | UNKNOWN | UNKNOWN | CONDITIONAL | Recover original CSVs/W&B or produce new five-run experiments |
| Complete state for a reproducible replacement run | Any chosen dataset | LoSTer | Method/checkpoint provenance | Yes | No historical inputs | No | No | No | Save routine incomplete | No state-producing artifact | HIGH | YES | New complete state requires new training if no external bundle; exact old-state recovery remains impossible |
| Alternative NMI from recovered assignments | All datasets | All methods | NMI definition/tables | Common evaluator yes | Historical aligned labels absent | No | No | No | Evaluation source yes | UNKNOWN | UNKNOWN | CONDITIONAL | Would become METRIC RECOMPUTATION ONLY if predictions/contingencies recovered; otherwise fresh runs needed |
| Original submission/source archive | All reported datasets | All methods | Original vs R1 document snapshots | Source snapshot yes | No | No | No | No | Document source yes | Git import 0c83687; creation/export dates distinct | HIGH | NO | Recover reported numbers/content, not experiments |
| Deleted M5 instructions | M5 | Preparation | README/dataset setup | Historical text yes | No | Not applicable | No | No | Historical recipe yes | Added ee0a197, deleted 99df4aa | HIGH | NO | Recover with git show; no unique lost science |

No matrix row is currently classified **METRIC RECOMPUTATION ONLY**, because the necessary historical assignments/contingencies are absent. That would be the appropriate rerun category after such artifacts are recovered, not an assertion that they exist today.

## F10/F11/F19/F21/F22/F23 Decision Table

Priorities: P0 = before any new experiment; P1 = before final manuscript rewrite; P2 = reproducibility improvement; P3 = archive only. These are recommendations, not authorization to change code or execute experiments.

| Finding | Current verdict | New evidence | Impact on published results | Can repair without retraining? | Recommended action | Priority |
|---|---|---|---|---|---|---|
| F10 | Confirmed incomplete save design; no active historical states/predictions recovered | All historical saves model-only; same checkpoint names across seeds; no serialized states in any reachable tree or local ignored directory | In-memory scores remain valid as computed; exact reconstruction/resumption unavailable | Future save design can be repaired without training; old absent states cannot be recreated by a code edit. Original scalar CSVs/W&B could avoid scalar reruns | Recover original external artifacts first; define complete seed-specific persistence before any new experiment; do not call new training an exact recovery | **P0** for new-run design; P2 for archival state recovery |
| F11 | **CONFIRMED PARSER BUG, IMPACT QUANTIFIED** | Official numeric TSV rows; eight exact files; all 17 metadata counts; all supplied loader families retain default header inference | Two samples lost on standard inputs; four checked k values unchanged; historical score/ranking effect unknown | Data/header/provenance check without training; parser repair without training. Corrected full-cohort training usually requires new runs if defective historical parsing confirmed | Establish historical file identities/counts; approve canonical headerless protocol and consistent affected-method rerun scope before new experiments | **P0** |
| F19 | Confirmed wrong filter slice; actual changed membership unknown | Full saved 1150-index list and 29 labels; inclusion of string store_id/omission of d_1913; exact boundary-removal condition derived | No demonstrated change or numerical invalidity; potential cohort and k change | Yes for data-only membership calculation. Clustering reruns unnecessary if no change; conditional if corrected cohort differs and is adopted | Obtain selected-row boundary summary or validation sales columns; compare membership/labels before choosing M5 reruns; replace hard-coded size only in later authorized work | **P0** |
| F21 | Store Sales pipeline unavailable in all reachable history | Result already in February original PDF; only manuscript numbers/citation and timing PDF survive; no deleted preparation path | Reproducibility gap, not evidence fabricated/wrong scores | Could recover original pipeline/arrays/metrics without training; cannot reproduce new results from current repository alone | Obtain original preparation/config/data manifest; if unrecoverable, separately define a reproducible replacement experiment or revise unsupported claims | **P0** for retaining/rerunning this dataset |
| F22 | Convergence artifacts traceable, single-run provenance incomplete; timing method unavailable | Full run-name/column/span map; 11 LoSTer epochs; final-PDF x-axis offset; identical Figure 2/3 hashes; exact timing bars and dated original PDF | Curves are not five-run statistics; one-epoch labeling error; performance timing claims not independently verifiable here | Curves/labels/lineage repair yes. Original benchmark records may resolve timing without new runs; otherwise fresh timing measurement needed | Replot with actual Epoch, label curves as named runs unless aggregate evidence emerges; recover seed/config and benchmark records; explicitly define benchmark before new measurement | **P1**; P0 before any new benchmark |
| F23 | Shared arithmetic implementation certain under pin; arithmetic table convention most plausible but runtime unverified | One unchanged evaluator, 14 metric symlinks, no average_method, unchanged sklearn pin | Geometric printed definition mismatches supplied implementation; no demonstrated heterogeneity or numeric score error | Manuscript-only repair if arithmetic run provenance confirmed; metric recomputation if assignments recovered; training only if required predictions unavailable | Verify runtime/table provenance; choose/document one explicit NMI convention before further experiments; preserve historic values pending evidence | **P0** convention decision; P1 equation correction |

## What Must Be Rerun

These are conditional research requirements, **not instructions executed in this task**.

1. **Complete new trained models or new predictions:** with the current evidence, training is needed to produce these artifacts. Fixing save code alone cannot supply missing past states. Exact historical state/resumption is an artifact-recovery problem; rerunning cannot guarantee byte-identical reconstruction.
2. **Five-run statistics/ARI:** if the authors cannot provide the original five per-seed records, aligned predictions/contingencies, or suitable W&B histories, fresh five-run experiments are necessary for new std/ARI and independent validation. They must be reported as new experiments; their statistics cannot be attributed retroactively to the old means.
3. **Canonical full UCR evaluation:** if the reported runs used this parser on standard files, retrain/re-evaluate the affected compared methods on the corrected full pool to support revised full-dataset claims. Existing reduced-cohort metrics cannot simply be relabeled as full-cohort metrics. Historical input provenance should be resolved first.
4. **Corrected M5 evaluation:** only if the data-only comparison shows changed membership and the intended filter is adopted. If the sets are identical, F19 alone does not warrant a clustering rerun.
5. **Store Sales reproduction:** if retaining that experiment and original data/pipeline/run records cannot be recovered, preparation and new runs are required. Its shape/filter/labels must be defined before executing methods.
6. **Timing validation:** if benchmark script/raw measurements cannot be recovered, fresh repeated measurements are necessary to substantiate retained timing claims. They need their own defined benchmark scope; full accuracy retraining is not inherently required solely to time an epoch/inference pass.
7. **Geometric NMI:** if the revision deliberately adopts the printed geometric equation, use recovered predictions/contingencies where possible. Training is conditional only when those assignments cannot be recovered; a definition mismatch by itself does not compel all-method training.

There is no evidence-based mandate here to rerun every historical experiment indiscriminately. Resolve data cohorts, metric convention and artifact availability first. The definite limitation is that this repository alone cannot reconstruct missing five-run records or full learned states.

## What Can Be Recovered Without Training

- Preserve the exact original/revised table means and reported timings, with explicit provenance limits.
- Recover both current convergence figures from four CSVs and the notebook, including thirteen named histories, spans and last recorded scalars.
- Recover pre-Preformer convergence CSV/PDF/notebook versions and show that adding Preformer preserved every previous UWave entry.
- Correct the one-epoch plotting axis offset in separately authorized replotting; no new metric values are needed.
- Recover all 1,150 historical M5 grouped-row indices, 29 label counts, array shapes and STL/filter recipe from stored notebook output.
- Calculate the M5 membership correction later from the minimal boundary summary or official validation sales data, without clustering.
- Recover the deleted M5 instructions from Git; this is archival (P3), since README already preserves the recipe.
- Establish default-header row loss for all 17 canonical UCR datasets using metadata/schema, and unchanged k for the four inspected datasets.
- Identify the shared supplied NMI implementation and fixed library pin; correct documentation after execution convention is established.
- Compute ARI/NMI alternatives/std later from **subsequently recovered** original predictions/contingencies/per-seed scalars. These artifacts are not present now.
- Inspect/export original W&B histories and configs if the authors supply authorized access or exports. The existing run display names/project name provide search leads, not a completed remote recovery.

## Remaining Unknowns

1. The actual February table-generating code/environment, input file hashes, included sample indices, seeds, run dates and aggregation records.
2. Whether the authors added headers locally or used another unpublished parser; canonical sample loss is established, historical executed cohort is not.
3. Pooled class frequencies/first-row identities for the thirteen raw datasets not inspected, and historical k for all table executions.
4. M5 H=399/d_1913=0 boundary rows, corrected cohort/class count, and whether the stored notebook output/arrays generated the table scores.
5. Original final per-seed RI/ARI/NMI records, complete models, active original/augmented centroid states, aligned labels/predictions and training-resume state.
6. Immutable W&B entity/project/run IDs, seed/config exports, stop reasons and whether named convergence histories belong to any table seed cohort.
7. Exact Store Sales source version, pivot/filter/date/missing-value/normalization recipe and label manifest.
8. Accuracy and benchmark batch settings for historical transformer runs, benchmark warm-up/synchronization/scope/repeats and raw measurements.
9. Exact soft-k-means/w/o-CL invocation mappings and historical common-versus-external evaluation provenance.
10. Separate PR-D-25-02649 manuscript/code snapshot and any unpublished/unreachable history or author-machine archive outside this project.

## Recommended Next Step

Perform one targeted **artifact/protocol resolution step before training or manuscript rewriting**. Ask the authors to supply original per-seed metric CSVs or W&B exports with configs, historical UCR headers/counts/checksums, the small M5 selected-row boundary summary, and the Store Sales/benchmark recipes. Resolve arithmetic-versus-geometric NMI and freeze the intended dataset cohorts.

Then prepare a separately authorized minimal repair/rerun plan: explicit header handling and cohort assertions, complete seed-specific state/prediction persistence, conditional M5 preparation fixes, explicit metric semantics, and a documented benchmark. Replotting and manuscript-only changes can proceed under their own later authorization once provenance decisions are recorded. Preserve the current historical source/results rather than overwriting them.


## Completion Verification

Stage A HEAD and origin/revision-2026 both remain f25f9194568e074827fd8063947b3bbf747a7aab. Stage B created only this report; it was not committed or pushed.

A before/after manifest covered all **298 pre-existing non-Git project files**, including ignored Drafts/data/log locations. For each file it compared relative path, byte size, nanosecond modification time and SHA-256; every bucket matched exactly. Models, both manuscript trees, all three Drafts artifacts, both notebooks, all four convergence CSVs, figures, requirements, safeguards and prior reports are unchanged. All twelve ZIP entries still equal the corresponding files in both manuscript trees byte for byte.

The provenance matrix has fourteen columns in every row. Artifact sizes and current/source Git identities were checked. No training, metric recomputation, benchmark, package installation, scientific code change, manuscript compilation or large dataset download occurred. Public-data checks were in memory; no temporary files were created.

Final git status --short:

```text
?? analysis/audits/02-provenance-recovery-audit.md
```
