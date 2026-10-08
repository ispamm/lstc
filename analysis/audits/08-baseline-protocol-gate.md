# LoSTer 2026 Baseline and Fair-Comparison Gate

## Executive Decision

**CORE-44: NOT AUTHORIZED.** This is a source/protocol gate dated 2026-10-08, not an implementation or experimental campaign. Retain all **11 Tier-1 comparators**: **3 READY FOR IMPLEMENTATION**, **5 READY WITH CONDITIONS**, **3 BLOCKED**. No Tier-1 method is moved or dropped. Readiness means eligibility to begin engineering, not a successful reproduction or permission to run Final36.

Stage A is complete: commit **cab7b6ffecab92bfcf6fe46e73e32dc0c040ee59**, message *Finalize LoSTer 2026 method after warp validation*, on **revision-2026**, pushed only to **origin**. Annotated tag **loster-2026-frozen**, message *Frozen LoSTer 2026 method: Legacy-Clean with MonotoneWarp*, also pushed only to origin. No upstream push occurred. The protected-path diff was empty before the commit; staged whitespace verification passed with the repository's CRLF convention treated as end-of-line whitespace.

The final method remains **LoSTer-Legacy-Clean + MonotoneWarp**. No scientific code, frozen configuration, manuscript or historical output is changed by this audit. The frozen specification's source_committed=false and parent_git_sha record the validation campaign's contemporaneous provenance; they do not describe the subsequently committed/tagged working tree. Their bytes remain unchanged. The benchmark manifest remains design-only with selected_variant=null; this gate refers to the tagged frozen specification without rewriting that manifest.

The immediate blockers are missing official CKM code/reproduction details, DTCC's contradictory dependency declaration and disconnected cluster-contrast gradient, and CDCC's released augmentation/temporal-axis behavior. TFMCC needs a valid environment and a globally frozen resource configuration. None is resolved by substituting a local historical implementation or reducing capacity after observing Final36 accuracy.

Only this report and the [baseline registry](../baselines/baseline-registry-2026.json) are new repository deliverables. Baseline environments were not installed; baseline code was not imported/executed; no synthetic fit, Development8 training, Final36 fit or CORE-44 run occurred.

## Sources and Verification Method

Read the supporting [Audit 01](01-paper-code-audit.md), [Audit 02](02-provenance-recovery-audit.md), [literature Audit 03](../literature/03-literature-novelty-benchmark-audit.md), [Audit 04](04-method-and-experiment-plan.md), [Audit 05](05-implementation-readiness.md), [pilot Report 06](../experiments/06-pilot-results.md), [warp Report 07](../experiments/07-warp-validity-resolution.md), and [frozen specification](../experiments/frozen-method-2026.json). Audit 04 governs the current five final seeds and development ceiling; older ten-run proposals in Audit 03 do not override it.

Verification combined primary publisher/proceedings metadata, final author-hosted/publisher PDFs, author-linked repositories, release tags, full Git object IDs, root licenses/dependency declarations, and static inspection of loader, preprocessing, training, extraction, metric, checkpoint and seed paths. Notebook cells were parsed as JSON without execution. Paper PDFs were downloaded/extracted for reading only. All ten source clones reside outside the tracked repository at:

G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/src

All selected clone working trees were checked clean. R-Clustering is detached at its publication **submission** tag; its algorithm notebook matches current HEAD, whose later changes are metadata. KASBA and scikit-learn are detached at the selected release tags. Other author sources have no publication tag established; the pinned author repository snapshot is reproducible, but is not proof that those exact bytes produced the original paper table.

Source absence is bounded evidence: CKM title, authors, institutional paper page and repository searches did not identify official/author code. This is **NO IMPLEMENTATION FOUND**, not proof that no private/lost implementation ever existed. DTCC's final publisher metadata/source link were verified; accessible detailed method evidence includes the authors' preprint. Exact final-paper parameter equivalence remains unresolved and is not asserted.

Provenance and correspondence are separate. OFFICIAL says who released the code; it does not certify that it implements the paper correctly. A canonical maintained library for Lloyd k-means is explicitly THIRD-PARTY relative to Lloyd's paper, rather than falsely classified as author code.

The registry records full source pins, exact dataset memberships, conditions and evidence URLs. Frozen byte anchors are:

| Anchor | SHA-256 |
| --- | --- |
| Frozen method JSON | 89ace607dbb10392733974753bc13275de9a41bc1a668f95fa1ed47acd030a1e |
| Final benchmark manifest | be753c357d5176101c9d40ad55e68eb66e263243d5776b495455b945c9d0ec7e |
| Common metrics module | 8c9dc5afc5ebcd9955c56787311e67c43ada7ccfe97d387f1ec6fae0910375b1 |

Primary PDFs are retained externally in baselines/audit-papers. Their hashes are: CKM b9242de91d862574f408a4e6fd62072ef1359f567ec045d6733cea6e28312490; CDCC fd36048237074ce73da269f788c55a556c1e79286959b53a3494d32170c95f89; FCACC 494be232a54513a06dac08c5de3c1076d7900b79fdc3403a379bab0774406277; TFMCC dc24e83527a1bdfc27985004da25d42650cb3f1a4f835c7384b0542146e275a4; FASA 2d1bd81a81ab20dc0b2af5e7fe45533fdb15a114b06b0b49252250251ce08732; k-Shape a10ae60a47746727effd4ca9d7fb1a54eebf7e0cf742ea1a83111f61aa7f8f56.

## Tier-1 Baseline Overview

The following bibliography is the canonical identity of each comparator. A representation learner with a newly declared k-means head is identified as a pipeline rather than attributed as a native clustering experiment.

| Comparator | Exact publication and authors | Venue/year | DOI / primary source |
| --- | --- | --- | --- |
| Euclidean k-means | *Least squares quantization in PCM*. Stuart P. Lloyd. | IEEE Transactions on Information Theory 28(2), 129-137; 1982 | [10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489) |
| k-Shape | *k-Shape: Efficient and Accurate Clustering of Time Series*. John Paparrizos; Luis Gravano. | ACM SIGMOD,1855-1870; 2015 | [10.1145/2723372.2737793](https://www.paparrizos.org/papers/PaparrizosSIGMOD15.pdf) |
| CKM | *Deep Clustering with Concrete K-Means*. Boyan Gao; Yongxin Yang; Henry Gouk; Timothy M. Hospedales. | IEEE ICASSP,4252-4256; 2020 | [10.1109/ICASSP40776.2020.9053265](https://www.pure.ed.ac.uk/ws/files/134695608/Deep_Clustering_with_Concrete_GAO_DOA24012020_AFV.pdf) |
| DTCC-2023 | *Deep Temporal Contrastive Clustering*. Ying Zhong; Dong Huang; Chang-Dong Wang. | Neural Processing Letters55,7869-7885; 2023 | [10.1007/s11063-023-11287-0](https://link.springer.com/article/10.1007/s11063-023-11287-0) |
| CDCC | *Cross-Domain Contrastive Learning for Time Series Clustering*. Furong Peng; Jiachen Luo; Xuan Lu; Sheng Wang; Feijiang Li. | AAAI38(8),8921-8929; 2024 | [10.1609/aaai.v38i8.28740](https://ojs.aaai.org/index.php/AAAI/article/view/28740) |
| FCACC | *Fuzzy cluster-aware contrastive clustering for time series*. Congyu Wang; Mingjing Du; Xiang Jiang; Yongquan Dong. | Pattern Recognition173,112899; 2026 | [10.1016/j.patcog.2025.112899](https://doi.org/10.1016/j.patcog.2025.112899) |
| TFMCC | *Time-Frequency Augmented Multi-level Contrastive Clustering for Time Series*. Congyu Wang; Mingjing Du; Xiang Jiang. | AAAI40(31),26142-26150; 2026 | [10.1609/aaai.v40i31.39817](https://ojs.aaai.org/index.php/AAAI/article/view/39817) |
| TS2Vec + k-means | *TS2Vec: Towards Universal Representation of Time Series*. Zhihan Yue; Yujing Wang; Juanyong Duan; Tianmeng Yang; Congrui Huang; Yunhai Tong; Bixiong Xu. | AAAI36(8),8980-8987; 2022 | [10.1609/aaai.v36i8.20881](https://ojs.aaai.org/index.php/AAAI/article/view/20881) |
| R-Clustering | *Time series clustering with random convolutional kernels*. Jorge Marco-Blanco; Rubén Cuevas. | Data Mining and Knowledge Discovery38(4),1862-1888; 2024 | [10.1007/s10618-024-01018-x](https://link.springer.com/article/10.1007/s10618-024-01018-x) |
| KASBA | *Rock the KASBA: blazingly fast and accurate time series clustering*. Christopher Holder; Anthony Bagnall. | Data Mining and Knowledge Discovery40,article21; 2026 | [10.1007/s10618-026-01189-9](https://link.springer.com/article/10.1007/s10618-026-01189-9) |
| FASA | *MUFASA: Fast and Accurate Multivariate Time-Series Clustering*. Haojun Li; John Paparrizos. | Proceedings of ACM on Management of Data4(3),SIGMOD,article213; 2026 | [10.1145/3802090](https://paparrizos.org/papers/LiSIGMOD26.pdf) |

The Lloyd paper is also available as an [IEEE article PDF hosted by Stanford](https://web.stanford.edu/class/ee398a/handouts/papers/Lloyd%20-%20Least%20Squares%20Q%20in%20PCM.pdf). The [SIGMOD 2015 proceedings contents](https://www.sigmod2015.org/toc_sigmod.shtml) confirm k-Shape's conference identity. FCACC's online publication is December 2025, with the final volume dated 2026; FASA is the univariate method in the MUFASA publication, not an independently titled paper.

| Method | Source owner / provenance | Selected tag/version | Frozen source commit | License |
| --- | --- | --- | --- | --- |
| Euclidean k-means | [scikit-learn](https://github.com/scikit-learn/scikit-learn); THIRD-PARTY | 1.4.1.post1 | 719c0c6036ac99d45ffb45ed7c8b4e4b221384b5 | BSD-3-Clause |
| k-Shape | [TheDatumOrg](https://github.com/TheDatumOrg/kshape-python); AUTHOR-MAINTAINED | package1.0.6; no corresponding Git tag | 778e735624d15848556e4d23fe38c321d453937a | MIT |
| CKM | none identified; NO IMPLEMENTATION FOUND | none established | null; no source selected | NOT ESTABLISHED |
| DTCC-2023 | [07zy](https://github.com/07zy/DTCC); OFFICIAL | none established | 8531376c0d4efcbcf7a16fda881b219b3d306397 | NOT STATED |
| CDCC | [JiacLuo](https://github.com/JiacLuo/CDCC); OFFICIAL | none established | 013865ff4d13f08d0f5dcdf5bcfba0f5e009c8a0 | NOT STATED |
| FCACC | [Du-Team](https://github.com/Du-Team/FCACC); OFFICIAL | none established | 78b5e5a138ed8c83fea64e5b6668a5cded79d2dc | NOT STATED |
| TFMCC | [Du-Team](https://github.com/Du-Team/TFMCC); OFFICIAL | none established | 882764f52b9efa1ec0950fb90369ed113e988116 | NOT STATED |
| TS2Vec + k-means | [zhihanyue](https://github.com/zhihanyue/ts2vec); OFFICIAL | none established | b0088e14a99706c05451316dc6db8d3da9351163 | MIT |
| R-Clustering | [jorgemarcoes](https://github.com/jorgemarcoes/R-Clustering); OFFICIAL | submission | 3ed571eebe9eb8d399a0c995a6917f62d838f0fc | GPL-3.0 |
| KASBA | [aeon-toolkit](https://github.com/aeon-toolkit/aeon); OFFICIAL | v1.1.0 | 0fed29f21908cfad2981ea645f51aaf018f0fe89 | BSD-3-Clause |
| FASA | [TheDatumOrg](https://github.com/TheDatumOrg/MUFASA); OFFICIAL | none established | 9c05d415efee81fca1a87c1e91623db613ea2ecc | NOT STATED |

No license was found at the root of DTCC, CDCC, FCACC, TFMCC or MUFASA. Record NOT STATED, retain clones privately outside the repository, and resolve distribution permissions before copying/releasing their source. Do not infer a code license from the article's license or a bundled component. R-Clustering's GPL source also stays external; future wrappers must preserve attribution and applicable license terms. License status does not establish algorithm correctness.

## Paper-Code Correspondence

| Method | Static correspondence | Entry / loader / extraction / selection evidence |
| --- | --- | --- |
| Euclidean k-means | HIGH CONFIDENCE MATCH | sklearn/cluster/_kmeans.py; fit X, labels_; label-free inertia restart selection. Canonical algorithm/library match, not a claim of original 1982 experimental code. |
| k-Shape | HIGH CONFIDENCE MATCH | kshape/core.py CPU class; SBD/eigenvector refinement, random partition, assignment stop; no label selection. |
| CKM | NOT ESTABLISHED | Official source unavailable; paper objective established, executable recipe incomplete. Local models/CKM is not selected. |
| DTCC-2023 | PARTIAL / INCOMPLETE | dtcc.py and utils.py contain the intended recurrent/contrastive components, but cluster-contrast gradient and dependency declaration are inconsistent; native output selects best label metrics. |
| CDCC | MISMATCH | main.py, config/CDCC.yaml, algorithm/CDCC/{model,lstm,augmentations}.py: time/frequency losses present; univariate temporal axis and augmentation selectors contradict intended behavior. |
| FCACC | PROBABLE MATCH | batch_run.py, fcacc.py, models/{encoder,Metrics}.py: fuzzy/contrastive stages present; fixed last epoch; environment and exact paper-wide training recipe unpinned. |
| TFMCC | PROBABLE MATCH | TFMCC/{batch_run_single,train,tfmcc}.py and models/encoder.py: wavelet/time views and three-level losses; final head predictions. Paper Adam versus code AdamW remains disclosed. |
| TS2Vec + k-means | HIGH CONFIDENCE MATCH for encoder; new pipeline declared | ts2vec.py, train.py, tasks/classification.py: official encoder/full_series extraction; fixed downstream head is a benchmark task adapter. |
| R-Clustering | PROBABLE MATCH | R_Clustering_on_UCR_Archive.ipynb cells 2/4/10 definitions, cell 12 pipeline; requested 500 becomes 420 features; zero-PC edge; no safe executable module provided. |
| KASBA | HIGH CONFIDENCE MATCH | aeon/clustering/_kasba.py, averaging/_kasba_average.py, distances/elastic/_msm.py; publication explicitly identifies aeon release 1.1.0. |
| FASA | HIGH CONFIDENCE MATCH | Clustering/Running_baseline_iter_univariate.py selects FASA_I_I for -a FASA; FFT alignment and fast aligned centroid update, not multivariate MUFASA. |

These findings distinguish **implementation bugs** (e.g. CDCC integer indexing), **manuscript/code inconsistencies** (e.g. TFMCC optimizer, R-Clustering feature count), **methodological changes** (e.g. replacing a recurrent encoder or contrastive batch), and **optional refactoring** (none performed). A static dependency defect is not an observed runtime failure: no method was executed.

## Native Dataset Protocols

All benchmark fits are transductive pooled clustering of the canonical TRAIN-then-TEST records, with per-series population z-normalization and known k. This is a declared common protocol, not a reproduction of each paper's original aggregate. The [frozen panel](../../experiments_new/loster2026/configs/final/benchmark.json) and registry contain all 40/44/8/36 names. No score-informed dataset substitution is allowed.

| Method | Native cohort / split | Native input transformation and missingness | Common-protocol consequence |
| --- | --- | --- | --- |
| Euclidean k-means | Generic algorithm, no UCR cohort | Finite numeric vectors; no built-in time-series normalization | Same normalized series as neural methods; no augmentation. |
| k-Shape | Original 48-task UCR experiment, not later 85-task archive | Fused TRAIN/TEST, z-normalized; intrinsic centroid z-score | All44 are format-compatible; no source missing-value helper used to repair inputs. |
| CKM | MNIST, USPS, 20 Newsgroups | Image/text inputs, text tf-idf; no UCR recipe | UCR dense input adapter must be explicit; architecture/training recovery still blocked. |
| DTCC-2023 | Ten named UCR tasks | Released pooled np.loadtxt; no normalization in loader; no established missingness policy | Canonical loader needed, but not sufficient to fix training/selection defects. |
| CDCC | Exactly CORE-40 | Pooled global scalar mean/std in released loader; FFT magnitude and native views | Replace outer normalization with common per-series input; preserve intrinsic views only after correspondence resolution. |
| FCACC | A different 40-task subset | Pooled batch runner; selected datasets split-wise scalar normalization; NaN center/trim branch | Common normalization bypass is declared. Keep all IDs; never trim samples silently. |
| TFMCC | Another 40-task subset | Pooled; selected tasks TRAIN scalar statistics applied to TEST; crop/wavelet transforms | Common normalized input; retain native time/frequency views and full-series predictions. |
| TS2Vec | Paper classification 125 UCR/29 UEA; loader supports 128/30 | TRAIN-only SSL, TEST supervised evaluation; selected TRAIN-global normalization; NaN masks | Pooled SSL is an explicit clustering task adaptation; supervised head/grid excluded. |
| R-Clustering | 117 fixed-length tasks, historical 41-task development set | Pooled aeon load, feature standardization/PCA; Dodger/Melbourne exceptions | Canonical loader replaces native special row filtering; feature scaling/PCA stay intrinsic. |
| KASBA | 112 finite/equal-length tasks | Main TRAIN-fit/TEST-predict plus pooled comparison; z-normalization | Common pooled track corresponds to its published pooled setting, not its held-out table. |
| FASA | 128 UCR tasks; 30 UEA is MUFASA | Pooled processed HDF5; z-normalization and archive-wide resampling/interpolation | Finite fixed44 require format/axis adapter, not new imputation or resampling. |

Exact overlap was recomputed from the final CDCC/FCACC/TFMCC tables and TFMCC's released union_list, using dataset names rather than paper averages:

| Method | Native tasks | Intersection with CORE-40 | Intersection with EXTENDED-44 |
| --- | --- | --- | --- |
| CDCC | 40 | **40** | **40** |
| FCACC | 40 | **19** | **19** |
| TFMCC | 40 | **18** | **18** |

CDCC aliases were expanded explicitly: DSR=DiatomSizeReduction; DPOAG=DistalPhalanxOutlineAgeGroup; DPTW=DistalPhalanxTW; EOGVS=EOGVerticalSignal; GPMVF=GunPointMaleVersusFemale; GPOVY=GunPointOldVersusYoung; Large.Kit.App=LargeKitchenAppliances; MPOAG=MiddlePhalanxOutlineAgeGroup; MPTW=MiddlePhalanxTW; PAwP=PigAirwayPressure; PAP=PigArtPressure; PPTW=ProximalPhalanxTW; SHMC2=SemgHandMovementCh2; S.C.=SyntheticControl; TS1=ToeSegmentation1; WS=WordSynonyms. Thus Audit 04's CDCC-aligned CORE-40 is reproducible; it is not a shared 40-task subset of all three modern papers.

FCACC's intersection is ACSF1, Car, CricketX, CricketZ, DistalPhalanxTW, ECGFiveDays, Fungi, GunPointOldVersusYoung, HouseTwenty, LargeKitchenAppliances, MiddlePhalanxOutlineAgeGroup, PigAirwayPressure, PigCVP, Plane, SemgHandMovementCh2, ShapeletSim, ShapesAll, SyntheticControl, ToeSegmentation1.

TFMCC's intersection is Beef, CBF, Car, CricketX, CricketY, CricketZ, ECGFiveDays, FaceFour, FiftyWords, GunPointOldVersusYoung, MiddlePhalanxOutlineAgeGroup, OSULeaf, ProximalPhalanxTW, ShapeletSim, ShapesAll, SwedishLeaf, SyntheticControl, WordSynonyms. None of the four extensions occurs in these three native lists. The registry stores each exact native 40 and absent-from-44 list. Published subset averages cannot populate matched rankings, uncertainty or timing ratios.

R-Clustering's native eleven exclusions are PLAID, AllGestureWiimoteX/Y/Z, GestureMidAirD1/D2/D3, GesturePebbleZ1/Z2, PickupGestureWiimoteZ and ShakeGestureWiimoteZ (notebook cell7). None intersects EXTENDED-44. The exact original k-Shape48 list, KASBA112 manifest and TS2Vec125 classification inclusion list were not independently reconstructed here; their stated cohort sizes/eligibility are primary-source evidence, not verified array-level native manifests. This does not alter the exact common44 manifest or permit importing their unmatched averages.

## Label-Use Audit

**Unsupervised training with known k** permits only the oracle class count, not membership targets. **Label-informed model selection** includes choosing architecture, hyperparameters, epochs or retries by target ARI/NMI/RI/ACC even when the loss is unsupervised.

| Method | Target membership in fit loss? | Native selection / label risk | Required benchmark rule |
| --- | --- | --- | --- |
| Euclidean / k-Shape / KASBA / FASA | No | k and metrics only; inertia or assignment/convergence stop | Fit receives X,k,seed; labels only central evaluation. |
| CKM | Paper objective unsupervised | Code/checkpoint/search behavior not established | Block until recipe/source established. |
| DTCC | No in inspected differentiable loss | **Separate best-NMI and best-RI epoch outputs** | Fixed published-compatible last/intrinsic stop policy; no label-best table. Source repair gate first. |
| CDCC | No | Paper best-parameter search. Code tracks best RI, but model-save line is commented and final predict uses last epoch | Do not incorrectly claim the released final prediction selects best RI; source mismatch still blocks. |
| FCACC | No; model-generated fuzzy assignments are not true labels | Per-epoch label diagnostics; loss scheduler; no active label-best checkpoint found. Paper lr/batch/dropout search | Released single global recipe; diagnostic metrics cannot select anything. |
| TFMCC | No | Periodic label diagnostics; fixed final averaged network, no active label-best branch found | Fixed epoch/global configuration before Final36; no highest-metric epoch. |
| TS2Vec + k-means | No SSL labels | Native supervised classification head uses training labels/search | Exclude supervised head entirely; fixed unsupervised head. |
| R-Clustering | No | Historical 41-task development search selected kernel configuration using clustering quality | Retain published global defaults, disclose historical label-informed design; no new Final36 search. |

The proposed baseline fit/export process receives canonical X, IDs, k, configuration and seeds. Ground-truth arrays are held by the independent evaluator. Where author APIs carry labels merely for diagnostics, adapters must demonstrate that excluding metric calculations preserves optimizer flow, model modes and RNG consumption. Removing a diagnostic forward can change training if it consumes randomness or switches eval/train mode; preserve its unlabeled computation when needed and test parity. Do not disable label use by secretly changing an active label-dependent training/stopping branch. Such a method is INCOMPATIBLE PROTOCOL until a separately justified adaptation is frozen.

## Metric and Prediction Contract

Use the committed [LoSTer 2026 metrics module](../../experiments_new/loster2026/src/loster2026/metrics.py), independent of the baseline optimizer:

- **ARI:** sklearn adjusted_rand_score; chance-corrected pair agreement with its declared degenerate-case conventions.
- **Arithmetic NMI:** normalized_mutual_info_score with explicit average_method='arithmetic', equivalently 2 I(Y;C)/(H(Y)+H(C)) in nondegenerate cases.
- **RI:** (TP+TN)/choose(N,2), counted from integer contingency totals without an N-by-N pair matrix; N=1 returns 1.
- **ACC:** maximum one-to-one matched contingency count divided by N via linear_sum_assignment(-counts). Remap arbitrary true/predicted labels with unique values; use a rectangular contingency. This is not many-to-one purity or raw equality of label integers.

| Method | Native ARI | Native NMI | Native RI | Native ACC / caution |
| --- | --- | --- | --- | --- |
| Euclidean / k-Shape | No estimator metrics | No estimator metrics | k-Shape paper RI; core has no runner | Central only. |
| CKM | Paper yes | Paper yes, normalizer unresolved | Not established | Paper calls ACC cluster purity; Hungarian equivalence not established. |
| DTCC | sklearn | average_method omitted, dependency unpinned | Pair-count implementation | Hungarian via removed sklearn helper; label assumptions. Best NMI/RI can be different epochs. |
| CDCC | Not active runner | main.py calls sklearn without argument; pinned 1.0.2 default arithmetic | sklearn rand_score | Not active runner. |
| FCACC | sklearn import | sklearn alias, argument omitted; freeze runtime convention | comb/bincount pair counts | Hungarian after subtracting min label; assumes contiguous labels and square k table. |
| TFMCC | sklearn | argument omitted; verify runtime | comb/bincount pair counts | Hungarian contingency then accuracy_score; central direct matched-count definition avoids unmatched-label ambiguity. |
| TS2Vec | Native classification is not clustering ARI | No native pipeline clustering metric | No native pipeline RI | Supervised classification accuracy is not Hungarian ACC. |
| R-Clustering | sklearn adjusted_rand_score | Not notebook output | Not notebook output | Not notebook output. |
| KASBA | Paper yes; estimator none | Paper information metric; exact experimental normalizer not recovered from estimator | Paper yes | Paper CLACC defines permutation matching; estimator has no metrics. |
| FASA | sklearn adjusted_rand_score | average_method omitted; dependencies unpinned | sklearn rand_score | Not native runner. |

Native metrics are diagnostic only. Every registered method must export one prediction per canonical sample; no primary-table native-metric exception is approved. If export cannot be made faithful, record INCOMPATIBLE PROTOCOL, rather than merging unmatched native scalar scores.

Prediction CSV/Parquet columns: sample_id, predicted_cluster, seed, dataset, method, run_id. Sample IDs are **dataset/TRAIN/zero_based_row** and **dataset/TEST/zero_based_row** from headerless original records. CSV seed is the root seed, not an arbitrary runner iteration label. Cluster IDs are finite integers; label spelling and cluster numbering need not match the ground truth.

The validator joins by ID, requires the exact expected ID set and N, rejects duplicates/missing/extra rows/nonfinite or noninteger predictions, checks dataset/method/seed/run_id against the manifest, then restores TRAIN-then-TEST order before evaluation. A natural empty cluster or fewer occupied centers is a scientific outcome, not an export failure and not grounds to force all k clusters occupied. ID-bearing predictions must be saved before labels are opened.

The run manifest must contain: cohort/split/ID hash; raw and normalized X hashes; N,L,k,dtype; root and phase seeds; exact source/tag, adapter/config Git SHA and clean/dirty status; parameter JSON/hash; environment lock/hash, Python/framework/CUDA/driver versions; CPU/GPU/device/thread policy; actual epochs/iterations, stop reason, peak memory and stage timings; active model/SWA/centroid state and inference mode; output/checkpoint/representation hashes; timestamp and failure status. Store fresh results outside legacy outputs, indexed by this complete identity. Cache reuse requires every scientific identity match; paper means or old untraceable scalar scores are not reusable runs.

Metric unit/regression checks from Report 07 are already available (58 tests in the frozen harness). They do **not** validate the new baseline ID join, shuffled export, checkpoint reload, timing or statistical pipeline. Those integration smokes remain gates; tests for label permutation, singleton/one-cluster cases, rectangular ACC, native/common disagreement and invalid IDs are necessary.

## Seeds and Determinism

All eleven selected methods are **STOCHASTIC**, including classical methods: Euclidean/k-Shape/FASA initialize randomly; KASBA also randomizes barycenter subsets; R-Clustering has random kernel/bias calibration and clustering. Fixed seeds can make repeatable runs, but do not turn a stochastic method into a deterministic algorithm. CKM is stochastic by its paper even though its executable seed plumbing is unknown. No deterministic method is being artificially duplicated five times.

| Method | Published outer repetitions | Random sources / seed adaptation |
| --- | --- | --- |
| Euclidean | No native UCR repeat protocol | random_state; 10 internal minimum-inertia initializations are part of one benchmark fit. |
| k-Shape | 10 in original paper | NumPy initial partition/empty recovery; seed before fit. |
| CKM | 15 | Gumbel, initialization, stochastic optimization; exact source hook unknown. |
| DTCC | Not established | TF graph/initializers, NumPy noise/jitter/shuffle, downstream KMeans; no comprehensive hook. |
| CDCC | Not established; default helper seed2333 | Python/NumPy/Torch/CUDA; recurrent h/c randomness persists during eval. |
| FCACC | Not established; runner seed1127 | Python/NumPy/Torch, crop/mask/dropout/shuffle; published source center-init seed0 retained. |
| TFMCC | Not established | Python/NumPy/Torch, crop/wavelet level/scaling/shuffle/GPU; seed hook must cover all. |
| TS2Vec | Native clustering repetitions not applicable | Encoder/crop/mask/shuffle plus independently seeded downstream head. |
| R-Clustering | 10 paper outer runs; notebook single fit | Host NumPy, **Numba compiled RNG**, possible randomized PCA, KMeans. |
| KASBA | Pooled k-Shape comparison 10; main count not fully established | estimator random_state controls elastic initialization and stochastic barycenters. |
| FASA | 10 paper runs | NumPy initial assignments/empty recovery; native itr names files, it does not seed RNG. |

Final root seeds are **100,101,102,103,104**. Record Python/NumPy/framework/CUDA seeds and RNG state. Derive independently needed phase seeds by the first eight hexadecimal digits of SHA256 of UTF-8 **loster2026-baseline-v1|method_id|dataset|root_seed|phase**, interpreted as uint32; use declared phases such as kmeans, pca and inference. Do not use Python's process-randomized hash(). Preserve method-specific fixed constants such as FCACC's center seed0 and disclose them, rather than silently randomizing a fixed paper recipe.

A Numba compiled seed bridge is required for R-Clustering; numpy.random.seed in ordinary Python does not seed an already compiled Numba generator. Repeatability tests must cover fitted biases and predictions, not only host random draws. Set PYTHONHASHSEED before interpreter startup when needed. GPU kernel nondeterminism remains possible for the six deep methods; freeze framework determinism/precision settings and record unsupported operations rather than promise bitwise cross-device equality.

If a later approved comparator proves deterministic, use one fit per task plus a same-input repeatability check; record run_count=1 and no invented seed SD. Outer five fits, internal restart counts, and five warmed inference passes are three different quantities.

## Hyperparameter Policy

The complete per-method inventory/classification is machine-readable. Every parameter falls into METHOD-INTRINSIC, GLOBAL DEFAULT, DATASET-DEPENDENT BUT LABEL-FREE, LABEL-TUNED or UNCLEAR. Input length, known k, dilation grids, label-free PCA dimensions and native min(B,N) rules are legitimate dataset dependence. A hardcoded per-dataset exception is not automatically label-free: record its basis, and bypass outer archive normalization exceptions for the common input.

| Method | Frozen candidate global recipe / stopping | Policy |
| --- | --- | --- |
| Euclidean | k-means++, n_init10, Lloyd, max300, tol1e-4 | USE PUBLISHED DEFAULTS/library algorithm with explicit benchmark restart convention. |
| k-Shape | CPU zero-centroid/random-partition, max100, stable assignment | USE PUBLISHED DEFAULTS. |
| CKM | Published dense architecture/objective known; optimizer/anneal/init recipe incomplete | NOT REPRODUCIBLE FAIRLY at present. |
| DTCC | Source units100/50/50, dilation1/4/16, lr.005, epochs200, B=N, lambda.001, SVD interval10 | NOT REPRODUCIBLE FAIRLY as released; preprint differs and cluster-contrast gradient discrepancy is unresolved. |
| CDCC | Runner epochs300, B256, lr.01, layers3/dropout.5, hidden1024/output512/feature256; temps.5/1 | NOT REPRODUCIBLE FAIRLY as released. Constructor defaults differ from runner; do not mix them. |
| FCACC | B8, representation64, hidden64, depth10, pretrain100 at .001, joint30 at hardcoded1e-4; m1.5, w_c/hard_w.2 | USE PUBLISHED DEFAULTS as released single global recipe, conditional on environment/persistence/parity. |
| TFMCC | B256, representation320, hidden64/depth10, epochs1400, lr.001, pooled length512/head256; batch-driver temps.2/.2/.5 | DEVELOPMENT CALIBRATION REQUIRED for valid environment/resource batch; preserve architecture/losses. |
| TS2Vec + k-means | CLI B8, representation320, hidden64/depth10, AdamW.001; default200/600 iterations by input.size; full_series; fixed Euclidean head | USE PUBLISHED DEFAULTS for encoder plus predeclared head. |
| R-Clustering | requested500/actual420 features, length9, max dilation32, variance<.01 PC rule, KMeans n_init10 | USE PUBLISHED DEFAULTS as released; disclose rounding/zero-PC edge, no fresh kernel search. |
| KASBA | MSM c1, max300, tol1e-6, BA subset.5, step.05, decay.1 | USE PUBLISHED DEFAULTS; do not copy another paper's comparator cap100. |
| FASA | zero initial centers/random assignments, max100/stable assignments | USE PUBLISHED DEFAULTS. |

CDCC's paper searched lr/layers/batch/dropout and reported best parameters; FCACC searched lr/batch/dropout. Complete candidate lists and criteria are not recoverable here. Do not invent search spaces or reproduce per-Final36 best choices. R-Clustering's paper gives a historical 20-configuration search (feature counts 100..20000, kernel lengths 7..13) on 41 development tasks; the selected global values are usable published defaults with that history disclosed.

Only if genuinely needed, Audit 04 permits a ceiling of **3 global configurations x Development8 x seeds0,1,2 = 72 fits per baseline**. This is not an automatic grant to perform those fits now. Resource/import debugging need not become an accuracy search. For TFMCC, first recover a valid environment and assess native B256 without target metrics; a later proposed resource ladder B256/B64/B16 can fit within three global configurations, but is **not frozen or authorized by this audit**. Choose one global valid recipe before any Final36 run; no dataset-specific rescue batch or architecture changes. Author source replacement for the three blocked methods reopens their source/reproduction gate, not LoSTer's frozen definition.

## Method Adaptation Rules

**NON-SCIENTIFIC ADAPTER:** headerless canonical loader, shape/axis conversion, paths/cwd, known-k injection, faithful RNG hooks, fixed metric/export interfaces, checkpoint persistence of the active state, timing/provenance and safe Windows entry guards. Show that predictions, optimizer flow, modes and RNG consumption are preserved on a synthetic/Development8 parity test.

The common pooled/per-series-normalized task is a **declared benchmark protocol adaptation** where it differs from native split/normalization. It is not an unreported reproduction of the native paper table. Keep method-intrinsic transformations (FFT magnitude, feature StandardScaler/PCA, wavelet views, k-Shape centroid normalization). Non-deep comparators receive no LoSTer augmentation.

**SCIENTIFIC METHOD CHANGE:** changing losses, Concrete/ST gradients, centroid updates, recurrent time axes, view distributions, learned dimensions, wavelet/head structure, spectral update logic or choosing labels to stop. Correcting CDCC's obvious indexing bug changes the effective augmentation distribution; identifying a bug does not authorize its scientific repair. Changing contrastive batch changes negatives/BatchNorm and is a material global configuration decision even if exposed by CLI. Gradient accumulation generally does not reproduce the original denominator or BatchNorm behavior.

Substantial repairs, ports or common-backbone substitutions are **NOT A FAITHFUL REPRODUCTION** until a separately reviewed, equation-traceable implementation decision/source artifact is available. Do not quietly relabel them as official baselines. Minor API replacements can be adapters only after semantic/RNG parity is demonstrated. This task implements none of them.

## CKM

The competing method is [Gao et al.'s final ICASSP paper](https://www.pure.ed.ac.uk/ws/files/134695608/Deep_Clustering_with_Concrete_GAO_DOA24012020_AFV.pdf), not the local historical CKM adapter. Its reconstruction-plus-centroid-distortion objective jointly learns the autoencoder and centers. A radial-basis assignment distribution uses squared latent-center distance; Gumbel/Concrete supplies a hard one-hot forward assignment and relaxed straight-through backward gradient, with temperature annealing. The paper's image encoder is **500-500-2000-10**, mirrored by the decoder; its text encoder is **250-100-20**. It reports 15 repetitions on image/text datasets. Its cluster-purity terminology does not establish Hungarian ACC.

Algorithm/equation evidence is sufficient to identify the scientific challenge, but not a complete frozen executable recipe: exact optimizer, batch/epoch budgets, RBF scale, temperature schedule, center initializer and inference implementation remain unestablished. The algorithm accepts centers after pretraining without a fully recoverable initialization recipe. UCR was not the native evaluation. Do not fill these gaps with LoSTer hyperparameters.

**Recommendation: C conditionally, with A in the main table and B only as a separate ablation.** A must represent actual published CKM: recover official code or obtain an explicitly reviewed paper-faithful reconstruction, preserve the dense architecture/objective/training recipe, and disclose input-width L adaptation. B, a CKM objective in the LoSTer/common dense backbone, tests assignment lineage under matched capacity; it is not the published competing method. No B experiment is authorized here, and no code has been changed.

Historical local models/CKM uses a different dense width (1000 rather than published 2000), inherited loader/protocol choices and no recovered author-source provenance. It cannot become A by renaming it. Current readiness **BLOCKED / NOT REPRODUCIBLE FAIRLY** means an unresolved benchmark slot, not removal of the Concrete challenge. Nearest-center inference may be a sensible proposed reconstruction, but is not treated as recovered author behavior.

## DTCC

The selected method is **Zhong, Huang and Wang, Neural Processing Letters 2023**, linked by its [publisher page](https://link.springer.com/article/10.1007/s11063-023-11287-0) to [07zy/DTCC at the frozen commit](https://github.com/07zy/DTCC/tree/8531376c0d4efcbcf7a16fda881b219b3d306397). It is not the old modified DTC or the local PyTorch DTCC variant.

The [author preprint](https://arxiv.org/html/2212.14366) specifies bidirectional dilated recurrent encoders, a recurrent decoder, reconstruction/spectral cluster-distribution and instance/cluster contrast. Its ten datasets are Beef, DistalPhalanxOutlineAgeGroup, ECG200, ECGFiveDays, Meat, MoteStrain, OSULeaf, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW. The released loader pools TRAIN/TEST. Exact parameter equivalence to the final published version is unresolved: the preprint says B=N/2 and SVD every5, whereas the released runner uses B=N and interval10.

Static source findings:

- **Environment inconsistency:** README declares tensorflow-gpu2.4.0; source uses TF1 Session/placeholder/tf.contrib GRU/legacy_seq2seq and private recurrent APIs. compat.v1 alone does not restore tf.contrib.
- **Gradient correspondence:** dtcc.py lines143-151 declare F/F_aug nontrainable; lines266/274 assign placeholder-fed SVD values; line289 computes cluster_loss only from those assignment outputs. The declared optimizer at line342 minimizes loss+cluster_loss, but this cluster term has no dependency on trainable encoder parameters. Its periodic NumPy SVD update is outside autodifferentiation. Thus the released cluster-level term does not supply the claimed encoder gradient. This is a source/method correspondence concern, not a repair authorization.
- **Feed distinction:** lines377-382 omit F_new_value/F_aug_new during the ordinary gradient step; the periodic cluster-loss value fetch at lines410-412 supplies them. Since autodifferentiation can prune the disconnected term, a missing-placeholder runtime failure at epoch0 is **not established**. No TensorFlow runtime was used. Do not conflate this static observation with FAILED REPRODUCTION.
- **Label selection:** lines458-467 return separate best-NMI and best-RI epochs. Those scalars are not a single fixed model's matched predictions. Fixed last/intrinsic-stop extraction, explicit head seeds/restarts, and ordered fresh inference are needed after correspondence is resolved.
- **Resources:** full-pool recurrent reconstruction, N-by-N spectral/instance matrices and fixed-N cluster indicators give VERY HIGH GTX1080 risk. Smaller batch changes the computation; accumulation is not an equivalent spectral/contrastive batch.

Released candidate globals are units100/50/50, dilations1/4/16, latent400, lr.005, epochs200, jitter.03, denoising.1 and runner lambda.001. The code's output folder containing wo_kmeans_aug does not by itself prove which ablation was executed; inspect the active loss rather than the path name.

**BLOCKED, PARTIAL / INCOMPLETE, NOT REPRODUCIBLE FAIRLY as released.** A historical TF1.15/Python3.7/CUDA10 environment is a hypothesis to investigate, not a verified recipe or environment installation. Restoring gradient coupling or porting to PyTorch would require an explicit methodological implementation decision.

Old modified DTC and historical DTCR/local DTCC scores become **SECONDARY/HISTORICAL or ABLATION/CONTEXT**, preserving attribution/provenance. They do not enter the new main table, are not rerun automatically, and cannot stand in for the 2023 method. Audit 02's missing historical prediction/cohort provenance and parser issue still apply.

## CDCC

The [final AAAI 2024 paper](https://ojs.aaai.org/index.php/AAAI/article/view/28740) points to [JiacLuo/CDCC](https://github.com/JiacLuo/CDCC/tree/013865ff4d13f08d0f5dcdf5bcfba0f5e009c8a0). Time/frequency views and within-/cross-domain instance and cluster contrast are present. The source runner uses 300 epochs; its time branch is a BiLSTM and frequency branch a CNN. Published best-parameter search is distinct from unsupervised losses and cannot be repeated on Final36.

Three concrete source facts block a faithful main-table wrapper:

1. In [augmentations.py lines59-75](https://github.com/JiacLuo/CDCC/blob/013865ff4d13f08d0f5dcdf5bcfba0f5e009c8a0/algorithm/CDCC/augmentations.py#L59), expressions such as aug_1[1-li_onehot[:,0]]=0 use integer advanced indices containing 0/1, rather than a Boolean per-row choice mask. They repeatedly zero rows0/1; other rows combine views differently from the intended selector. Repair changes the actual training view distribution.
2. In [lstm.py lines13-29](https://github.com/JiacLuo/CDCC/blob/013865ff4d13f08d0f5dcdf5bcfba0f5e009c8a0/algorithm/CDCC/lstm.py#L13), univariate X shaped N,1,L becomes 1,N,L for the default sequence-first LSTM, with input_size=L. Recurrent sequence length is **one**, not L timestamps. Converting to an L-step recurrent encoder changes model behavior and architecture interpretation.
3. h_0/c_0 are fresh torch.randn values on every forward, including eval. Zeroing them would change behavior; a wrapper cannot silently do so for reproducibility.

The runner also uses B256/drop_last, yielding no complete batch on Beef(N60) or Fungi(N204). A globally smaller batch could solve this compatibility issue only after the larger correspondence gate is resolved. Do not treat it as the sole bug. The output-channel dictionary is mostly a length/shape rule, not evidence of target-label tuning.

True labels do not enter the inspected loss. Best RI is tracked, but the save operation is commented; the final prediction uses the last epoch. Native RI is sklearn rand_score and NMI is normalized_mutual_info_score with no argument; the pinned sklearn1.0.2 default is arithmetic. These facts do not overcome the released algorithm mismatch.

**MISMATCH / BLOCKED.** Seek an author-corrected, pinned artifact or explicitly approve and name a paper-faithful reconstruction in a future stage. Running the current released-code variant could be informative separately, but cannot silently be called faithful CDCC or be used to tune against LoSTer.

## FCACC

The [final FCACC paper](https://dumingjing.github.io/files/paper-21_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series/2026_PR_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series.pdf) and [Du-Team source](https://github.com/Du-Team/FCACC/tree/78b5e5a138ed8c83fea64e5b6668a5cded79d2dc) establish fuzzy cluster-aware contrast, overlapping time crops and scaled third views. Its reported UCR40 intersects CORE-40 on19 tasks. Native labels are used for k/diagnostics; fuzzy memberships and pseudo-labels are computed from the model, not ground-truth membership.

The released batch runner's concrete global recipe is B8, representation64, hidden64/depth10, pretrain100 at AdamW.001, joint30 at hardcoded1e-4, m1.5, contrast weight.2, hard weight.2, confidence.95, extraction gamma.5, scale.8 and center-init KMeans seed0. The class representation default32 is overridden by runner64. Do not combine class and CLI defaults arbitrarily. A convergence figure with100+100 epochs is not evidence that every published experiment used200; common-protocol use of released100+30 must be disclosed as such. Loss-based ReduceLROnPlateau is permitted; highest label metric is not the inspected checkpoint policy.

Two export/persistence adapters are required. eval_with_test_data iterates a shuffled loader and ignores its existing row indices; native predictions and labels stay mutually aligned, but canonical order is lost. Scatter predictions by those indices before independent evaluation. The deployed encoder uses an averaged/SWA state; preserve that active state plus centers, configuration and inference state for reload parity, rather than saving only the raw encoder.

A native unused N-by-L-by-d float64 representation allocation can inflate CPU RAM; omitting/streaming it is a candidate non-scientific optimization only after checking extra forward/RNG/mode semantics. The fuzzy loop recomputes representations by cluster, making k60 tasks a time risk. Native NaN trim can desynchronize indices/labels; do not invoke it to drop benchmark records.

**PROBABLE MATCH / READY WITH CONDITIONS / USE PUBLISHED DEFAULTS.** Exact author framework/numeric pins are absent; build an isolated candidate and verify API, n_init, SWA reload, mode/RNG and ID parity on Development8. This does not authorize a paper-search recreation or an improved loss/update implementation.

## TFMCC

The [AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/39817) and [Du-Team source](https://github.com/Du-Team/TFMCC/tree/882764f52b9efa1ec0950fb90369ed113e988116) provide time/frequency augmentation and temporal-, instance- and cluster-level contrast. The released union_list exactly resolves the paper's40 aliases; overlap with CORE-40 is18, not40. Prediction is the learned head's argmax; the old KMeans evaluator branch is not the selected method.

The batch driver supplies temperatures temporal.2, instance.2, cluster.5, matching the paper, while CLI defaults differ. Candidate defaults: 1400 epochs, lr.001, B256, representation320, hidden64/depth10, max_train_length3000; native effective B=min(B,N). The encoder's adaptive pool produces length512 and flattens 320x512 into a 256-unit head: its first dense matrix alone has **41,943,040 weights**. Raw and averaged networks, gradients, optimizer state, crop activations and quadratic temporal/instance similarities compound memory demand. The article's 5090D32GB machine does not establish GTX1080 feasibility.

Frequency augmentation uses db4 decomposition, randomly chosen levels up to12, coefficient scaling and source boundary/reconstruction handling. Preserve those transformations; do not replace with LoSTer warp or generic FFT noise. The code uses AdamW while the paper names Adam. Use the released-code recipe as a declared candidate, but resolve/document this discrepancy before claiming exact paper reproduction; do not silently swap optimizers.

README dependency strings are ambiguous (Python3.9, torch2.8.1, scipy1.6.1, numpy1.26, pandas2.23) and do not provide a demonstrated compatible lock. No package was installed and no PyTorch-wheel GTX support was assumed. Import closure includes pytorch_wavelets/PyWavelets, skfuzzy, NumPy/SciPy/pandas/sklearn and runner export dependencies. Native Windows paths require adapters.

**PROBABLE MATCH / READY WITH CONDITIONS / DEVELOPMENT CALIBRATION REQUIRED.** First establish a valid faithful environment, then make one pre-score global resource configuration decision on Development8. Smaller batches are accepted by the source interface but change contrastive negatives and BatchNorm; accumulation is not mathematically equivalent. Reducing representation/head/pool dimensions is a scientific capacity change, not an OOM fix authorized here. CPU or a newer GPU can preserve the algorithm if the identical frozen recipe runs; report hardware-specific cost separately, not a same-device speed ratio.

## TS2Vec + k-means

Use the [official AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/20881) and [author code](https://github.com/zhihanyue/ts2vec/tree/b0088e14a99706c05451316dc6db8d3da9351163). The whole-series representation is explicitly supported by README and tasks/classification.py: **encode(X, encoding_window='full_series')**, temporal max pooling of the final averaged encoder. No favorable mean/max concatenation, multiscale aggregation, sliding window, supervised SVM selection or benchmark-wide encoder is introduced.

For each dataset/root seed:

1. Fit one official encoder on that dataset's canonical pooled unlabeled X, preserving crops, masking, loss and averaged state. Use CLI B8, output320, lr.001, hidden64/depth10; default label-free iteration budget200 when input.size<=100000, otherwise600. The constructor's B16 default is not the CLI default selected here.
2. Extract one ordered full_series vector per complete original series in eval mode with the official extraction path; do not cluster random crops. Hash/save representations with source/config/seed/input identity.
3. Fit one downstream Euclidean KMeans head: k from metadata, k-means++, Lloyd, n_init10, max300, tol1e-4, independent deterministic phase seed. Ten internal restarts are label-free minimum-inertia choices, not ten encoders or ten reported runs.
4. Export IDs/predictions and save averaged encoder/head state. Reuse exact representations only within the same dataset/encoder/source/config/root seed; another seed requires its own fit.

Native paper classification uses TRAIN SSL and TEST supervised evaluation; this common pooled clustering pipeline is explicitly **TS2Vec + k-means**, not the native classification score. All representation training, full-pool extraction and head initializations/refinement count in pipeline cost.

**HIGH CONFIDENCE encoder match / READY WITH CONDITIONS / USE PUBLISHED DEFAULTS.** Conditions are faithful legacy environment, explicit head/phase seed, ordered export and pipeline timing smoke. Long-crop temporal similarity is still a GTX memory risk even at B8.

## Classical / Non-Deep Baselines

**Euclidean k-means.** Use [scikit-learn1.4.1.post1](https://github.com/scikit-learn/scikit-learn/blob/719c0c6036ac99d45ffb45ed7c8b4e4b221384b5/sklearn/cluster/_kmeans.py) on the exact normalized series matrix N,L supplied to neural methods. Explicit n_init10 prevents the modern auto/k-means++ default of one initialization from silently changing the reference. Restarts select minimum SSE, not maximum ARI. It supports all 44 finite fixed-length inputs and uses CPU only. Assignment/refinement cost is approximately O(R I N k L), where R=10 internal initializations; input/center/working arrays, not a required N-by-N distance matrix, dominate memory.

**k-Shape.** Use the author's [CPU core](https://github.com/TheDatumOrg/kshape-python/blob/778e735624d15848556e4d23fe38c321d453937a/kshape/core.py), input N,L,1. Initial labels are random and centers default to zero; native empty-cluster behavior remains. SBD uses normalized FFT correlation and zero-padded alignment, then a projected L-by-L scatter/eigenvector centroid update. Assignment is about O(N k L log L); centroid matrix/eigendecomposition can add L-squared storage and L-cubed work in this implementation. Long-series cost can be high despite no GPU use. max100/stable-assignment stopping stays unchanged. __init__.py does not force CuPy: use CPU imports; include sklearn explicitly because the source imports it despite setup listing only NumPy. Even n_jobs1 uses a multiprocessing pool, requiring a Windows main guard and thread control. Do not substitute tslearn's solver without declaring a new source choice.

**R-Clustering.** Use the [submission notebook](https://github.com/jorgemarcoes/R-Clustering/blob/3ed571eebe9eb8d399a0c995a6917f62d838f0fc/R_Clustering_on_UCR_Archive.ipynb), selecting algorithm definitions and pipeline logic only (cell indices here are zero-based). Never execute install/download/full-archive experiment cells. The modified MiniRocket pipeline has fixed length9 kernels, random bias calibration, feature StandardScaler, variance-rule PCA and KMeans. A request for500 features rounds down to 84*floor(500/84)=**420**. Preserve/disclose this source behavior; forcing500 or588 is not a loader adapter. The PCA selector argmax(explained_variance_ratio_<.01) can return0 if no component meets the condition or the first does; do not silently clamp to1, switch to explained cumulative variance, or choose dimensions using labels. Preflight this rule on Development8 and classify unsupported cases under the frozen failure policy. Original paper gives10 outer runs; the notebook has one pipeline and internal n_init10. CPU cost includes O(N F L) convolutional work (F actual420), feature scaling/PCA and KMeans; no obligatory N-squared pair matrix. Numba seed, PCA solver/version and notebook extraction must be frozen; current readiness has conditions, not a guarantee all 44 avoid the zero-PC edge.

**KASBA.** Use the publication-linked [aeon release1.1.0 estimator](https://github.com/aeon-toolkit/aeon/blob/0fed29f21908cfad2981ea645f51aaf018f0fe89/aeon/clustering/_kasba.py), not a comparator copy bundled by another paper. MSM distance c1, elastic k-means++ initialization, stochastic barycenter updates and metric triangle pruning are intrinsic. Preserve subset.5/initial step.05/decay.1/max300/tol1e-6 and actual inner defaults; no new restart loop. Known k and random_state are supported, y is ignored. Use N,1,L common pooled input. It is CPU-only; MSM dynamic programs have L-squared pair cost and barycenter updates add work, with pruning improving actual rather than guaranteed worst-case complexity. High long-series runtime/RAM risk requires measurement, not invented minutes. All44 satisfy the planned fixed-length input type, subject to canonical finite-input validation and timeout. Main paper split and pooled comparison remain distinct.

**FASA.** Use [FASA_I_I](https://github.com/TheDatumOrg/MUFASA/blob/9c05d415efee81fca1a87c1e91623db613ea2ecc/Clustering/FASA_I_I/FASA_I_I.py), selected by the univariate -a FASA runner. SBD FFT alignment and aligned mean-like centroid update replace k-Shape's eigenvector step. Keep random assignments/zero centers, empty-cluster handling, native zero-norm conventions and max100/stable labels. A sum-zero centroid can pass through divide/zscore/nan_to_num handling; do not improve it silently. itr is an output naming parameter, not an RNG seed. Cluster member index lists permit complete ordered export. Assignment is approximately O(I N k L log L), with aligned centroid work and input/assignment buffers; no k-Shape L-by-L eigenproblem. All44 are format-compatible; no LoSTer augmentation, resampling or new interpolation is needed for finite fixed-length inputs. Method-only NumPy/SciPy/HDF5/tqdm closure avoids installing unrelated broad baseline requirements.

## Environment Matrix

**Design only.** Environments belong under G:/Articoli/Articoli da Completare/LoSTer 2026/baselines/envs. Exact transitive locks, wheel hashes, CUDA runtime and GPU architecture coverage are not yet verified. The existing LoSTer Python3.9.13/torch1.13.1+cu117/numpy1.22.4/pandas1.5.3/sklearn1.4.1.post1 environment stays untouched.

| Method | Python / framework | CUDA / CPU | Dependencies / obsolete interfaces | Windows and LoSTer conflict | License |
| --- | --- | --- | --- | --- | --- |
| Euclidean | 3.9/3.10 candidate; sklearn1.4.1.post1 | CPU | NumPy/SciPy/threadpoolctl; explicit KMeans arguments | Native wheels; low conflict but separate env | BSD-3-Clause |
| k-Shape | 3.9/3.10 candidate; NumPy core1.0.6 | CPU; no CuPy needed | NumPy + source-imported sklearn | Pool spawn guard; pin BLAS/threads; GPU path not selected | MIT |
| CKM | Unknown | Unknown | Source/recipe absent | Not established; no local-code environment fallback | Unknown |
| DTCC | README>=3.7; TF1 candidate3.7 | TF1.15/CUDA10/cuDNN7 hypothesis; CPU possible in principle | Declared TF2.4 conflicts with tf.contrib/private TF1 APIs; numpy1.19.5/scipy1.7.3, removed sklearn assignment helper | Legacy Windows closure unverified; cannot use LoSTer/PyTorch stack | Not stated |
| CDCC | 3.8 candidate; author torch1.10.2+cu113 | CUDA11.3 candidate /CPU | numpy1.21.6, sklearn1.0.2, scipy1.7.3, pandas1.1.5, YAML6; broad extras | Old wheels; differs from LoSTer; newer NumPy ragged permutation rules can break source | Not stated |
| FCACC | Author unpinned; 3.8/3.9 +torch1.13.1 candidate only | CUDA11.7 candidate /CPU | NumPy/SciPy/pandas/sklearn/matplotlib; SWA and scheduler APIs; freeze n_init semantics | Cwd/path wrapper, API smoke; no verified author lock | Not stated |
| TFMCC | README3.9/torch2.8.1 ambiguous; working pins unresolved | Verify GTX sm_61/wheel/driver, or separate hardware track | pytorch_wavelets/PyWavelets/skfuzzy/numeric stack/openpyxl; malformed README versions | Paths/wheels unresolved; do not assume all modern CUDA wheels support Pascal | Not stated |
| TS2Vec | Author3.8/torch1.8.1 | Author CUDA wheel needs pin /CPU | numpy1.19.2, scipy1.6.1, pandas1.0.1, sklearn.24.2, statsmodels.12.2, Bottleneck1.3.2 | Historical wheels/import smoke; isolate older stack | MIT |
| R-Clustering | Paper3.6 but unpinned modern aeon incompatible; candidate3.10 | CPU | Candidate numpy1.26/scipy1.13/sklearn1.4.1.post1/numba.60/pandas; bypass aeon loader | Paper used Windows; safe module extraction, compiled RNG and parity required | GPL-3.0 |
| KASBA | aeon supports>=3.9,<3.14; candidate3.10/aeon1.1.0 | CPU | numpy>=1.21,<2.3; scipy>=1.9,<1.16; pandas>=2,<2.3; sklearn>=1,<1.7; numba>=.55,<.62 | Native wheels candidate; LoSTer pandas1.5.3 fails aeon constraint | BSD-3-Clause |
| FASA | Author unpinned; candidate3.10/NumPy-SciPy | CPU | Narrow NumPy/SciPy/h5py/tqdm closure; broad requirements_UTS is not needed wholesale | Pure Python/numeric source plausible; smoke required | Not stated |

Candidate versions are **not final frozen environments**. Recover author-compatible recipes where possible; when modernization is necessary, prove semantic/RNG/prediction parity before selecting it. Do not equate an environment that imports with a reproduction that trains and exports the method faithfully. No environment, dependency or CUDA driver was installed/changed in this audit.

## GTX 1080 Feasibility

The inspected device is GTX1080 with 8 GB VRAM; this is a qualitative static assessment, not measured runtime/peak memory. CPU RAM capacity and sustained CPU performance have not been assumed.

| Method | VRAM | CPU RAM | Runtime | Length risk | Batch / feasibility implication |
| --- | --- | --- | --- | --- | --- |
| Euclidean | LOW (CPU) | MODERATE | MODERATE | MODERATE | No neural batch; all 44 format-compatible, actual cost pending. |
| k-Shape | LOW (CPU) | HIGH | HIGH | HIGH | L-by-L centroid scatter/eigen solve and pool copies; CPU long-task issue. |
| CKM | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Cannot infer feasibility without source/complete recipe. |
| DTCC | VERY HIGH | HIGH | VERY HIGH | VERY HIGH | Released B=N and N-squared matrices plus recurrent decoder; unlikely all 44 on8GB. |
| CDCC | HIGH | HIGH | HIGH | HIGH | Wide recurrent/source model and B256; small-N zero-batch bug. Actual source is one recurrent step, so do not estimate it as a corrected L-step LSTM. |
| FCACC | HIGH | HIGH | HIGH | HIGH | Native B8 is valid; dilated activations, k-loop recomputation and unused representation allocation need smoke. |
| TFMCC | VERY HIGH | HIGH | VERY HIGH | VERY HIGH | Large320x512 head/optimizer/SWA and B256 contrast; unlikely all 44 on8GB with native defaults. |
| TS2Vec + k-means | HIGH | MODERATE | HIGH | HIGH | Native B8 mitigates instance batch, not crop-length-squared temporal similarity. |
| R-Clustering | LOW (CPU) | MODERATE | MODERATE | MODERATE | N-by420 features/PCA; compiled CPU operations and seed bridge. |
| KASBA | LOW (CPU) | HIGH | HIGH | HIGH | MSM/barycenter length-squared work; GPU upgrade does not accelerate source. |
| FASA | LOW (CPU) | MODERATE | MODERATE | MODERATE | FFT/aligned centroid arrays; no eigensystem; actual all 44 runtime unmeasured. |

All eleven L>=1024 tasks must remain in coverage; dropping them after failures would hide the long-series challenge. Smaller native min(B,N) rules are already part of TFMCC/TS2Vec behavior. A new globally smaller requested batch is a separately frozen configuration, not an automatic task-by-task OOM fallback. Contrastive methods do not inherit an equivalence claim from gradient accumulation. CPU execution can preserve equations where supported, but speed and feasibility must be reported in a separately declared hardware track. More VRAM can preserve the frozen method, whereas shrinking a head or truncating sequences changes the evaluated method. This audit authorizes neither hardware-dependent configuration changes nor new hardware runs.

## Complete-Pipeline Timing Contract

Answer the question: **how much end-to-end computational work is required to obtain the full clustering from standardized dataset input?**

Define t0 after raw file decoding into the canonical numeric array/IDs and before common z-normalization. Charge normalization and all subsequent per-run scientific computation through final full-pool extraction/prediction; t1 is after GPU synchronization and completed ordered predictions. If a pre-normalized cache is used, account for that run's normalization cost equivalently and record cache policy. Do not precompute features, centers, representations or graphs off the clock.

| Method | Counted per-run stages |
| --- | --- |
| Euclidean | Common normalization, conversion, all10 center initializations/Lloyd restarts, selected labels/full-pool assignment. |
| k-Shape | Normalization/shape conversion, partition/centroid initialization, all FFT assignments/alignment/scatter/eigen updates, pool overhead, full-pool labels. |
| CKM | Normalization/input preparation, AE pretrain, center initialization, Concrete joint training/annealing, final inference after faithful recipe recovery. |
| DTCC | Normalization, view/noise generation, recurrent training/reconstruction, indicator initialization/all SVD updates, final declared clustering extraction. |
| CDCC | Normalization, time views/FFT-frequency views, both encoders/all losses and fixed-epoch extraction after source resolution. |
| FCACC | Normalization, crops/masks/scale, pretrain, representation extraction/center initialization, fuzzy joint refinement/all per-cluster forwards, final active-SWA inference. |
| TFMCC | Normalization, crop generation, wavelet decomposition/scaling/reconstruction, encoder/head training/all contrast levels, final SWA head inference. |
| TS2Vec + k-means | Normalization, all representation training, complete full_series extraction, all downstream initializations/refinement and ordered assignment. |
| R-Clustering | Normalization, kernel/dilation/quantile/bias fitting, compiled feature extraction, feature standardization, PCA fit/transform, all KMeans restarts and assignment. |
| KASBA | Normalization/shape conversion, elastic initialization, MSM distance programs, barycenter updates, pruning/assignment, final full-pool prediction. |
| FASA | Normalization/format conversion, initial assignment/centers, FFT distances/alignment/centroid updates, final member-to-row assignment. |

Do not count only final KMeans for TS2Vec or R-Clustering. No selected method requires a separate graph stage; any future recovered source that does must include graph construction. Genuine per-fit JIT compilation or initialization belongs in end-to-end cost; cache the scientific representation only under a complete identical-run identity, and disclose cold/warm compiler-cache treatment.

Exclude one-time repository/data download, package installation and archive extraction; separately record file-read/process-start and checkpoint/export I/O as operational overhead. Central evaluation/Hungarian scoring and optional dashboard/silhouette diagnostics are outside algorithm-only timing. Required unlabeled diagnostic forwards that preserve native model mode/RNG remain charged scientific computation; disabling diagnostic telemetry must not change the fit.

Use host monotonic wall-clock stages, one fit at a time, CPU BLAS/OpenMP/Numba and process thread budget1, fixed precision/device/batches. Synchronize CUDA immediately before and after GPU stages and total boundaries; GPU event timings are supporting kernel measures, not replacements for wall-clock pipeline cost. Keep backend warm-up in a disposable operation with measured fit RNG/weights restored; do not run an extra fit for timing.

On Audit04's predeclared detailed panel (all 11 long tasks plus Beef, ECG200, SyntheticControl, ShapesAll), inference uses three untimed warm-up batches and five complete timed full-pool passes; report median/spread, throughput and B/device. Preserve inference RNG for inherently stochastic extraction and use the frozen extraction procedure; no quality-based pass selection. Those passes are not five training seeds. Record peak VRAM/RAM and per-task failure/timeout coverage. Actual timers, synchronization and memory instrumentation are future implementation gates, not completed checks.

## Baseline Readiness Matrix

Full pins are in the source table and registry; abbreviations here refer only to those objects. Paper verification confirms identity/primary evidence, not complete recovery of every experimental setting.

| Method | Paper verified? | Official code? | Frozen source commit | License | Native protocol understood? | Predictions exportable? | Known-k compatible? | Label leakage risk | Adaptation needed | Environment complexity | GTX1080 feasibility | Development calibration? | Scientific code change required? | Readiness | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Euclidean | Yes | Canonical library; third-party to Lloyd | 719c0c6036ac | BSD3 | Yes | Yes, labels_ | Yes | LOW | Loader/seeds/export/timing | LOW | CPU, moderate cost | No | No | READY FOR IMPLEMENTATION | Integration smokes pending |
| k-Shape | Yes | Author-maintained | 778e735624d1 | MIT | Yes | Yes, labels_ | Yes | LOW | Shape/spawn/seeds/export | LOW-MODERATE | CPU; high long-L risk | No | No | READY FOR IMPLEMENTATION | Windows spawn/centroid/export smoke |
| CKM | Yes | None found | null | Unknown | PARTIAL | Unknown | Paper yes | UNKNOWN | Source/recipe recovery | UNKNOWN | UNKNOWN | After source recovery only | New implementation decision if source absent | BLOCKED | Official artifact/complete recipe absent |
| DTCC-2023 | Yes; final methods partly preprint | Yes | 8531376c0d4e | Not stated | PARTIAL/discrepant | Difficult | Yes | HIGH, best metric epochs | Source/env/selection/extraction | VERY HIGH | VERY HIGH | Conditional after source resolution | Gradient-coupling repair is scientific | BLOCKED | TF2/TF1; disconnected cluster term; label-best output |
| CDCC | Yes | Yes | 013865ff4d13 | Not stated | Yes; released behavior differs | In principle; eval RNG | Yes | Paper tuning; final checkpoint not best RI | Corrected source/reviewed reconstruction | HIGH | HIGH; small-N batch invalid | After source resolution only | View/axis repair changes behavior | BLOCKED | Integer selector and one-step recurrent axis |
| FCACC | Yes | Yes | 78b5e5a138ed | Not stated | Mostly | Yes with index scatter | Yes | Diagnostics/search history | Loader/index/SWA/config | MODERATE-HIGH | HIGH; B8/CPU RAM | No automatic search | No algorithm change expected | READY WITH CONDITIONS | Env/n_init/SWA/mode/ID parity |
| TFMCC | Yes | Yes | 882764f52b9e | Not stated | Mostly | Yes, final head | Yes | Diagnostics; no best branch | Env/path/global resource config | HIGH | VERY HIGH | Required | New B is material config | READY WITH CONDITIONS | Ambiguous pins,8GB native feasibility, optimizer discrepancy |
| TS2Vec + k-means | Yes | Official encoder | b0088e14a997 | MIT | Yes; task differs | Yes | Yes | Supervised native head excluded | Pooled SSL/full_series/fixed head | MODERATE | HIGH long crops | No | No; explicit task adapter | READY WITH CONDITIONS | Pipeline/export/timing/env smoke |
| R-Clustering | Yes | Yes, notebook | 3ed571eebe9e | GPL3 | Yes, caveats | Yes | Yes | Historical development labels | Safe cells/Numba RNG/PCA/export | MODERATE | CPU, moderate | No | No silent feature/0PC repair | READY WITH CONDITIONS | Compiled RNG/420-feature/zeroPC validation |
| KASBA | Yes | Publication-linked library | 0fed29f21908 | BSD3 | Yes | Yes | Yes | LOW | Shape/seed/export | MODERATE | CPU; HIGH long-L cost | No | No | READY FOR IMPLEMENTATION | Isolated aeon/Numba/Development8 smoke |
| FASA | Yes | Yes, FASA_I_I | 9c05d415efee | Not stated | Yes | Yes via members | Yes | LOW | Shape/member export/seed | LOW-MODERATE | CPU, moderate | No | No | READY WITH CONDITIONS | Seed/env/export/license-for-distribution clarity |

**Counts:3 READY FOR IMPLEMENTATION;5 READY WITH CONDITIONS;3 BLOCKED;0 REPLACE / REMOVE CANDIDATE.** Every method still needs future engineering evidence; these are not successful import, fit, prediction-reload or timing checks.

## Tier-1 Reassessment

| Candidate | Decision | Distinct scientific challenge |
| --- | --- | --- |
| Euclidean k-means | KEEP TIER 1 | Direct Euclidean distortion on raw normalized series; simple reference. |
| k-Shape | KEEP TIER 1 | Classical shift/shape invariance and eigenvector centroid. |
| CKM | KEEP TIER 1 | Concrete hard assignments/joint centers; lineage competitor. |
| DTCC-2023 | KEEP TIER 1 | Recurrent reconstruction plus spectral/contrastive clustering. |
| CDCC | KEEP TIER 1 | Cross-domain time/frequency contrast. |
| FCACC | KEEP TIER 1 | Fuzzy memberships modify contrastive pair treatment/centers. |
| TFMCC | KEEP TIER 1 | Multi-scale wavelet views plus temporal/instance/cluster losses. |
| TS2Vec + k-means | KEEP TIER 1 | Learned hierarchical representation with separate clustering optimization. |
| R-Clustering | KEEP TIER 1 | Untrained random convolution/PCA features with k-means. |
| KASBA | KEEP TIER 1 | Elastic metric, stochastic barycenters and triangle pruning. |
| FASA | KEEP TIER 1 | Efficient aligned centroid without k-Shape eigensystem. |
| Historical modified DTC / DTCR / local historical DTCC scores | HISTORICAL ONLY | Preserve context; no faithful matched-protocol substitution. |

Contrastive methods overlap in some losses, but differ in domain, membership, augmentation and optimization. k-Shape/FASA test a specific quality/cost tradeoff. None is retained merely for table size or removed for execution difficulty. Formal pre-result unavailability may leave a documented Tier-1 slot without scores; this must not be described as all eleven reproduced.

## Failure Policy

Predeclare failures before accuracy. The three current BLOCKED assessments are **not three empirical failed runs**. Preserve every attempt/log/config/seed identity and negative/unexpected outcome; never overwrite original outputs.

| Classification | Objective criterion | Reporting / remedy |
| --- | --- | --- |
| UNAVAILABLE | No official/author artifact after documented primary/author/repository searches and insufficient complete faithful recipe; or pinned dependency/wheel closure cannot be obtained/constructed | Record searches/date/pins/missing component. CKM currently has source-unavailable evidence. Third-party replacement is not automatic. |
| FAILED REPRODUCTION | Future pinned native/synthetic/Development8 execution or numerical/contract failure on valid input, or demonstrated algorithm behavior contradicting the claimed method under declared checks | Retain traceback/invalid output and source evidence. Lower ARI than a paper alone is not proof without matching cohort/recipe/evaluator/repetitions. |
| INCOMPATIBLE PROTOCOL | Target-membership-dependent training/selection cannot be removed faithfully; or one prediction per canonical record cannot be exported without changing the method | Identify the failed contract. Native best-only scalars cannot fill the common table. |
| RESOURCE FAILURE | Identical frozen method/config/seed has verified device OOM, host allocation failure or exceeds the frozen wall cap | Log task/seed/device/B/peak memory/elapsed cap. No score-based batch/architecture/length rescue. |

Propose a **common 24-hour wall-clock cap per dataset/method/root-seed pipeline**, including initialization/JIT/representation training. This is an engineering budget chosen before comparative accuracy, **not an estimated runtime**. Freeze timeout, monitoring interval and a host-RAM admission rule based on measured physical capacity before final runs; current timeout gate is pending. A shorter tiny-smoke limit is declared separately and cannot be reported as final algorithm infeasibility.

A verified transient infrastructure interruption permits one retry with **identical** source/input/config/environment/seed and a new attempt ID; retain both. Source errors, numeric failures, OOMs and timeouts do not trigger open-ended retries or changed settings. CPU/new-device retries require a predeclared separate hardware track and budget, without post-accuracy rescue.

R-Clustering zero-PC output on valid input is a traceable native representation failure/domain incompatibility, not permission to clamp its PCA rule. Failed checkpoint/ID export is a contract failure unless a faithful adapter is established. Disabling a label branch that changes the actual trained method requires a separately named scientific adaptation. No missing value is imputed as ARI0, no failed seed replaced by a favorable one, and no unmatched paper score substituted.

Publish per-task/per-seed failure coverage and denominators. Available-case descriptions and conditional common-support analyses must list excluded tasks/methods and changed n. A reduced panel is incomplete planned coverage, not the original Final36 result. Freeze method-level unavailable/incompatible decisions before final accuracy; later observed resource/numeric failures follow these rules.

## Development8 / Final36 Firewall

Development8 is exactly **SyntheticControl, Beef, ECG200, OSULeaf, ShapesAll, SemgHandMovementCh2, CinCECGTorso, StarLightCurves**. It may support debugging/import/synthetic checks, one-task fit/export/reload smoke, global compatibility/resource choices and only needed bounded calibration. LoSTer's closed method selection does not reopen.

Final36 is exactly:

ACSF1, Adiac, ArrowHead, CBF, Car, CricketX, CricketY, CricketZ, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistalPhalanxTW, ECGFiveDays, EOGVerticalSignal, FaceFour, FiftyWords, Fish, Fungi, GunPointMaleVersusFemale, GunPointOldVersusYoung, HouseTwenty, LargeKitchenAppliances, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxTW, MixedShapesRegularTrain, MixedShapesSmallTrain, PigAirwayPressure, PigArtPressure, PigCVP, Plane, PowerCons, ProximalPhalanxTW, ShapeletSim, SwedishLeaf, ToeSegmentation1, Trace, WordSynonyms.

It is EXTENDED-44 minus those eight, not a favorable-cost/score subset. Final36 was not used for new baseline adaptation/tuning here. Published/historical task knowledge persists: "untouched" describes the2026 development-selection firewall, not globally secret datasets.

Before Final36 execution freeze: retained/unavailable methods; source/adapter commits and env locks; input/ID/normalization/intrinsic transforms; architecture/loss/head/B/budgets/stopping; root/phase seeds/restarts; active checkpoint/inference/export; timing/thread/precision/device/cache policy; resource/failure/retries; statistical branch and source-family mapping. Keep baseline fit workers free of ground-truth membership; save immutable predictions/manifests before central evaluation.

Final36 labels may not select preprocessing, architecture, augmentation, latent dimension, epoch, seed, checkpoint or recovery batch. Do not selectively repair final failures after inspecting accuracy: preserve failures and document a campaign-level source/config decision and rerun scope before new comparisons. Exact-identity reuse is permitted; old untraceable scalar results are not reusable benchmark runs.

## Implementation Waves

**Proposal only; none executed.** Every admitted method must: verify external clone/commit; build a hashed isolated environment; run import/tiny synthetic smoke; run one predeclared Development8 fit; validate IDs/export/active-state reload and central/native metric conventions; instrument complete timing/memory; freeze wrapper/config/env/source/failure ledger. Tiny smoke budgets are not final accuracy recipes.

| Wave | Methods / rationale | Development smoke emphasis | Exit evidence |
| --- | --- | --- | --- |
|1 | Euclidean, k-Shape, KASBA first; R-Clustering and FASA with their conditions | ECG200 for finite/order checks; k-Shape spawn; KASBA seed/MSM; R compiled RNG/420/PCA; FASA member IDs/zero-norm behavior | Environment, repeatability, ordered export/reload, central metrics and timers pass. |
|2 | TS2Vec + k-means, FCACC, then TFMCC | ECG200 pipeline/SWA/head; Beef small-N; ShapesAll k60; SemgHandMovementCh2/CinCECGTorso/StarLightCurves only as needed for global resources | Preserve official views/losses; fixed recipes; TFMCC valid env/global B decision before Final36. |
|3 | CKM, DTCC-2023, CDCC source-resolution first | No training until faithful source/recipe candidate established; recover CKM; resolve DTCC gradient/env/selection; corrected CDCC artifact or reviewed reconstruction | Reopened source correspondence gate, then same synthetic/one-Development8/export/evaluator/timing/freeze, or formal unavailable decision. |

Ten identified source clones already exist: do not reclone unchanged sources to satisfy a checklist. Never execute native all-UCR drivers during one-task smoke. Future tracked artifacts are registry, thin wrappers, configs and notes; no vendor repos inside ISPAMM. No environments are installed in this task.

Use debug/calibration seeds0,1,2 as applicable, not final100..104. One Development8 smoke establishes integration, not all 44 memory feasibility; additional targeted pre-score resource checks must be justified. Do not automatically spend72 calibration fits. Implementation is the next stage and requires a new user instruction; current authorization stops at these deliverables.

## CORE-44 Authorization Gates

All twelve gates must pass **and** the user must explicitly authorize the final campaign.

| Gate | Evidence required | Current status |
| --- | --- | --- |
|1 LoSTer frozen | Committed specification/implementation, annotated origin tag | **PASS** cab7b6f / loster-2026-frozen. |
|2 EXTENDED-44 list | Canonical40+4 names | **PASS (list)**; run input/ID validation pending. |
|3 Final36 firewall | Exact36 and no2026 score-based adaptation | **PASS for audit/design**; enforce execution firewall next. |
|4 Tier-1 set frozen | Pre-result inclusion/unavailability ledger committed | **DESIGNED**;11 retained/3 blockers, intentionally uncommitted report. |
|5 Every comparator smoke or formal blocked/unavailable | Pinned import/fit/export pass or final formal source/protocol classification before results | **PENDING**:8 candidates untested,3 source blockers recorded. |
|6 Common evaluator | Metric plus cross-baseline integration checks | **PARTIAL**: frozen metric unit checks exist; new integration pending. |
|7 Predictions/export | Exact IDs, invalid-input rejection, active checkpoint reload | **PENDING** contract only. |
|8 Timing implemented | Stage/sync/thread/memory/cache instrumentation | **PENDING**. |
|9 Source/env freeze | Source/adapter commits and reproducible locks | **PARTIAL**:10 source pins;CKM missing;no baseline locks/adapters. |
|10 Failure/OOM/timeout frozen | Execution limits/retries/hardware/failure ledger | **DESIGNED**; proposed limits not execution-frozen. |
|11 Final seeds |100..104, phase mapping, native constants, repeatability | **DESIGNED**; root seeds tracked; hooks untested. |
|12 Statistics script | Prepared/tested on fabricated data without final outcomes | **PENDING**; no new statistics script executed/created here. |

Prepare dataset-level seed-mean ARI analysis;36x5 is not180 independent tasks. Audit04's changed-method branch applies to the explicit MonotoneWarp choice: with complete support, final LoSTer+11 comparators+Legacy-Clean historical-warp reference yields **13 methods and12 Holm contrasts** (11 baseline contrasts plus separately identified method-change validation). The reference is a scientific control, not a twelfth Tier-1 baseline or a fallback selected by final scores. Freeze this branch before results; no reference run is authorized here.

Use Friedman alpha.05 then predeclared two-sided control Wilcoxon contrasts after omnibus rejection, Holm correction, Wilcox zero removal/average ties, effective n, raw/adjusted p-values, paired mean/median effects, wins/ties/losses and signed rank-biserial effects. Use 10,000 dataset-level bootstrap resamples with fixed analysis seed and source-family sensitivity. Arithmetic NMI is secondary, with any tests a separate exploratory corrected family; RI/ACC add no primary test family. Synthetic validation must cover all ties, missing seeds/failures, unavailable methods, related-task families and13-method/12-contrast handling. If unavailable methods reduce support, predeclare/report conditional analysis and actual n; do not present it as full planned coverage.

**CORE-44: NOT AUTHORIZED.** Warp/source preservation passes do not satisfy engineering gates or override the explicit stop.

## Recommended Next Step

The next authorized engineering task should start with Wave1's Euclidean/k-Shape/KASBA adapters and common ID/evaluator/timing infrastructure, while resolving the three blocked sources and deep environments. Deliver reviewable wrappers/locks/configs, tiny/Development8 prediction-reload records, timing evidence and a pre-result failure ledger. Resolve TFMCC's environment/global resource recipe within the allowed firewall. Do not alter frozen LoSTer behavior or silently repair competing algorithms.

Commit the baseline registry/protocol only in that later stage after implementation conditions have evidence. **Do not commit or push Audit08 now.**

Terminal summary: LoSTer **cab7b6ffecab92bfcf6fe46e73e32dc0c040ee59 / loster-2026-frozen**;11 retained,0 moved/dropped;3 ready,5 conditional,3 blocked;development calibration TFMCC (other blocked-method choices only after source recovery);GTX1080 most problematic DTCC/TFMCC, with FCACC/TS2Vec long-series and CPU k-Shape/KASBA risks;Wave1 classical/random features -> Wave2 official deep -> Wave3 source/legacy resolution;**NOT AUTHORIZED**. Main blockers: CKM artifact/recipe, DTCC correspondence/env/selection, CDCC view/axis defects, TFMCC env/resources, and baseline export/timing/env/statistical integration smokes.

