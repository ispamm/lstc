# LoSTer 2026 Literature, Novelty, and Benchmark Audit

Literature cutoff: **2026-10-05**. Audit date: **2026-10-05**. Analysis only; no scientific implementation, manuscript, dataset, result, training environment, or experiment was changed.

## Executive Summary

1. **What is novel?** No new optimization primitive, residual dense block, or dual contrastive objective is established. The defensible contribution is a particular integration of Concrete K-Means-style hard latent assignments, a TiDE-derived global residual dense autoencoder, and existing instance/cluster contrastive regularization for fixed-length, known-k, univariate clustering. Whether that integration produces a useful accuracy–efficiency tradeoff remains an empirical question. Finding no identical configuration in this search does not establish priority.
2. **Claims to abandon:** first differentiable Gumbel k-means; first neural hard-forward k-means; optimization without any surrogate; globally better optimization guaranteed by hard assignments; a fundamentally new dense architecture; invention of dual contrastive learning; universal Transformer inadequacy; all Transformers having quadratic point-wise attention; arbitrary-length support; current state-of-the-art superiority established by the historical tables.
3. **Mandatory modern comparators:** CDCC (2024), DEETO (2024), TSGCC (2025), FCACC (2026), TFMCC (2026), and the eligible 2026 cross-view diffusion DTCC. APCL (KDD 2026) is additionally mandatory for novelty discussion and especially important experimentally for a hard/prototype-centered narrative. These are scientific requirements, not claims that every released artifact is immediately executable. DEETO, diffusion DTCC, and APCL have paper/protocol or artifact gaps that must be resolved before faithful experiments. R-Clustering, KASBA and FASA are mandatory strong modern non-deep comparators for an efficiency narrative.
4. **Dataset count:** 40 predeclared UCR datasets is the minimum broad empirical campaign; **44 is the recommended practical strong campaign**: CDCC's fixed-length 40 plus four historical datasets that strengthen natural long-sequence coverage. The historical 17 alone is inadequate for broad 2026 superiority claims, but weak and defensible for a tightly scoped mechanistic study. Full UCR is not automatically necessary.
5. **Transformers:** zero additional forecasting-backbone adapters are required for the minimum clustering paper. One representative efficient backbone, preferably PatchTST, is useful for the stronger architecture/efficiency narrative. Multiple forecast adapters do not replace contemporary direct clustering baselines. APCL is a separate direct clustering comparator whose author code includes a Transformer encoder.
6. **Multivariate:** **UNNECESSARY FOR A CLEAR UNIVARIATE SCOPE**. Several leading recent papers remain univariate. EMTC's 15 UEA datasets show that multivariate clustering is established; the manuscript's “underexplored” implication needs revision. A multivariate extension broadens scope and only adds methodological novelty if it introduces and validates a real cross-variable mechanism.
7. **Long sequences:** keep length as a measured stress regime, not a priority claim or an established separate benchmark category. Prefer **C: efficient deep time-series clustering**, with **D: hard-assignment temporal clustering** as the mechanistic question. Competitors already examine long sequences or scalability.
8. **Smallest campaign likely to support a strong submission:** 44 UCR datasets, five fresh seeds initially and ten for the strong tier; six modern direct competitors, CKM and DTCC mechanism controls, Euclidean k-means, k-Shape, R-Clustering, KASBA, FASA, and TS2Vec + k-means; targeted APCL comparison after its protocol is recovered; one optional efficient Transformer control; a small controlled scaling study and eight predeclared ablation datasets. This is a conditional recommendation, not a prediction of acceptance or positive results.

Stage A is complete: Audit 2 was committed as **e09b9db462b165ac16af100ab5d06ded64bb151c**, message “Add LoSTer provenance and artifact-recovery audit,” and pushed only to **origin revision-2026**. The branch was clean before literature work. Audit 3 remains uncommitted.

## Search Method and Evidence Quality

### Scope, sources, and limits

The search started from the requested precursors and five named modern competitors, then followed their publisher reference lists, competing-method tables, author publication pages, and author-linked repositories. Additional searches used combinations of “time series clustering,” “contrastive clustering,” “prototype,” “Gumbel,” “graph,” “topological,” “evolutionary,” and publication years 2024–2026. Searches also covered efficient non-deep clustering and multivariate UEA evaluations. Conference proceedings, publisher pages, DOI registration records, accepted author manuscripts, and author code were used as evidence. Secondary indexes served only to locate primary records.

This is a focused reproducible scoping audit, not a claim of exhaustive systematic-review recall. Search completion does not prove that no earlier temporal CKM adaptation exists. Internet access was available. Individual publisher challenges, paywalls, HTTP 403 responses, and intermittent raw GitHub connection resets are recorded as access limitations, not absence of a paper or code.

Published final papers take precedence. Where a final publication exists, preprint versions were not used as the operative evidence for final experimental counts or methods. DTC and TFEC are explicitly identified as preprints without a verified final publication. CKM's accepted final manuscript was accessible through indexed method passages even though its binary download was blocked. TiDE's author source was retrieved through GitHub's read-only API after raw-host failures. Temporary publisher PDFs used for extraction were outside the repository and removed; no model code was executed.

**Evidence labels:**

- **FULL-PAPER EVIDENCE:** final publisher PDF/HTML or accepted final manuscript method passages inspected. Where only selected passages were accessible, that restriction is stated.
- **ABSTRACT-LEVEL EVIDENCE:** primary abstract, indexed introduction/section excerpts, or metadata only. Such evidence cannot establish unseen experiment protocols.
- **CODE EVIDENCE:** static inspection of a named author repository. A script default is not proof of the configuration used to produce a publication's table.
- **UNKNOWN:** not established from inspected evidence. It does not mean “no,” “unsupported by any version,” or “code unavailable everywhere.”

For FULL PDFs of CDCC, CC, TFMCC and EMTC, the main methods and experiment sections were inspected; a partial text extraction is not an exhaustive audit of supplements. DEETO, TSGCC, DETC, spectral masking, diffusion DTCC and APCL retain explicit full-paper/protocol gaps. Claims about their superiority are not adopted as independently validated facts.

### Local anchors

- [Audit 1](../audits/01-paper-code-audit.md): implemented network, active losses, assignment gradients, metrics, loader defects, adapter fairness.
- [Audit 2](../audits/02-provenance-recovery-audit.md): recoverability of runs, checkpoints, retail preprocessing, timing and convergence artifacts.
- [Historical manuscript](../../paper/original_r1/elsarticle-template-num-revised.tex), abbreviated **M** below. Line numbers refer to this immutable file, not a reconstructed draft.
- Scientific inherited commit: **b594b650f68ebed80436b8de7313fa1536d3d95c**. Audit reports document later preservation work; it is not new scientific training provenance.
- Current requirements describe PyTorch 1.13.1+cu117, NumPy 1.22.4, pandas 1.5.3, scikit-learn 1.4.1.post1, SciPy 1.13.0 and tslearn 0.6.3. These are declared pins, not a verified installed/runnable environment.

The original submission artifact predates the inherited August 2025 code. Dated correspondence establishes a September 2025 revision context; uniform May 2026 export timestamps do not establish when individual claims were first made. No private correspondence is reproduced. Later 2026 competitors matter for a new 2026 submission even where they cannot refute an earlier priority date by themselves.

## Literature Timeline 2020-2026

| Period | Verified development | Consequence for LoSTer |
|---|---|---|
| Before 2020 | DTC is a 2018 preprint; DTCR is NeurIPS 2019 | Joint temporal representation and clustering already existed. DTC is not verified as an accepted ICLR paper. |
| 2020 | CKM, ICASSP | Concrete/Gumbel assignments, hard forward/relaxed backward differentiation, and jointly learned latent centroids predate LoSTer. |
| 2021 | Contrastive Clustering, AAAI | Row-wise instance and column-wise cluster contrast plus balancing entropy are established components. |
| 2022 | TS2Vec, AAAI | Strong general time-series representations provide a meaningful two-stage clustering comparator. Original representation/classification results are not original clustering results. |
| 2023 | TiDE, TMLR; DTCC, Neural Processing Letters | Residual dense temporal modeling and dual-view temporal reconstruction/cluster/contrastive integration precede LoSTer. |
| 2024 | CDCC, AAAI; DEETO, KBS; R-Clustering and RandomNet, DMKD; TimesURL, AAAI | Cross-domain contrast, topology alignment and inexpensive random-feature clustering substantially expand the comparison set. |
| 2025 | UCL-TSC, Electronics; TSGCC, KBS | Smaller 12-dataset studies still exist, while graph-augmented work reports the full 128 UCR archive. |
| 2026, eligible before cutoff | FCACC, PR; TFMCC and EMTC, AAAI; KASBA, DMKD; FASA/MUFASA, SIGMOD/PACMMOD; DETC, IPM; APCL, KDD; spectral masking, IDA; diffusion DTCC, ESWA advance online | Fuzzy/cluster-aware, time-frequency, masking, graph/evolutionary and prototype alternatives invalidate a comparison limited to legacy RNNs and forecast adapters. |
| Emerging only | TFEC, January 2026 preprint | Relevant multivariate context; “submitted to ICASSP 2026” is not verified acceptance. |

FCACC's 2026 issue paper was available online in December 2025. Diffusion DTCC has a December 2026 issue assignment but verified **28 May 2026 advance-online publication**; it is eligible at the cutoff. An ambiguous secondary match to a different 2025 ESWA DOI was not substituted for that verified record. APCL's publisher-registered KDD record has August 2026 publication dates and is eligible. See the bibliography for full author lists, status, DOI and code.

## CKM vs LoSTer

### The objective and the differentiation distinction

For a latent matrix Z, centers C, and one-hot assignment matrix H, canonical latent k-means has the constraint H in {0,1}, with each row summing to one, and objective ||Z − HC||². CKM's accepted manuscript explicitly introduces that hard formulation, reconstruction, distance-derived assignment probabilities, Concrete sampling, and joint learning. Its method sections and Algorithm 1 are the decisive prior evidence. [CKM accepted final manuscript](https://www.research.ed.ac.uk/files/134695608/Deep_Clustering_with_Concrete_GAO_DOA24012020_AFV.pdf), [ICASSP record](https://cmsworkshops.com/ICASSP2020/Papers/ViewPaper.asp?PaperNum=4196).

LoSTer's active code likewise computes distance/RBF-derived logits (sigma=1) and calls PyTorch Gumbel-softmax with hard=True, annealing temperature from 10 by a factor of 0.65 per epoch to a floor of 0.01. A sampled forward assignment is one-hot. The backward path differentiates a temperature-dependent soft relaxation. Consequently:

- The **forward clustering loss value** uses a hard selected center.
- The **gradient estimator** is a straight-through soft surrogate and generally biased.
- This does not make the true discrete assignment map differentiable, establish an unbiased exact gradient, or guarantee the global minimum of canonical k-means.
- A sampled hard assignment need not select the deterministic nearest center during training. Reconstruction and contrastive terms also change the complete training objective.
- “Uses a hard latent squared-distance term rather than DEC's KL target-distribution loss” is accurate. “Uses no surrogate of any kind” is inaccurate.

These are mathematical/implementation conclusions, not new claims made by CKM.

### Twelve-axis comparison

| Axis | CKM, accepted final evidence | LoSTer, active implementation | Novelty implication |
|---|---|---|---|
| Canonical objective | Binary one-hot k-means constraint | Hard sampled latent squared distance, averaged over batch and latent coordinates | Same family; distinguish complete regularized loss from pure k-means |
| Concrete/Gumbel | Distance probabilities and Concrete sampling | RBF log probabilities, sigma=1, annealed Gumbel temperature | Pre-existing mechanism |
| Straight-through estimator | Explicit straight-through formulation | hard=True soft-gradient path | Pre-existing estimator |
| Hard forward | Hard assignment formulation and ST sampling | One selected cluster per example/view | Not new |
| Relaxed backward | Temperature relaxation | Soft backward despite one-hot forward | No “surrogate-free” differentiation |
| Centroids | Learned center matrix | Active criterion centers, optimized separately from network | Joint center learning pre-existing; checkpoint defect remains |
| End-to-end | Encoder, decoder and centers jointly optimized after pretraining | Two pretrained autoencoders then joint reconstruction/clustering/contrastive training | Integration differs; principle does not |
| Architecture | Fully connected/shallow networks in reported experiments | Global residual dense encoder/decoder, fixed width 256 | Temporal architecture adaptation |
| Clustering loss | Latent center-distance plus reconstruction | Two-view center-distance plus reconstruction, instance/cluster contrast and entropy | Added established regularizers |
| Initialization | Pretraining and supplied initial centers in accepted Algorithm 1; exact center-initialization policy not fully verified here | Separate view-wise full-pool k-means initialization, seeds 0–4 | Exact initialization comparison needs complete artifact |
| Inference | Exact final test rule not fully verified in accessible accepted passages | Original-view embedding, deterministic nearest active center | Do not fabricate a verified identical test pipeline |
| Modality | MNIST, USPS, text and tabular experiments | Fixed-length univariate UCR and saved retail arrays | Domain-specific application, not a new estimator |

An official CKM author repository was not identified. LoSTer's inherited CKM comparator is a local implementation, not a verified official-author release. Its status and architectural matching must be disclosed. Accessible accepted sections establish the decisive priority facts; unknown initialization/inference details do not reverse them.

### Verdicts on possible first claims

| Proposed claim | Verdict | Reason and permissible scope |
|---|---|---|
| 1. First Gumbel-softmax differentiable k-means | **NOT NOVEL**; the first claim is false | CKM explicitly precedes this primitive. |
| 2. First direct hard k-means optimization in a neural model | **NOT NOVEL**; unqualified exact-optimization wording **FALSE / UNSUPPORTABLE** | CKM supplies the hard-forward neural mechanism. Both methods use relaxed gradients. |
| 3. First adaptation of CKM to time-series clustering | **NOT ESTABLISHED** | No exhaustive priority proof; a temporal application is not itself a new methodological primitive. APCL adds current temporal Gumbel/prototype context, but its code loss is not canonical CKM. |
| 4. First CKM method for long time series | **NOT ESTABLISHED** | No standardized “long” threshold or demonstrably distinct long-sequence mechanism; most architectural dependence is on fixed input length. |
| 5. First CKM plus dual contrastive temporal clustering | **NOT ESTABLISHED** | No identical complete configuration was located, but an intersection of known ingredients does not prove firstness or substantial novelty. |
| 6a. This implementation combines a residual dense AE, hard centroid loss and dual-view contrastive regularization | **SUPPORTED WITH QUALIFICATION** | Factual integration claim; cite CKM, TiDE, CC and DTCC. Its benefit requires controlled evidence. |
| 6b. Hard assignments avoid a periodically frozen spectral assignment update | **SUPPORTED WITH QUALIFICATION** | Describes a procedural difference from selected DTCR/DTCC implementations; does not prove absence of inconsistency or instability. |
| 6c. No surrogate gradients; exact/unbiased hard-objective differentiation | **FALSE / UNSUPPORTABLE** | Straight-through Gumbel gradients are relaxed surrogate gradients. |
| 6d. Better accuracy or convergence follows necessarily from the hard loss | **FALSE / UNSUPPORTABLE** | No theorem or valid controlled evidence establishes necessity. |

APCL's [author assignment code](https://github.com/William-Liwei/APCL/blob/main/models/soft_assignment.py) exposes a soft-to-hard Gumbel option and argmax evaluation. Its assignment loss is cross-entropy/entropy-based and its default loss path is soft; this is related evidence, not proof that APCL exactly reproduces CKM's squared-distance objective. The report therefore does not use APCL to manufacture a stronger priority counterexample than the evidence warrants.

## TiDE vs LoSTer

The operative architectural evidence is TiDE's [official model source](https://github.com/google-research/google-research/blob/master/tide/models.py) and [README](https://github.com/google-research/google-research/tree/master/tide), together with Audit 1. The final TMLR record is verified, but the OpenReview final PDF was blocked; this section is **CODE EVIDENCE**, not a claim to have read the complete final TiDE article.

For both ordinary residual blocks, the structural expression is:

**LayerNorm(Dropout(W2 ReLU(W1 x + b1) + b2) + Ws x + bs)**

TiDE makes layer normalization configurable; LoSTer's ordinary active block uses it. The learned linear skip, one hidden ReLU, second linear projection, dropout location and post-addition normalization coincide. This establishes structural borrowing/adaptation, not proof of verbatim source copying.

| Architectural axis | TiDE author implementation | LoSTer active network | Assessment |
|---|---|---|---|
| Residual dense block | Two dense transformations and learned linear skip | Same core design | Pre-existing architectural unit |
| Normalization | Optional block LayerNorm; optional reversible input transform | Block LayerNorm; active runs do not enable RevIN | Normalization is not an added innovation |
| Skip projections | Learned projection per block plus separate history-to-horizon linear branch | Learned block projections; no overall AE input-to-output residual | Do not conflate block skip and whole-model skip |
| Encoder | Dense residual stack over history, encoded past/future features and series embedding | Three ordinary blocks, first L→256, then width 256 | Simpler global temporal encoder |
| Decoder | Dense horizon decoder, reshape, per-time final decoder with future features | Two ordinary 256 blocks, then an unnormalized prediction residual block to L | Forecast-specific decoder removed |
| Bottleneck | Configurable hidden dimensions; forecast embedding | Fixed 256-coordinate series representation | Not necessarily dimensionality reduction when L<256 |
| Input/output | Past observations and future-known covariates → future horizon | [B,L,1] → [B,256] → [B,L,1] reconstruction | Task substitution |
| Length dependence | History/horizon affect input and output dense shapes | First and last matrices depend on L; network is instantiated per length | Fixed-length global model, not length-invariant |
| Covariates | Dynamic numeric/categorical covariates and learned series identity; README does not implement generic static-attribute support | No covariate branch | Do not attribute absent features to LoSTer |
| Forecast components | Feature projection, horizon reshaping, temporal final decoder, global history residual | Removed | Meaningful task adaptation, no new dense primitive |
| Cluster components | None | Two independent AEs, active trainable centers, hard loss and dual contrast | New configuration for a different task, all component mechanisms have precursors |

**Classification: MINOR ADAPTATION for architectural novelty.** The whole training pipeline is substantially different in purpose, but changing forecasting outputs to reconstruction, deleting covariate/forecast branches and adding known clustering losses does not establish an independently distinct encoder principle. It is not a direct copy of the whole TiDE forecasting system. The honest architectural claim is a **TiDE-inspired residual dense autoencoder tailored to global fixed-length univariate embeddings**.

For width d, Audit 1 counts one active AE as **4Ld + 14d² + 26d + 2L** parameters; two independently parameterized views double that, with another 2kd active centroid parameters. At fixed d and k, leading computation is linear in L, but parameters also grow with L. Finite dense projections do not support arbitrary sequence lengths at inference. A matched plain dense AE, residual AE + post-hoc k-means, and residual AE + CKM are better tests of the architectural contribution than a forecast-only TiDE score.

## Contrastive-Clustering Lineage

| LoSTer component | Direct lineage and equation citation | Actual role in a new paper |
|---|---|---|
| Instance-level cross-view normalized contrast | CC instance loss; temporal two-view adaptation in DTCC §3.4.1, Eqs. 14–15 | **Supporting component**; a familiar instance-discrimination regularizer |
| Cluster-level column contrast | CC's assignment-matrix columns; DTCC §3.4.2, Eqs. 16–18 | **Supporting component**; cluster correspondence regularization |
| Negative entropy/balancing | CC cluster-loss balancing and DTCC Eq. 18 | **Supporting component**; encourages use of clusters, not a proof against every collapse mode |
| Two views | CC uses two augmentations/shared encoder and separate heads; DTCC uses original/augmented temporal reconstruction views | **Implementation choice within an established design**; explain augmentation and weight sharing |
| Hard assignment columns as contrast inputs | LoSTer substitutes Gumbel outputs for preceding soft/spectral assignment constructions | Specific integration choice; CKM plus CC/DTCC citations required |
| Extra softmax after one-hot output | Active LoSTer applies another softmax before column loss | Implementation choice requiring justification/ablation, not a newly derived objective |

[CC final paper and code](https://ojs.aaai.org/index.php/AAAI/article/view/17037), [CC loss implementation](https://github.com/XLearning-SCU/2021-AAAI-CC/blob/main/modules/contrastive_loss.py), [DTCC final publication](https://link.springer.com/article/10.1007/s11063-023-11287-0) establish that **dual instance/cluster contrastive learning is not a LoSTer contribution**.

DTCC already integrates temporal reconstruction, relaxed k-means and both contrastive levels. CDCC then places instance/cluster objectives in time, frequency and cross-domain branches; TFMCC adds temporal hierarchy and wavelet augmentation. FCACC's “cluster-aware contrast” means cluster-guided positive/negative selection, not necessarily the same column-wise CC loss. Diffusion DTCC likewise refines pair reliability and imposes column orthogonality. These differences must not be erased by calling all of them “dual contrastive” and treating objectives as interchangeable.

LoSTer's negative-entropy term is equivalent to CC's balancing term up to constants where the same marginal probabilities are used. Its extra softmax converts a one-hot row into a positive distribution with selected/nonselected ratio e, which changes both column contrast and entropy. That is a code-level behavioral difference; it has no established novelty or benefit.

Audit 1's incorrect printed denominator masks, N-versus-k cluster notation, and missing loss scaling distinctions require mathematical repair in a later authorized manuscript task. Citation cannot repair an equation automatically. Instance directions are summed as two view means, not their half-sum; reconstruction averages over BL coordinates while the center term averages over Bd. View-wise center identities are not explicitly permutation matched. The resulting correspondence risk should be measured; the audit does not claim it proves an implementation failure.

## Modern Competitor Matrix

The following keyed tables together contain every requested field. “Uni/Multi” means **published clustering evaluation**, unless explicitly marked code-capable or representation-only. “Variable” distinguishes padded variable-length handling from native ragged support. “Joint” means representation and clustering interact during learning; alternating stages are identified rather than called a single simultaneous gradient update. UNKNOWN is retained instead of imputing a common protocol.

### Publication, code and evidence

| Method | Year; venue | Peer reviewed? | Official code? | Evidence inspected |
|---|---|---|---|---|
| [CKM](https://doi.org/10.1109/ICASSP40776.2020.9053265) | 2020; ICASSP | Yes | Not identified; local comparator is not official | Accepted final indexed method sections; complete binary PDF blocked |
| [CC](https://ojs.aaai.org/index.php/AAAI/article/view/17037) | 2021; AAAI | Yes | Yes: XLearning-SCU/2021-AAAI-CC | FULL-PAPER + loss source |
| [DTC](https://arxiv.org/abs/1802.01059) | 2018; arXiv | Not verified | FlorentF9 reproduction is third-party | ABSTRACT-LEVEL preprint; historical implementation context |
| [DTCR](https://papers.neurips.cc/paper_files/paper/2019/hash/1359aa933b48b754a2f54adb688bfa77-Abstract.html) | 2019; NeurIPS | Yes | Yes: qianlima-lab/DTCR | Final paper indexed methods/experiments + README |
| [DTCC-2023](https://doi.org/10.1007/s11063-023-11287-0) | 2023; Neural Processing Letters | Yes | Yes: 07zy/DTCC | FULL-PAPER + source |
| [CDCC](https://ojs.aaai.org/index.php/AAAI/article/view/28740) | 2024; AAAI | Yes | Yes: JiacLuo/CDCC | FULL-PAPER + encoder/loader/runner source |
| [DEETO](https://doi.org/10.1016/j.knosys.2024.112434) | 2024; KBS | Yes | Not identified | ABSTRACT-LEVEL + indexed introduction; final full text unavailable |
| [TSGCC](https://doi.org/10.1016/j.knosys.2025.114602) | 2025; KBS | Yes | Yes: zololululu/TSGCC | ABSTRACT-LEVEL/indexed sections + static code |
| [FCACC](https://doi.org/10.1016/j.patcog.2025.112899) | 2026; PR; online 2025 | Yes | Yes: Du-Team/FCACC | FULL-PAPER final author-hosted publisher PDF + code |
| [TFMCC](https://ojs.aaai.org/index.php/AAAI/article/view/39817) | 2026; AAAI | Yes | Yes: Du-Team/TFMCC | FULL-PAPER + loader/runner/train source |
| [Diffusion DTCC](https://doi.org/10.1016/j.eswa.2026.133028) | 2026; ESWA; online May 28 | Yes | Yes: qinqinhan/DTCC | ABSTRACT-LEVEL/indexed intro + README; distinct from DTCC-2023 |
| [UCL-TSC](https://doi.org/10.3390/electronics14081660) | 2025; Electronics | Yes | Not identified | Final publisher indexed methods/tables; FULL-PAPER passage evidence |
| [DETC](https://doi.org/10.1016/j.ipm.2025.104409) | 2026; IPM | Yes | Publisher-linked anonymous artifact | ABSTRACT-LEVEL/indexed sections; artifact contents not verified |
| [EMTC](https://ojs.aaai.org/index.php/AAAI/article/view/39777) | 2026; AAAI | Yes | Yes: yueliangy/EMTC | FULL-PAPER main sections + README; supplements not inspected |
| [APCL](https://doi.org/10.1145/3770855.3817773) | 2026; KDD V.2 | Yes | Yes: William-Liwei/APCL; partial demo/components | Publisher-registered metadata + author code; full final paper unavailable |
| [Spectral-mask method](https://doi.org/10.1177/1088467X251413367) | 2026; Intelligent Data Analysis, OnlineFirst | Yes | Not identified | ABSTRACT-LEVEL; this label is shorthand, not a verified published acronym |
| [R-Clustering](https://doi.org/10.1007/s10618-024-01018-x) | 2024; DMKD | Yes | Yes: jorgemarcoes/R-Clustering | FULL-PAPER publisher HTML + README |
| [RandomNet](https://doi.org/10.1007/s10618-024-01048-5) | 2024; DMKD | Yes | Yes: Jackxiini/RandomNet | FULL-PAPER publisher HTML + README |
| [KASBA](https://doi.org/10.1007/s10618-026-01189-9) | 2026; DMKD | Yes | Yes: aeon + tsml-eval reproduction notebook | FULL-PAPER publisher HTML + aeon 1.1.0 metadata |
| [FASA / MUFASA](https://doi.org/10.1145/3802090) | 2026; PACMMOD/SIGMOD | Yes | Yes: TheDatumOrg/MUFASA | Final method/setup passages + README; FULL-PAPER passage evidence |
| [TS2Vec + k-means](https://ojs.aaai.org/index.php/AAAI/article/view/20881) | 2022; AAAI representation paper | Yes, representation method | Yes: zhihanyue/ts2vec; clustering head is an adaptation | Publisher + official README; original clustering campaign not claimed |
| [TimesURL + k-means](https://ojs.aaai.org/index.php/AAAI/article/view/29299) | 2024; AAAI representation paper | Yes, representation method | Yes: Alrash/TimesURL | Publisher metadata; clustering-head choice must be explicit |
| [TFEC](https://arxiv.org/abs/2601.07550) | 2026; preprint | Acceptance not verified | Code link advertised in preprint; target not resolved here | ABSTRACT-LEVEL; emerging, not mandatory peer-reviewed comparator |

### Data support and backbone

| Method | Univariate? | Multivariate? | Variable length? | Encoder/backbone |
|---|---|---|---|---|
| CKM | No temporal benchmark | No temporal benchmark | N/A to reported fixed-vector experiments | Dense/shallow AE; image, text, tabular representations |
| CC | N/A: images | N/A: images | N/A | ResNet-34 and instance/cluster heads |
| DTC | Yes | UNKNOWN published scope here | UNKNOWN | CNN–BiLSTM AE |
| DTCR | Yes | Not established | UNKNOWN native support | Bidirectional three-layer dilated GRU encoder; single recurrent decoder |
| DTCC-2023 | Yes | Not demonstrated in inspected 10-UCR suite | Fixed dataset lengths | Dilated bidirectional RNN AEs |
| CDCC | Yes | No published UEA evaluation verified | Dataset-specific fixed flattening dimensions | Time BiLSTM; frequency three-block CNN; separate projection heads |
| DEETO | Yes/temporal experiments described | UNKNOWN; archive naming alone is insufficient | UNKNOWN | AE; precise architecture requires full paper |
| TSGCC | Yes | Not demonstrated in reported UCR study | UNKNOWN native handling | Neural representation and SCAN-derived graph/neighbor pipeline; precise final backbone unresolved |
| FCACC | Yes | No original UEA results verified | Padded/centered variable-length handling in code; not a ragged guarantee | Input projection and dilated residual CNN |
| TFMCC | Yes | UEA loader exists; no original UEA results verified | Padded/cropped data pipeline; ragged handling not established | Dilated CNN with hierarchical temporal features |
| Diffusion DTCC | Yes: README specifies UCR | UNKNOWN published multivariate coverage | UNKNOWN | Time/frequency embeddings and shared subspace; exact encoder unresolved |
| UCL-TSC | Yes | Not demonstrated | Fixed lengths in 12-set suite | Residual TCN/multi-view temporal feature fusion |
| DETC | Univariate formulation described | UNKNOWN | UNKNOWN | AE with graph manifold component and evolutionary learning |
| EMTC | Not benchmarked as univariate in this paper | Yes: 15 UEA | UNKNOWN native ragged support | Dilated CNN encoders; MLP reconstruction/transformation decoders |
| APCL | Temporal clustering code/demo | UNKNOWN published evaluation | UNKNOWN | Transformer encoder in released project; full paper needed |
| Spectral-mask method | Temporal data; precise uni count UNKNOWN | UNKNOWN | UNKNOWN | Precise backbone UNKNOWN |
| R-Clustering | Yes | Not demonstrated | Excludes 11 unequal-length UCR datasets | Untrained random convolutional kernels, PCA |
| RandomNet | Yes | Future extension, not original result | Zero-padding in reported full-archive study | Random CNN/LSTM ensembles; no learned deep weights |
| KASBA | Yes | Not original study | Excludes unequal length and missing values | No neural encoder; elastic distance and barycentres |
| FASA / MUFASA | FASA: yes | MUFASA: yes | Resampling/interpolation in paper, not native ragged guarantee | Non-neural shape-distance centroids |
| TS2Vec + k-means | Representation supports uni | Representation supports multi | Masked/padded timestamp processing; specify adaptation | Dilated CNN; full-series pooling |
| TimesURL + k-means | Representation method | Representation method | UNKNOWN verified handling here | SSL time-series encoder; exact implementation settings not inspected |
| TFEC | Not principal scope | Yes, six real-world datasets reported | UNKNOWN | Dual time/frequency representation paths; exact backbone UNKNOWN |

“No UEA evaluation verified” does not mean an implementation cannot accept multiple channels. Conversely, a multi-channel loader or another paper's adapted FCACC results does not establish an original multivariate clustering benchmark.

### Objective and domain structure

| Method | Clustering objective | Contrast? | Time? | Frequency? | Graph/topology? | Joint representation + clustering? | Known k? |
|---|---|---|---|---|---|---|---|
| CKM | Hard latent center distance + reconstruction | No | Not temporal | No explicit branch | No | Yes, after pretraining | Yes |
| CC | Soft-head instance/column discrimination + entropy | Yes | N/A | No | No | Yes, cluster-head learning | Yes |
| DTC | Student-t soft assignment / KL target + reconstruction | No | Yes | No explicit branch | No explicit component | Yes, after pretraining | Yes |
| DTCR | Spectral k-means relaxation + reconstruction + fake-sample classification | No | Yes | No explicit branch | Spectral algebra, not an input graph | Yes, with alternating indicator updates; final k-means | Yes |
| DTCC-2023 | Spectral trace objective + reconstruction + two contrasts/entropy | Yes | Yes | No explicit branch | Spectral relaxation | Alternating indicator updates | Yes |
| CDCC | Time/frequency/cross-domain instance and cluster contrast, balancing entropy | Yes | Yes | Yes, FFT-derived | No graph component | Yes, soft clustering heads | Yes |
| DEETO | Latent representation alignment with eigendecomposition/topological structure | Not established | Yes | UNKNOWN | Yes, topological/eigen structure | Claimed integrated learning; detailed update UNKNOWN | UNKNOWN from accessible final evidence |
| TSGCC | Graph-weighted instance and cluster contrast | Yes | Yes | UNKNOWN | Weighted kNN graph | Publication claims integrated learning; released pretext/SCAN/self-label stages | Yes, released configured head |
| FCACC | Fuzzy membership-weighted center distance + cluster-aware contrast | Yes | Yes | Not an explicit frequency branch | No graph objective verified | Alternating membership/center refinement and neural learning | Yes |
| TFMCC | Multi-level temporal, instance and cluster contrast with entropy | Yes | Yes | Yes, wavelet augmentation | No graph objective verified | Yes, soft-head clustering | Yes |
| Diffusion DTCC | Reliable cross-view cluster diffusion, contrast and column orthogonality | Yes | Yes | Yes, FFT | Neighbor/local + global diffusion structure | Claimed integrated learning; full update details UNKNOWN | UNKNOWN final protocol; likely head parameter is not treated as verified |
| UCL-TSC | Contrast, clustering and regularization with neighbor pseudo-relations | Yes | Yes | Not established | Neighbor relations; explicit graph module UNKNOWN | Yes, described objective | Cluster count configured |
| DETC | Evolutionary representation search + graph manifold/clustering optimization | Not established | Yes | UNKNOWN | Yes | Coupled/evolutionary stages, not assumed one gradient loop | Cluster count configured in described formulation |
| EMTC | Evolving variate masks; intra/inter-view reconstruction; clustering-guided contrast | Yes | Yes | Frequency mask alternative is an ablation, not its defining branch | Not an explicit input graph | Iterative cluster-guided representation training | True g supplied in benchmark |
| APCL | Adaptive prototypes and assignment/contrastive learning | Yes | Yes | UNKNOWN | UNKNOWN | Code connects prototype and representation learning; full-paper protocol UNKNOWN | Author claims adaptive MDL k; code supports selection, not an oracle-k benchmark proof |
| Spectral-mask method | Spectral masking + hierarchical contrast + fuzzy loss | Yes | Yes | Yes | UNKNOWN | Joint loss claimed at abstract level | UNKNOWN |
| R-Clustering | Post-transform Euclidean k-means | No | Yes | No explicit branch | No | No trained representation | Yes |
| RandomNet | k-means and ensemble consensus over random embeddings | No | Yes | No explicit branch | HBGF ensemble graph, not learned temporal graph | No trained representation | Yes |
| KASBA | MSM elastic k-means, stochastic barycentres, triangle-inequality pruning | No | Yes | No | No | N/A: direct data-space clustering | Yes |
| FASA / MUFASA | SBD/SBD-D with efficient centroid update | No | Yes | FFT for distance, not a learned frequency-view encoder | No | N/A: no learned neural representation | Supplied k required; oracle-count rule unresolved in inspected setup |
| TS2Vec + k-means | SSL representation then chosen k-means head | Yes, temporal/instance | Yes | No defining branch | No | No, two-stage adaptation | Yes for chosen head |
| TimesURL + k-means | SSL representation then chosen head | Yes | Yes | Frequency augmentation reported | No graph verified | No, two-stage adaptation | Yes for chosen head |
| TFEC | Dual-domain preservation/co-enhancement and clustering distribution | Yes | Yes | Yes | UNKNOWN | Claimed integrated framework | UNKNOWN |

### Benchmark and reporting fields

NR = not reported in inspected final main material; UNKNOWN = inaccessible/unverified. These are not interchangeable.

| Method | Datasets used; # UCR; UEA/multi | Metrics verified | Runs | Mean/std or CI | Statistical tests | Runtime/memory evidence | Primary contribution |
|---|---|---|---|---|---|---|---|
| CKM | MNIST, USPS, 20News and 10 UCI; UCR 0, UEA 0 | Clustering benchmarks; exact metric conventions not extracted here | UNKNOWN | UNKNOWN | UNKNOWN | No comparable temporal efficiency study | Concrete latent k-means |
| CC | Six image corpora; UCR 0, UEA 0 | ACC, NMI, ARI | UNKNOWN | UNKNOWN | UNKNOWN | Training GPU-hours; not a TS scaling comparison | Instance/cluster contrastive heads |
| DTC | Earthquake/spacecraft sensor examples; exact UCR count UNKNOWN | UNKNOWN from inspected preprint abstract | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Temporal AE and clustering layer |
| DTCR | 36 UCR main + separate StarLightCurves case; UEA 0 | RI, NMI; ACC in supplement | **5** | Mean ± SD | Wilcoxon signed-rank, p<.05; correction not verified | Architecture discussion/hardware; standardized scaling not verified | Reconstruction + spectral clustering + fake samples |
| DTCC-2023 | **10 UCR**, UEA 0 | RI, NMI | NR in inspected text | Point tables; uncertainty NR | NR in inspected text | Comparable scaling/memory NR | Dual contrast in temporal AE clustering |
| CDCC | **40 UCR**, UEA 0 | RI, NMI | NR; runner has one default seed | Point tables; SD/CI NR | Wilcoxon + Bonferroni; Nemenyi | Hardware described; common quality–runtime frontier NR | Cross-domain contrast |
| DEETO | UCR/UEA repository wording; exact evaluated counts UNKNOWN | UNKNOWN verified protocol | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Topological/eigen latent alignment |
| TSGCC | **128 UCR** reported, 36 highlighted main cases; UEA 0 verified | ACC/ARI/NMI/RI in released evaluation; final metric convention incomplete | UNKNOWN | UNKNOWN | UNKNOWN full paper | UNKNOWN full paper | Graph-weighted contrastive positives |
| FCACC | **40 UCR**, UEA 0 original paper | NMI, RI | NR; single default seed is code evidence only | Point tables; SD/CI NR | Pairwise p-values shown; test family/correction not stated in inspected passages | Hardware described; common runtime/memory benchmark NR | Fuzzy cluster awareness and pair selection |
| TFMCC | **40 UCR**, UEA 0 original paper | NMI, RI in paper; code also ACC, ARI, FMI | NR; runner seeds=[666], run_times=[1] | Point tables; SD/CI NR | NR in inspected main text | Hardware and code timer; formal frontier NR | Time-frequency multi-level contrast |
| Diffusion DTCC | **26 UCR according to author repository's dataset description**; paper reports 26 datasets/14 domains; UEA not verified | Exact final metrics UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Reliable cross-view diffusion |
| UCL-TSC | **12 UCR**, UEA 0 | ACC, NMI, purity, F-score, RI | Unspecified multiple experiments | Loss SD in Table 4; clustering-score SD not established | Formal test NR in inspected sections | Training/convergence illustrations; quantitative frontier NR | Temporal multiview fusion/contrast |
| DETC | **15 public temporal datasets**; UCR/UEA breakdown UNKNOWN | NMI discussed; complete set UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Evolution + graph manifold learning |
| EMTC | UCR 0; **15 UEA** | ACC, F1, NMI, ARI | NR in inspected main text | Mean ± SD table entries | Main paper says BD-test at 95%/99%; full tests in linked extended version, not inspected | Efficiency study announced; full details partly outside extracted main passages | Evolving variate masking and cluster-guided views |
| APCL | Final dataset counts UNKNOWN; released synthetic demo is not a UCR benchmark | UNKNOWN final benchmark | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Adaptive prototype clustering |
| Spectral-mask method | Counts and exact collections UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Adaptive frequency masking + hierarchical contrast |
| R-Clustering | **117 fixed-length UCR**, UEA 0 | ARI | **10** | Repeated means; exhaustive per-score SD table not established | Friedman plus Wilcoxon described; exact post-hoc correction unresolved here | Explicit efficiency/scalability | Fast untrained random features |
| RandomNet | **128 UCR**, 20 tuning cases excluded from **108** main test aggregate; UEA 0 | RI primary | Core repeats UNKNOWN; noise/scalability experiments 10 runs | Repeated robustness/efficiency evidence; full accuracy-SD policy unresolved | CD/statistical comparisons | Explicit L/N scalability and efficiency | Random networks and ensemble consensus |
| KASBA | **112 eligible UCR**; particular all-complete comparisons 102 or 93; UEA 0 | CLACC, ARI, mutual-information metrics; precise four main figure labels not all transcribed; RI only in separate reproduction | Main count unresolved; matched k-Shape reproduction **10** | Aggregates; core per-seed SD policy unresolved | Wilcoxon + Holm cliques; some heatmap p-values explicitly unadjusted | Explicit total/runtime comparisons; no neural GPU-memory benchmark | Fast elastic clustering |
| FASA / MUFASA | FASA **128 UCR**; MUFASA **30 UEA**, original and separately downsampled comparisons | RI primary; ARI/NMI also discussed | **10** | Mean reported; per-score SD policy not extracted | Wilcoxon; Friedman/Nemenyi | Explicit length scalability and runtime; neural runtimes measured on CPU | Fast shape-centroid updates and cross-channel alignment |
| TS2Vec + k-means | Original **125 UCR/29 UEA are classification counts**, not clustering counts | Adaptation should use common clustering metrics | Must set new run count | Must generate new uncertainty | New paired analysis needed | Representation paper ≠ common clustering frontier | Strong two-stage representation control |
| TimesURL + k-means | Original representation benchmark; exact counts not verified here | New clustering metrics required | Must set new count | New uncertainty required | New paired analysis | UNKNOWN comparable clustering frontier | Stronger SSL alternative, optional in minimum |
| TFEC | Six multivariate real-world datasets; UCR/UEA breakdown UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | Emerging dual-domain multivariate clustering |

The first six modern direct competitors cannot all be declared “replication ready.” “Mandatory baseline” here means mandatory scientific comparison or an explicitly documented reproduction barrier that restricts the eventual claim. A missing faithful comparator cannot be replaced by its abstract's advertised performance or silently removed while retaining universal superiority language.

## Benchmark Protocols

### Exact selections recovered

Dataset names below are canonicalized against the [official UCR 2018 summary](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv). This metadata validates names and lengths, not the finite/missingness status of downloaded arrays. Dataset files were not downloaded or changed in this audit.

**Historical LoSTer 17:** CinCECGTorso, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, StarLightCurves, UWaveGestureLibraryX, UWaveGestureLibraryY, MixedShapesRegularTrain, MixedShapesSmallTrain, EOGVerticalSignal, SemgHandMovementCh2, ECG5000, OSULeaf, Symbols, MiddlePhalanxOutlineAgeGroup, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW, SyntheticControl.

**CDCC 40, final Table 1:** ACSF1, Adiac, ArrowHead, Beef, Car, CBF, CricketX, CricketY, CricketZ, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistalPhalanxTW, ECG200, ECGFiveDays, EOGVerticalSignal, FaceFour, FiftyWords, Fish, Fungi, GunPointMaleVersusFemale, GunPointOldVersusYoung, HouseTwenty, LargeKitchenAppliances, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxTW, OSULeaf, PigAirwayPressure, PigArtPressure, PigCVP, Plane, PowerCons, ProximalPhalanxTW, SemgHandMovementCh2, ShapeletSim, ShapesAll, SwedishLeaf, SyntheticControl, ToeSegmentation1, Trace, WordSynonyms.

**FCACC 40, final Tables 1–2:** ACSF1, AllGestureWiimoteX, AllGestureWiimoteZ, BirdChicken, Car, CricketX, CricketZ, DistalPhalanxTW, ECGFiveDays, FaceAll, FacesUCR, Fungi, GesturePebbleZ1, GesturePebbleZ2, GunPointOldVersusYoung, HouseTwenty, InsectEPGRegularTrain, InsectEPGSmallTrain, ItalyPowerDemand, LargeKitchenAppliances, Lightning2, Lightning7, Mallat, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, PickupGestureWiimoteZ, PigAirwayPressure, PigCVP, Plane, ProximalPhalanxOutlineAgeGroup, SemgHandMovementCh2, ShapeletSim, ShapesAll, SmoothSubspace, Symbols, SyntheticControl, ToeSegmentation1, ToeSegmentation2, TwoPatterns.

**TFMCC 40, final table abbreviations checked against the author runner's union_list:** CricketZ, ToeSegmentation2, GestureMidAirD3, FaceAll, ShapeletSim, Lightning7, FordB, ECGFiveDays, ShakeGestureWiimoteZ, Symbols, SonyAIBORobotSurface2, ProximalPhalanxTW, GunPointOldVersusYoung, MiddlePhalanxOutlineAgeGroup, Beef, SyntheticControl, WordSynonyms, ShapesAll, FacesUCR, SmoothSubspace, SwedishLeaf, TwoPatterns, MelbournePedestrian, CricketX, MoteStrain, GesturePebbleZ1, AllGestureWiimoteX, GesturePebbleZ2, OSULeaf, DodgerLoopDay, CBF, AllGestureWiimoteY, AllGestureWiimoteZ, SonyAIBORobotSurface1, Meat, CricketY, Car, FaceFour, FiftyWords, PickupGestureWiimoteZ.

**DTCC-2023 10, final Table 1:** Beef, DistalPhalanxOutlineAgeGroup, ECG200, ECGFiveDays, Meat, MoteStrain, OSULeaf, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW.

**UCL-TSC 12, publisher Table 1:** ECG200, EOGHorizontalSignal, InsectEPGRegularTrain, Fish, Herring, ProximalPhalanxOutlineAgeGroup, GunPointOldVersusYoung, Lightning7, DodgerLoopWeekend, Plane, Coffee, Meat.

**EMTC 15 UEA:** BasicMotions, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, FingerMovements, HandMovementDirection, Heartbeat, MotorImagery, NATOPS, PEMS-SF, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, StandWalkJump.

**Unrecovered exact selections:** DEETO's experiment list/count; TSGCC's highlighted 36 list; diffusion DTCC's 26 list; DETC's 15 list; APCL's final benchmark list; spectral masking's final list. The count “128” in a repository configuration is not by itself evidence that a publication evaluated 128: CDCC's data_config covers more datasets than its published 40. TSGCC's full-archive claim comes from its publisher description, not from treating its Fish-specific runner as a full-archive reproduction.

### Main protocol comparison

| Method | TRAIN + TEST pooling | Normalization | k and label use | Fixed length / missingness | Seeds, metrics and inference caveat |
|---|---|---|---|---|---|
| Historical LoSTer | Yes, transductive pooled clustering | Per-series z-score | True class count supplied; labels evaluate metrics | Fixed-length finite-input assumption | Five seeds 0–4 stated; RI and arithmetic NMI in code; old per-run provenance unavailable |
| FASA / MUFASA | Pooled splits | z-score; resampling to dataset max length and interpolation of missing values | Supplied k; exact class-count/label-selection rule unresolved in inspected setup | Preprocessed fixed-length arrays; UEA original/downsampled tracks must remain separate | Ten seeds; RI and paired/rank tests; CPU-only runtime comparison is not LoSTer's GPU contract |
| DTCR | **Main paper trains TRAIN and evaluates TEST**; separate large case | Exact normalization not extracted | Known k; fake labels are generated, not class supervision | Main fixed-length suite | Five means/SD; final k-means after spectral training; must not compare to pooled LoSTer as identical protocol |
| DTCC-2023 | Released script concatenates splits | Exact loader normalization unverified | Known k; released script tracks best NMI/RI during epochs | Fixed lengths | Script includes an ablation-oriented configuration; its best-label-metric tracking is not proof that the final publication used that selection policy |
| CDCC | Yes, final description and code | Paper says normalization; inspected loader uses a **global merged-pool mean/std**, not per-series z-score | Known k; labels used for validation metrics; grid-search selection criterion not fully established | All 40 have fixed numeric lengths in UCR metadata | Paper runs NR; runner resets seed 2333; RI/NMI and corrected paired tests |
| DEETO | UNKNOWN | UNKNOWN | UNKNOWN; labels-only-for-evaluation not established | UNKNOWN | Protocol recovery requires final full paper |
| TSGCC | Full-paper pooling UNKNOWN; released Fish loader retains split objects | UNKNOWN full-paper policy | Head k configured; evaluation labels and Hungarian matching in code; selection policy unresolved | Full 128 includes variable length; handling needs verification | Multi-stage pretext/neighbor/SCAN/self-label workflow; full final protocol unavailable |
| FCACC | Yes, author batch runner | Default CSV path separately normalizes global TRAIN and TEST values for a specified non-normalized list; alternative loader differs | k from label count; labels supplied to training/evaluation interfaces; search-selection policy unresolved | Includes padded unequal-length cases | Code default seed 1127; published seed count NR; NMI/RI; do not assume labels never affect selection |
| TFMCC | **Yes, author train.py explicitly concatenates both splits** | For a specified non-normalized list, global TRAIN mean/std applied to TEST; otherwise existing archive values | k from TRAIN label count; merged labels passed into fit/evaluation; selection use needs source audit | Includes unequal-length/NaN-padded cases; max training length/cropping configuration exists | Runner one seed/one repeat; source logs epoch metrics; publication's complete seed/selection policy unresolved |
| Diffusion DTCC | UNKNOWN | UNKNOWN; processed CSV archive provided | README says grid search; oracle-k and label-selection rules unresolved | Exact dataset list/handling UNKNOWN | PyTorch release and data link verified; paper/table protocol not recovered |
| UCL-TSC | Publisher §4.1 merges TRAIN and TEST | Exact transform unresolved | Cluster count configured; pseudo-relations used; labels-only selection not established | 12 numeric fixed-length datasets | Loss SD ≠ clustering SD; no verified seed count |
| DETC | UNKNOWN | UNKNOWN | Cluster count in formulation; search/label-use details UNKNOWN | UNKNOWN | 15 datasets is not automatically 15 UCR |
| EMTC | Pooling UNKNOWN; Table 1 sizes match TRAIN sizes for several UEA sets, which alone does not prove training-only protocol | UNKNOWN | True g listed; per-dataset parameter optimization stated; selection criterion unresolved | Fixed lengths listed, some extremely long/many channels | Mean/SD shown; exact run count and complete supplement tests not inspected |
| APCL | UNKNOWN final campaign | UNKNOWN | Adaptive k reported by author; cannot silently equate with oracle-k protocols | UNKNOWN | Synthetic demo/components insufficient to reproduce final tables |
| R-Clustering | Exact original fit/evaluate policy unresolved in inspected passages | Exact policy unresolved here | Known k; hyperparameter search documented; held-out selection details need manifest | 117 excludes unequal-length cases | Ten centroid-seeded repeats; ARI; published values are not a pooled-LoSTer comparator without protocol matching |
| RandomNet | Yes | Exact normalization unresolved here | Known k; **20 datasets explicitly used for parameter selection**, 108 test aggregate | Zero-padding for unequal lengths | RI; separates tuning and test tasks; core repeat count unresolved |
| KASBA | **Primary TRAIN fit/TEST predict**, pooled analysis separately | z-normalization | True class count; paper explicitly says labels otherwise only evaluate | 112 excludes unequal length/missing values | Complete-case totals differ by comparison; do not transplant split-based scores to pooled analysis |

These differences are scientific, not file-format details. Pooled transductive clustering can be legitimate when declared; it estimates a different task from inductive prediction on unseen TEST cases. Recent KASBA results explicitly show that pooled versus held-out evaluation can change statistical conclusions. Label-derived k is already oracle information even if no class labels enter a gradient.

For a new common campaign, predeclare pooled TRAIN+TEST **as the primary task** to match LoSTer's intended clustering setting and several modern direct papers. Use one normalization contract, preferably per-series z-normalization with an explicit constant-series rule. Rerun every model on the same record IDs. Label-based epoch selection or target-dataset hyperparameter tuning is disallowed for the principal unsupervised claim. Original author normalization can be a small sensitivity study, not a hidden method-specific advantage. Strong-tier held-out evaluation is a separate track; it cannot be averaged together with pooled results.

### Intersections with the historical 17

| Competitor collection | Intersection size | Exact overlap or evidence limit |
|---|---|---|
| CDCC 40 | **6** | EOGVerticalSignal, SemgHandMovementCh2, OSULeaf, MiddlePhalanxOutlineAgeGroup, ProximalPhalanxTW, SyntheticControl |
| FCACC 40 | **5** | SemgHandMovementCh2, Symbols, MiddlePhalanxOutlineAgeGroup, ProximalPhalanxOutlineAgeGroup, SyntheticControl |
| TFMCC 40 | **5** | OSULeaf, Symbols, MiddlePhalanxOutlineAgeGroup, ProximalPhalanxTW, SyntheticControl |
| DTCC-2023 10 | **3** | OSULeaf, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW |
| UCL-TSC 12 | **1** | ProximalPhalanxOutlineAgeGroup |
| TSGCC reported full 128 | **17 under its full-archive coverage claim** | Exact highlighted-36 overlap UNKNOWN; full execution list not recovered |
| FASA full 128 | **17 under full-archive coverage** | Actual common input preprocessing must be checked |
| MUFASA 30 UEA | **0 identical archive tasks** | All 15 EMTC tasks fall within its reported full-UEA coverage; do not compare original versus downsampled results |
| RandomNet full 128 | **17** | Overlap with its final 108 test tasks UNKNOWN until its 20 tuning IDs are checked |
| R-Clustering fixed-length 117 | **17 by its stated unequal-length exclusion rule** | Historical names all have fixed numeric lengths; actual matched data manifest still required |
| KASBA eligible 112 | **UNKNOWN exact membership here** | Historical 17 are fixed-length, but missingness eligibility was not verified from arrays; do not infer full intersection from length alone |
| DEETO / diffusion DTCC / DETC / APCL / spectral-mask method | **UNKNOWN** | Exact final lists unavailable; guessed intersections would be invented |
| EMTC 15 UEA | **0 identical archive tasks** | UEA Cricket is not UCR CricketX/Y/Z; multivariate/univariate task names are not interchangeable |
| TS2Vec / TimesURL original representation suites | **Not a clustering intersection** | Original classification coverage is not a matched clustering evaluation |

CDCC, FCACC and TFMCC's 40-dataset selections have a **70-dataset union**. Equal sample counts do not make their macro-averages comparable. Even a common intersection is not sufficient without matching pooling, normalization, stopping, k, metrics and selection rules. No numerical cross-paper superiority calculation is made in this report.

The eligible [2025 PVLDB comparative study](https://paparrizos.org/papers/PaparrizosVLDB25.pdf) evaluates 84 methods on 128 tasks and explicitly audits reproducibility, statistical validation and scalability. It reinforces the need for strong non-deep baselines; its headline about an illusion of progress is a result under its own configuration/metric choices, not a theorem adopted here. KASBA's later discussion shows why normalization, restarts, pooled versus held-out data and RI versus adjusted metrics can change comparative conclusions.

## Is the 17-Dataset Benchmark Still Adequate?

**For the historical broad claims: INADEQUATE. For a newly delimited mechanistic study: WEAK BUT DEFENSIBLE.** These are scope-dependent judgments, not a universal editorial rule that 17 papers cannot be published.

Evidence supporting expansion is CDCC/FCACC/TFMCC's independent 40-set campaigns, TSGCC's reported 128, R-Clustering's 117, RandomNet's 128/108 evaluation distinction, and KASBA's 112 eligible tasks. FASA's 128-UCR and MUFASA's 30-UEA studies strengthen this evidence. DTCR already evaluated 36 plus a long large case in 2019. Evidence against a simplistic “everyone must run 128” rule is UCL-TSC's 12 UCR, DETC's 15 public datasets and EMTC's 15 multivariate datasets. DEETO's exact count remains UNKNOWN and is not used to manufacture a benchmark-size argument.

The inherited 17 are selected, partially overlapping tasks and include related families: paired fetal ECG sets, paired UWave axes, related phalanx datasets and regular/small TRAIN versions of MixedShapes. Correlated tasks do not offer 17 independent application domains. Since the historic selection rationale and raw runs are not recoverable, selecting only favorable historical tasks for a new claim would compound selection uncertainty.

**Recommendation:** CDCC's 40 as a predeclared fixed-length core, checked for finite values before any authorized future execution. It spans device, ECG, EOG, motion, image-outline, simulated, spectrographic and hemodynamic sources; some are classification encodings rather than naturally sampled physical signals, so “40 real-world temporal processes” would be an overclaim. Add CinCECGTorso, StarLightCurves, MixedShapesRegularTrain and MixedShapesSmallTrain for **44** tasks. This adds only 10% to the task count while strengthening historical/length continuity. Related MixedShapes tasks must be identified as one family in robustness interpretation.

The CDCC core lengths are **60–2000**. Per-dataset pooled counts must be verified from the canonical data manifest before execution. The 44 include StarLightCurves with **9236** pooled cases. All 40 core names have fixed numeric lengths in official UCR metadata; absence of missingness is not established by that summary alone.

A 20-task compromise would need an explicitly stratified rationale and would remain weak for broad comparative efficiency claims. A 40/44 campaign plus direct modern comparators and uncertainty provides a better evidence/workload tradeoff than preserving 17 and adding many forecast backbones. Full 128 requires additional unequal-length/missing-data decisions and is not the minimum credible revision.

## Long-Sequence Positioning

Long-input cost and representation quality are recognized issues across clustering research. This audit did **not** verify a widely adopted separate “long-sequence clustering” task definition, dedicated standardized archive, or agreed cutoff. A failure to locate such a benchmark is a search finding, not a proof none exists.

| Prior evidence | Length/scale information | Implication |
|---|---|---|
| Historical LoSTer | UCR **60–1639**; saved retail arrays around 1684/1913 | Its 17 tasks are not uniformly long; 60, 80 and 140 cannot all be labeled long without a stated relative criterion |
| DTCR | Separate StarLightCurves, **L=1024, N=9236**, already in 2019 | Large/long temporal clustering examples predate LoSTer |
| CDCC core | **L=60–2000** | Modern direct competitors already include longer inputs than the historic UCR maximum |
| FCACC and TFMCC | Some unequal-length cases; short tasks and long fixed cases coexist | Their “40” is not a dedicated long-sequence benchmark |
| RandomNet | Controlled scalability reported for **L=1000–10000** and **N=200–10000** | Direct efficiency/scaling prior; LoSTer cannot claim first scalable long-input clustering |
| EMTC | EigenWorms **L=17984**, MotorImagery 3000, StandWalkJump 2500 | Long multivariate clustering is already evaluated; channel and batch cost also matter |
| FASA | Explicit shape-centroid length scalability | An additional 2026 efficiency comparator, not just a multivariate paper |
| KASBA | Explicit broad-archive efficiency, elastic-distance and collection-size analysis | Speed claims must include strong non-neural alternatives |

The final recommended 44 have **11 tasks with L≥1024**: the seven CDCC core tasks ACSF1, EOGVerticalSignal, HouseTwenty, PigAirwayPressure, PigArtPressure, PigCVP, SemgHandMovementCh2, plus the four additions. The threshold is an audit's predeclared profiling stratum, not an established field definition. Full-cycle timings should be recorded for all 44, with detailed memory/throughput and time–quality traces on those 11. Include short tasks to show where dense/global modeling helps or fails.

Controlled length scaling should separate L, N, k, width and batch size. Synthetic resampling/concatenation/noise can change the clustering task; specify generation and do not count these conditions as extra UCR datasets. Full-length data must reach every comparator, with declared padding/cropping rules and time/memory limits. Unsupported/OOM cases are censored outcomes, not an excuse to secretly shorten one baseline.

**Position choice:** C is strongest if measurements establish a useful quality–cost frontier. D is a supporting mechanistic angle. B can remain a secondary regime, but no length-specific architectural mechanism has been established. A requires broad fresh comparative accuracy evidence. If the hard/contrastive additions lose to simpler CKM or random features, report the negative result and adopt a narrower component-study narrative.

## Transformer Baselines

| Method | Verified publication | Actual task and structural relevance | Role for a new clustering paper |
|---|---|---|---|
| Preformer | ICASSP **2023**, not AAAI; DOI in bibliography | Forecasting; multiscale segment correlations | **FORECASTING BACKBONE ADAPTED TO LOSTER** in inherited scripts; optional, not clustering SOTA |
| Pathformer | ICLR 2024 | Forecasting; multiscale patches and adaptive pathways | Same adapted-backbone category; inherited truncation to multiples of 96 invalidates a full-sequence comparison |
| iTransformer | ICLR 2024 | Forecasting; **variate tokens**, not one token per time point | Adapted backbone; a univariate one-variate token does not test cross-variate attention |
| PatchTST | ICLR 2023 | Forecasting/representation; patches and channel independence | Preferred single efficient controlled backbone if architectural comparison is retained |
| FEDformer | ICML 2022 | Forecasting; frequency decomposition and selected modes | Useful counterexample to universal dense quadratic-attention rhetoric; optional adaptation |
| LiteTransformer | ICLR 2020 | **NLP/mobile Transformer**, not a native temporal forecaster or clusterer | Task-distant adaptation; no scientific requirement to implement it |

All six are **not direct time-series clustering baselines in their original publications**. LiteTransformer falls outside the user's two-way forecasting/direct-clustering taxonomy and should be labeled explicitly. Their original forecast loss, datasets and rankings do not establish clustering performance. A clustering adapter using LoSTer's loss tests a backbone within a controlled training system; it is not the original method's published clustering algorithm.

**Minimum: zero new forecasting adapters. Strong tier: one PatchTST adapter**, trained with a matched clustering head/objective and declared treatment of positions, input length, capacity, batch size and stopping. Multiple adapters are justified only by a specific architectural research question; adding Preformer, Pathformer, iTransformer, PatchTST and FEDformer is redundant for a focused efficiency paper.

APCL is independently relevant as a direct clustering method with a released Transformer encoder; a forecast adapter cannot replace it in a prototype-centered study. Its partial artifact/full-paper limitations still apply.

The previous review history explains why the inherited experiments exist, but it does not impose scientific requirements on a new submission elsewhere. The immutable experiments can be documented without being repeated wholesale. Plain attention without positions is permutation equivariant at token outputs, not proof that an entire positional/patch/spectral Transformer discards time order. Dense weights themselves are position-specific, but that does not establish shift robustness or superiority at temporal dependence learning. Tiny batches can affect contrastive learning, yet causal claims need equal-batch and maximum-feasible-batch measurements, not inference from one failed adaptation.

## Multivariate Scope

**Verdict: UNNECESSARY FOR A CLEAR UNIVARIATE SCOPE.** CDCC, FCACC and TFMCC publish univariate UCR experiments; R-Clustering, RandomNet and KASBA likewise provide substantial univariate evidence. A carefully delimited univariate journal paper remains scientifically legitimate.

Multivariate clustering is nevertheless established. MUFASA evaluates all 30 UEA datasets, with original and downsampled tracks explicitly distinguished. EMTC publishes 15 UEA tasks and evaluates long/high-dimensional cases, while TFEC's six-dataset preprint illustrates continuing dual-domain multivariate work without verified acceptance. An original FCACC UEA result must not be inferred from EMTC adapting FCACC as a comparator. DEETO's mention of the “UEA & UCR repository” does not establish actual multivariate experiments.

Adding multivariate evaluation would be **OPTIONAL** for the recommended focused submission and **STRONGLY DESIRABLE** if it seeks general temporal-clustering applicability. It is essential only for claims about multivariate mechanisms or cross-variable dependencies. Broadening tensor input alone is not automatically new science; no multivariate implementation is designed here.

The ideal tier's 15 UEA set is the exact EMTC list above, not an arbitrarily convenient handful. That tier is conditional on a separately specified/validated extension and complete comparator protocols. It must not be portrayed as an experiment already supported by the current univariate LoSTer implementation.

## Baseline Reproducibility and Cost

These estimates concern engineering/reproduction effort, not GPU-hours. LOW does not mean computationally cheap: DTW barycentres can be straightforward to call but slow. No dependency was installed, model imported, script trained, or hardware benchmark run. Compatibility is inferred from declared pins and APIs, not empirically tested.

### Direct and representation methods

| Candidate | Official artifact; framework and versions | Compatibility with declared LoSTer pins | Loader / weights | License verified | Effort and main barrier |
|---|---|---|---|---|---|
| CKM | Official author code not identified; inherited local implementation is PyTorch | Local code is near existing stack, but scientific equivalence not validated | Local UCR loader has known row-loss defect; no external weights | Official license UNKNOWN | **MEDIUM** for a clearly identified faithful reproduction/matched control; cannot call it author code |
| CC loss controls | Author PyTorch loss source, exact Python/torch pins not extracted | Simple loss operations plausible under 1.13; no compatibility test | Image loaders irrelevant; no pretrained image encoder needed for a temporal loss ablation | MIT | **LOW** for audited loss controls; **HIGH** for an unnecessary complete image-pipeline adaptation |
| DTC | No verified author code; FlorentF9 third-party Keras implementation | Legacy Keras/TF differs from PyTorch; versions not verified | Third-party temporal loader; external weights not established | UNKNOWN | **HIGH**; disclose reproduction rather than treating code as official |
| DTCR | qianlima-lab/DTCR, **Python 3.6, TensorFlow 1.6** | Requires isolated legacy stack; incompatible with current PyTorch environment | UCR files; training from scratch | UNKNOWN | **HIGH**; old graph APIs and reproducibility. RandomNet paper also reports difficulty reproducing it, which is evidence of one attempt, not universal failure |
| DTCC-2023 | 07zy/DTCC, TensorFlow 1-style APIs including tf.contrib; Python pin not verified | Separate legacy environment necessary | UCR loader; active script appears ablation-oriented and tracks best label metrics; scratch pretraining | No license found in inspected tree | **HIGH**; reconstruct full method/config and remove oracle selection only as a disclosed common evaluation protocol |
| CDCC | Author PyTorch **1.10.2+cu113**; requirements also old NumPy/pandas/sklearn and Keras | PyTorch family close; direct blanket downgrade would conflict with LoSTer pins | Compressed CSV loader, pooled global normalization; train from scratch | No license found | **MEDIUM**; canonical TSV adapter and frozen data/selection contract |
| DEETO | No official repository identified; framework/versions UNKNOWN | UNKNOWN | Loader/pretrained requirements UNKNOWN | UNKNOWN | **VERY HIGH** for faithful full reproduction until final methods/artifact recovered; topology details must not be invented |
| TSGCC | Author repository; **Python 3.7.7, PyTorch 1.4.0, CUDA 10**, legacy Linux package builds | Separate environment; old torchvision/SCAN dependencies | Fish-specific data/configs, neighbor indices and stage-generated pretext checkpoints required; no verified external pretrained model requirement | **CC BY-NC 4.0**, inspected LICENSE; API's NOASSERTION is not “no license” | **HIGH**; requirements.txt is actually Conda YAML, GPU IDs hard-coded, full-128 runner absent |
| FCACC | Author README **Python 3.8.10, torch 1.10.0+cu113**, NumPy 1.21.4, pandas 2.0.3, sklearn 1.3.2, SciPy 1.10.1 | Isolated reproducible environment; incompatible pandas pin with LoSTer | UCR CSV/TSV utilities and batch runner; scratch training, padding helpers | No license found | **MEDIUM**; normalize/align rows consistently; default loader differs from alternative |
| TFMCC | Author README **Python 3.9**, torch==**2.8.1**, scipy==1.6.1, numpy==**1.26**, pandas==**2.23** as printed | Suspicious/incomplete pins; do not silently correct them or assume a working environment | Canonical headerless UCR TSV and UEA ARFF loaders; merged training; scratch training | No license found | **HIGH until environment pins are clarified**, then plausibly MEDIUM; README CSV wording differs from TSV source |
| Diffusion DTCC | Verified qinqinhan/DTCC; README Python ≥3.6, **torch 1.10.2+cu113** | Family close but not tested; isolate | Processed compressed UCR CSV data link; grid search; pretrained requirements UNKNOWN | No license found | **HIGH**; exact 26 IDs and final protocol still need recovery |
| UCL-TSC | No official code identified; final sections **Python 3.8, PyTorch**, exact version UNKNOWN | Likely feasible but unverified | No verified author loader/weights artifact | UNKNOWN | **HIGH** manual reproduction and ambiguous accuracy-uncertainty reporting |
| DETC | Publisher-linked anonymous.4open.science artifact; contents/framework UNKNOWN | UNKNOWN | Loader/weights UNKNOWN | UNKNOWN | **HIGH / VERY HIGH** until artifact and evolutionary update protocol inspected |
| EMTC | Author repo and final paper; **PyTorch 1.8.0**, Python pin UNKNOWN | Separate prior-stack environment; multivariate extension is additional research, not compatibility work | UEA-oriented artifact; complete loader/weights contract not inspected | UNKNOWN | **HIGH** for a faithful 15-UEA campaign and LoSTer extension |
| APCL | Author MIT repo; **Python ≥3.9**, torch ≥2.0, numpy ≥1.24, sklearn ≥1.3 | Newer torch/numpy than LoSTer; isolate | Synthetic demo and components; no verified full-UCR campaign, external weights requirement not established | MIT | **VERY HIGH** for final-table reproduction until full paper/protocol recovered; **MEDIUM** only for a clearly labeled controlled component study |
| Spectral-mask method | No official repo identified; framework/versions UNKNOWN | UNKNOWN | Loader/weights UNKNOWN | UNKNOWN | **VERY HIGH** without accessible full method/artifact |
| R-Clustering | Author notebook; Python/Numba/NumPy/pandas/sklearn/sktime, versions unpinned in README | Most numeric dependencies close; sktime not in declared LoSTer requirements | UCR experiment notebook; random kernels, no pretrained weights | **GPL-3.0** repository metadata | **LOW / MEDIUM**; inspect notebook preprocessing/selection before executing |
| RandomNet | Author repo; **Python 3.8, TensorFlow 2.1**, PyMetis | Separate TF stack; PyMetis/Linux assumptions add Windows friction | UCR workflow, random weights, no external pretrained weights | UNKNOWN | **HIGH** on this Windows environment; MEDIUM on a validated author-compatible Linux environment |
| KASBA | aeon **1.1.0**, Python **≥3.9,<3.14**, NumPy ≥1.21,<2.3, pandas ≥2,<2.3, SciPy ≥1.9,<1.16, sklearn ≥1,<1.7, Numba ≥.55,<.62 | pandas pin differs; isolate rather than change LoSTer environment | aeon time-series loaders; tsml-eval reproduction notebook; no weights | BSD-3-Clause toolkit | **MEDIUM**; align pooled versus held-out task and pin implementation/parameters |
| FASA / MUFASA | Official TheDatumOrg/MUFASA; **Python 3.10**; numeric/Numba pipeline, UTS requirements aeon .6.0, tslearn .6.3 | Isolate from LoSTer; MTS requirements include NumPy 1.25.2, pandas ≥2.0.3 and SciPy ≥1.15 | Public processed archives and runners; no pretrained neural weights | No repository license found; article license is not code license | **MEDIUM**; verify preprocessing and choose method-specific dependencies rather than all benchmark packages |
| TS2Vec + k-means | Official **Python 3.8, torch 1.8.1, SciPy 1.6.1, NumPy 1.19.2, pandas 1.0.1, sklearn .24.2** | Legacy numeric pins differ; isolate | UCR and UEA loaders; pooled full-series embedding must be declared; scratch SSL | MIT | **MEDIUM**; use official encoder and one explicit clustering head, not its supervised evaluation pipeline |
| TimesURL + k-means | Official repo verified; framework PyTorch, exact pins not inspected | UNKNOWN exact compatibility | Author artifact; loader/weight requirements not established here | MIT | **MEDIUM / HIGH**, depending on recovery; optional replacement/addition rather than automatic duplicate of TS2Vec |
| TFEC | Preprint advertises code; target repository not resolved | UNKNOWN | UNKNOWN | UNKNOWN | **HIGH / VERY HIGH** until artifact verified; optional emerging multi comparator |

“No license found” means no license in inspected evidence, not a legal finding about reuse. Unknown pretrained requirements stay unknown; availability of source does not establish reproducibility.

### Classical and architectural controls

| Candidate | Artifact / versions | Loader and weights | License / compatibility | Effort |
|---|---|---|---|---|
| Euclidean k-means | Existing scikit-learn 1.4.1.post1 declaration | Common canonical matrix; no weights | BSD-3-Clause; close to declared stack | **LOW**; explicitly set n_init, max_iter and seeds |
| k-Shape | Author Python repository; alternatively tslearn 0.6.3 implementation, disclosed as library implementation | Common z-normalized [N,L,1] data; no weights | tslearn BSD-2-Clause; author Python repository redirects to thedatumorg/kshape-python, MIT verified | **LOW** engineering; avoid confusing implementations |
| k-DBA | tslearn TimeSeriesKMeans(metric=dtw), with canonical DBA citation | Common tensor; no weights | Same library; existing declaration | **LOW** integration, potentially HIGH compute; optional diagnostic because KASBA supplies a strong elastic baseline |
| Plain dense AE + post-hoc k-means | Existing PyTorch architecture controls, exact new configuration must be recorded | Correct canonical loader; scratch training | Repository/control provenance, not a new official method | **LOW / MEDIUM**; essential mechanism ablation |
| TiDE-style AE control | Official TiDE **TensorFlow 2.10.1**, NumPy 1.21.6, pandas 1.3.5; author block inspected | Forecast loaders are not a clustering benchmark; no pretrained requirement for scratch model | Apache-2.0; use LoSTer's already adapted block for matched control, not an unnecessary full TF port | **LOW / MEDIUM** matched control; HIGH whole forecasting pipeline adaptation |
| Preformer adapter | Official Python **3.6**, torch **1.9.0** | Forecast loader; inherited adapter, no external weights needed for scratch test | License UNKNOWN; separate old-stack concern | **MEDIUM**, with fairness/provenance repair |
| Pathformer adapter | Official PyTorch repository; exact pins not inspected | Forecast loader; inherited adapter truncates inputs; scratch adaptation | License not found by metadata; compatibility untested | **HIGH** for fair full-length adaptation |
| iTransformer adapter | Official PyTorch repo; exact pins not extracted | Forecast loader; inherited temporal adapter; scratch model | License UNKNOWN here | **MEDIUM**; one-token univariate interpretation limits scientific value |
| PatchTST adapter | Official PyTorch repo; exact Python/torch pins not inspected | Forecast/representation loader; clustering head requires declared adaptation; pretrained weights optional, not required for scratch comparison | Apache-2.0; compatibility untested | **MEDIUM**; preferred optional efficient backbone |
| FEDformer adapter | Official PyTorch repo; exact pins not extracted | Forecast loaders; new clustering adapter required; scratch use does not inherently need weights | License UNKNOWN here | **HIGH** relative to benefit for focused paper |
| LiteTransformer adapter | Official NLP/mobile project, exact versions not inspected | NLP loader, no native UCR pipeline; pretrained/task-conversion requirements UNKNOWN | Custom/other license classification; exact terms not inspected | **VERY HIGH** for a faithful temporal clustering adaptation with little justified value |

Useful pinned inspection anchors: CDCC **013865ff4d13f08d0f5dcdf5bcfba0f5e009c8a0**, FCACC **78b5e5a138ed8c83fea64e5b6668a5cded79d2dc**, TFMCC **882764f52b9efa1ec0950fb90369ed113e988116**, TSGCC **177162926fdc99173aeee2250a35b6aa493004c3**, DTCC-2023 **8531376c0d4efcbcf7a16fda881b219b3d306397**, CC **6dd9370e960a0f592a342af8e259593e6441483c**. Other repository URLs identify inspected current artifacts without asserting a pinned reproduction. Future runs require full immutable SHAs, not moving default-branch links.

## LoSTer Novelty Matrix

The matrix measures **component overlap**, not whether an entire method is identical. PRE-EXISTING = the same component family is demonstrated in the comparator. RELATED = analogous mechanism/purpose, materially different formulation. DISTINCT = verified different mechanism at this axis; it does not establish LoSTer's priority. NOT APPLICABLE = axis is outside the comparator's scope. UNKNOWN = evidence insufficient. For later papers, PRE-EXISTING denotes established component availability in the 2026 comparison landscape, not proof of priority before LoSTer's initial artifact.

Column keys: **G** Concrete/Gumbel k-means; **H** hard-forward differentiable assignment; **D** dense residual encoder; **V** two-view architecture; **I** instance contrast; **C** cluster contrast; **E** entropy balancing; **L** long-sequence focus; **F** training efficiency; **J** joint clustering; **R** retail evaluation. Values use the requested full labels.

### Assignment, architecture and views

| Comparator | G | H | D | V |
|---|---|---|---|---|
| CKM | PRE-EXISTING | PRE-EXISTING | RELATED | DISTINCT |
| CC | DISTINCT | DISTINCT | DISTINCT | PRE-EXISTING |
| DTC | DISTINCT | DISTINCT | DISTINCT | DISTINCT |
| DTCR | DISTINCT | DISTINCT | DISTINCT | DISTINCT |
| DTCC-2023 | DISTINCT | DISTINCT | DISTINCT | PRE-EXISTING |
| TiDE | NOT APPLICABLE | NOT APPLICABLE | PRE-EXISTING | DISTINCT |
| CDCC | DISTINCT | DISTINCT | DISTINCT | PRE-EXISTING |
| DEETO | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| TSGCC | UNKNOWN | UNKNOWN | UNKNOWN | RELATED |
| FCACC | DISTINCT | DISTINCT | DISTINCT | RELATED |
| TFMCC | DISTINCT | DISTINCT | DISTINCT | PRE-EXISTING |
| Diffusion DTCC | UNKNOWN | UNKNOWN | UNKNOWN | PRE-EXISTING |
| UCL-TSC | UNKNOWN | UNKNOWN | DISTINCT | RELATED |
| DETC | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| EMTC | DISTINCT | UNKNOWN | RELATED | RELATED |
| APCL | RELATED | RELATED | DISTINCT | UNKNOWN |
| Spectral-mask method | UNKNOWN | UNKNOWN | UNKNOWN | RELATED |
| RandomNet | DISTINCT | NOT APPLICABLE | DISTINCT | RELATED |
| R-Clustering | DISTINCT | NOT APPLICABLE | DISTINCT | DISTINCT |
| KASBA | DISTINCT | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| FASA / MUFASA | DISTINCT | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| TS2Vec | NOT APPLICABLE | NOT APPLICABLE | DISTINCT | PRE-EXISTING |
| TimesURL | NOT APPLICABLE | NOT APPLICABLE | UNKNOWN | PRE-EXISTING |
| TFEC | UNKNOWN | UNKNOWN | UNKNOWN | RELATED |

FCACC's three views are RELATED, not a novel absence of two-view augmentation. TiDE's reconstruction task is absent; its block design still establishes D. KASBA/R-Clustering hard final labels are not the same H axis, which concerns neural hard-forward gradients.

### Contrastive terms

| Comparator | I | C | E |
|---|---|---|---|
| CKM | DISTINCT | DISTINCT | DISTINCT |
| CC | PRE-EXISTING | PRE-EXISTING | PRE-EXISTING |
| DTC | DISTINCT | DISTINCT | DISTINCT |
| DTCR | DISTINCT | DISTINCT | DISTINCT |
| DTCC-2023 | PRE-EXISTING | PRE-EXISTING | PRE-EXISTING |
| TiDE | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| CDCC | PRE-EXISTING | PRE-EXISTING | PRE-EXISTING |
| DEETO | UNKNOWN | UNKNOWN | UNKNOWN |
| TSGCC | PRE-EXISTING | RELATED | UNKNOWN |
| FCACC | PRE-EXISTING | RELATED | UNKNOWN |
| TFMCC | PRE-EXISTING | PRE-EXISTING | PRE-EXISTING |
| Diffusion DTCC | RELATED | RELATED | UNKNOWN |
| UCL-TSC | RELATED | RELATED | UNKNOWN |
| DETC | UNKNOWN | UNKNOWN | UNKNOWN |
| EMTC | RELATED | RELATED | UNKNOWN |
| APCL | RELATED | RELATED | RELATED |
| Spectral-mask method | RELATED | UNKNOWN | UNKNOWN |
| RandomNet | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| R-Clustering | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| KASBA | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| FASA / MUFASA | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE |
| TS2Vec | PRE-EXISTING | DISTINCT | DISTINCT |
| TimesURL | PRE-EXISTING | DISTINCT | UNKNOWN |
| TFEC | RELATED | RELATED | UNKNOWN |

### Positioning, efficiency, joint learning and applications

| Comparator | L | F | J | R |
|---|---|---|---|---|
| CKM | NOT APPLICABLE | UNKNOWN | PRE-EXISTING | NOT APPLICABLE |
| CC | NOT APPLICABLE | RELATED | PRE-EXISTING | NOT APPLICABLE |
| DTC | RELATED | UNKNOWN | PRE-EXISTING | UNKNOWN |
| DTCR | RELATED | RELATED | PRE-EXISTING | UNKNOWN |
| DTCC-2023 | RELATED | UNKNOWN | PRE-EXISTING | UNKNOWN |
| TiDE | RELATED | PRE-EXISTING | NOT APPLICABLE | UNKNOWN |
| CDCC | RELATED | UNKNOWN | PRE-EXISTING | UNKNOWN |
| DEETO | UNKNOWN | UNKNOWN | RELATED | UNKNOWN |
| TSGCC | RELATED | UNKNOWN | RELATED | UNKNOWN |
| FCACC | RELATED | UNKNOWN | RELATED | UNKNOWN |
| TFMCC | RELATED | UNKNOWN | PRE-EXISTING | UNKNOWN |
| Diffusion DTCC | UNKNOWN | UNKNOWN | RELATED | UNKNOWN |
| UCL-TSC | RELATED | UNKNOWN | RELATED | UNKNOWN |
| DETC | UNKNOWN | UNKNOWN | RELATED | UNKNOWN |
| EMTC | RELATED | RELATED | RELATED | UNKNOWN |
| APCL | UNKNOWN | UNKNOWN | RELATED | UNKNOWN |
| Spectral-mask method | UNKNOWN | UNKNOWN | RELATED | UNKNOWN |
| RandomNet | PRE-EXISTING | PRE-EXISTING | NOT APPLICABLE | UNKNOWN |
| R-Clustering | RELATED | PRE-EXISTING | NOT APPLICABLE | UNKNOWN |
| KASBA | RELATED | PRE-EXISTING | NOT APPLICABLE | UNKNOWN |
| FASA / MUFASA | PRE-EXISTING | PRE-EXISTING | NOT APPLICABLE | UNKNOWN |
| TS2Vec | RELATED | RELATED | DISTINCT | UNKNOWN |
| TimesURL | RELATED | UNKNOWN | DISTINCT | UNKNOWN |
| TFEC | UNKNOWN | UNKNOWN | RELATED | UNKNOWN |

Retail cells are largely UNKNOWN because absence of an application in an abstract is not evidence of absence everywhere. A particular M5/Store Sales case study could be distinct empirically, but application choice is not a new clustering method and its provenance is currently deficient.

**Defensible intersection:** a global fixed-length residual dense AE, CKM-style latent hard-forward center loss, and established two-view instance/cluster regularization, evaluated as a practical known-k univariate clustering system. The intersection is an integration and potentially a useful empirical finding. No verified cell supports claiming a new Concrete estimator, a new residual architectural principle, or new dual contrastive learning. The unexplored question is whether the combination improves quality per unit training cost relative to simpler matched systems and modern competitors.

## Claim Audit

Classifications apply to the historical artifact's **active claims** and their use in a new submission. KEEP can retain a factual observation only with proper provenance. REWRITE changes the proposition, rather than substituting weaker adjectives. No manuscript edits were made.

| Location in M | Major claim, paraphrased | Verdict | Reason / required revised scope |
|---|---|---|---|
| Abstract 148 | Learning useful temporal clusters is challenging | **KEEP** | General motivation; define task and avoid conflating class labels with intrinsic clusters |
| Abstract 148; Intro 183 | Longer inputs create scalability challenges | **QUALIFY** | Length, collection size, channels, width and batch size each contribute; define tested regime |
| Abstract 148; Intro 187 | Deep temporal methods must use inferior surrogates | **REWRITE** | CKM already supplies hard-forward learning; KL/spectral/fuzzy objectives are different inductive biases, not universally inferior |
| Intro 187 | DEC is the first joint optimization on static data | **REMOVE** as a priority assertion | Historical taxonomy can describe DEC without an unaudited universal first claim |
| Intro 187 | KL formulations are known to be suboptimal and favor overlap | **QUALIFY** | No universal theorem establishes inferior clustering; original wording also contains contradictory “not to be sub-optimal” |
| Intro 189 | DTCR/DTCC use spectral relaxation and periodically updated indicators | **QUALIFY** | Verified procedural distinction; state specific implementation and final inference |
| Intro 189; contribution 198 | Frozen updates cause inconsistent assignments, especially long inputs | **QUALIFY** | Hypothesis needs measured assignment lag/stability under equal architecture and stopping; not inevitable failure |
| Intro 191; contribution 198 | First neural temporal model solving assignment non-differentiability | **REMOVE** | CKM primitive precedes it; temporal priority not established; relaxed gradient does not solve exact discrete differentiation |
| Abstract 149; contribution 200 | Novel dense AE architecture | **REWRITE** | TiDE-derived residual dense autoencoder, with architectural adaptation rather than new block invention |
| Abstract 149; Conclusion 833 | Optimizes hard k-means through Gumbel trick | **QUALIFY** | Hard forward distance and relaxed ST backward; total loss includes other terms |
| Contribution 198 | Without suboptimal surrogates | **REMOVE** unqualified assertion | Avoids selected surrogate losses, but uses a surrogate gradient and no optimality guarantee |
| Abstract 148; Intro 193 | RNN autoregression inherently causes observed clustering errors | **QUALIFY** | Free-running decoder can accumulate errors; causal attribution requires matched decoder/teacher-forcing controls |
| Intro 193 | RNNs process observations recursively and can be slow | **QUALIFY** | Structural dependence exists, but runtime depends on implementation/hardware and model family |
| Abstract 148; Intro 193; Conclusion 831 | Time points lack meaning, so Transformers cannot capture dependencies | **REMOVE** | NLP analogy is not a proof about numerical temporal representation |
| Same | Attention discards chronological order | **REWRITE** | No-position attention is permutation equivariant; positional/patch structures change full-model behavior |
| Same | All relevant Transformers have quadratic point-wise cost | **REWRITE** | PatchTST, Pathformer, iTransformer and FEDformer do not fit that universal statement |
| Intro 193; Conclusion 831 | Smaller feasible Transformer batches necessarily reduce clustering quality | **QUALIFY** | Test equal-batch and max-batch regimes; effects are objective/model dependent |
| Intro 195 | Seventeen long-sequence benchmarks | **REWRITE** | Historical lengths 60–1639; explicitly mixed short/long fixed-length tasks |
| Abstract 149; Intro 195; Conclusion 833 | Extensive experiments prove superiority over current SOTA | **REMOVE** as a current conclusion | Contemporary direct methods absent; fresh matched uncertainty/statistics needed |
| Intro 195; Conclusion 840 | Two retail datasets prove real-world effectiveness/robustness | **QUALIFY** | M5 membership correction may require reruns; Store Sales preprocessing unavailable; application labels are proxies |
| Contribution 199 | Empirical evidence of generally poor SOTA Transformer capacity | **REWRITE** | Selected forecasting backbones adapted to a specific loss, not published clustering SOTA |
| Contribution 201; timings 824–826 | Significant speedup over RNN/Transformer models | **QUALIFY** | Historic per-pass/epoch numbers lack complete logs and a common total-cost procedure; CKM/IDEC already faster per epoch |
| Methods / ablation 741–743 | Improvements are due to dense architecture and hard optimization | **QUALIFY** | Old ablations suggest a hypothesis; fresh matched controls needed for attribution |
| Ablation 743 | All differences are statistically significant | **REWRITE** | Per-dataset five-run raw evidence unavailable; multiplicity and experimental unit must be specified |
| Convergence 746–762 | Rapid stable convergence in first 20 epochs | **QUALIFY** | Two selected plots and 13 recovered traces are not five-seed aggregate stability evidence |
| Convergence 762 | DTCR/DTCC volatility proves accumulated autoregressive error | **REMOVE** causal inference | Multiple mechanisms/confounders; a trace does not identify the cause |
| Convergence 762 | Early label agreement proves optimization convergence | **REWRITE** | Assignment stability is a stopping statistic, not proof of stationary objective or correct clusters |
| Convergence 764 | Preformer's behavior shows Transformers fundamentally ill-posed for LSTC | **REMOVE** | One adapter/dataset trace cannot establish a model-family impossibility |
| Timings 826 | RNN/Transformer real-world deployment infeasible | **REMOVE** | No deployment budget, full cost model or broad architecture evidence |
| Conclusion 829 | First study to expose/resolve hard-assignment drawbacks in LSTC | **REMOVE** | Priority not established; issues and alternative objectives already studied |
| Conclusion 831 | First comprehensive RNN/Transformer critique; limitations unexplored | **REMOVE** | Critique is not a new method; existing forecasting/clustering efficiency work predates this assertion |
| Conclusion 831 | Partitions arbitrarily long sequences | **REWRITE** | Fixed-length instantiated dense input/output layers; tested lengths are finite |
| Conclusion 833 | Avoids recurrence and attention | **KEEP** | Factual active architecture description |
| Conclusion 833 | Preserves temporal order | **QUALIFY** | Dense coordinates are position-specific; no learned locality/shift invariance or temporal modeling superiority follows |
| Conclusion 833 | Allows fast training and large batches | **QUALIFY** | Measure peak memory and total cost including both views/pretraining; dense parameter growth matters |
| Limitations 836 | Current model is univariate | **KEEP** | Exact implemented scope |
| Limitations 836 | Multivariate clustering remains underexplored | **REWRITE** | EMTC and other multi representation/clustering work establish an active benchmark literature |
| Limitations 838 | LoSTer does not advance contrastive clustering itself | **KEEP** | Correctly identifies contrastive learning as supporting, not an original contribution |
| Limitations 838 | Stronger contrastive objectives may improve it | **QUALIFY** | A hypothesis, not a verified gain; no new objective is proposed in this audit |
| Limitations 838 | LoSTer is a strong foundation for future methods | **QUALIFY** | Must first show reproducibility and incremental value over matched CKM/dense controls |
| Limitations 840 | Other domains may demonstrate generalizability | **KEEP AS FUTURE QUESTION** | Must not be presented as demonstrated generalization |
| Methods / metrics | Geometric NMI printed while code uses arithmetic NMI | **REWRITE** | Align definition with newly declared metric; no silent retroactive reinterpretation |
| Methods / architecture | Lower-dimensional bottleneck for every dataset | **QUALIFY** | d=256 exceeds L=60/80/140; “latent embedding” is accurate |
| Source availability 204 | Source code available on GitHub | **QUALIFY** | Source availability does not establish checkpoint/full-result recoverability |

The research highlights at M158–165 are inside a comment environment, not active rendered contributions. They repeat the active contribution claims and inherit their REMOVE/REWRITE/QUALIFY classifications; no claim is credited merely because it exists in commented source.

## 2026 Experimental Options

These are recommendations for a separately authorized experimental phase. They were **not executed**. Publication-quality evidence cannot be recovered by adding confidence bars to historical averages.

### Shared prerequisite: recover a valid scientific execution contract

Audit 2 imposes five non-negotiable conditions:

1. **Fresh seeds and run-level records:** historical five-run variance and exact trained LoSTer states cannot be recovered. Old point estimates cannot supply new SD, bootstrap intervals or paired run tests.
2. **Canonical record IDs:** under standard headerless TSV format the current loader drops the first TRAIN and TEST record. Correcting this later changes the evaluated dataset. All methods in a new table must use the corrected identical records, including initialization and metrics. Do not assert the historical printed table was necessarily produced with the current loader defect without provenance.
3. **Checkpoint completeness:** the active trainable centroid states must be saved alongside both AEs, optimizer/scheduler states and configuration. Dormant wrapper centers cannot substitute for the centers actually trained. Predictions, record ordering and preprocessing manifests must be retained.
4. **Retail provenance:** M5 needs an exact corrected membership comparison because the current filter includes store_id and omits a final day. If membership or inputs change, rerun all comparators for any retained new M5 table. Store Sales preprocessing is unavailable: exclude it from the minimum/strong campaign unless reconstructed and validated. Saved array shapes alone do not establish reproducibility.
5. **Method versus bug distinction:** loader/checkpoint repairs are implementation/provenance repairs; equation/metric alignment is manuscript consistency; changing the extra softmax, loss weights, augmentation or architecture is a methodological variant. Do not silently roll all of these into a “fixed LoSTer” result.

Before any future training, retrieve the missing full papers/protocols for DEETO, TSGCC, diffusion DTCC and APCL; verify code licenses/settings, data IDs and seed policy. If faithful reproduction remains impossible, state the limitation and restrict the claims. Reimplementing inaccessible methods from an abstract is not a credible substitute.

### Minimum credible revision

| Dimension | Predeclared recommendation |
|---|---|
| Scientific question | Does the specific dense/hard/contrastive integration offer useful known-k univariate clustering quality at a competitive total cost? |
| Datasets | **40 UCR**, exactly the CDCC list in Benchmark Protocols, subject to a transparent finite-value check. No outcome-based replacements. |
| Primary task | Pooled TRAIN+TEST transductive clustering; common per-series normalization, same complete records and oracle k. No target-label model selection. |
| Baselines | **14**: CDCC, DEETO, TSGCC, FCACC, TFMCC, diffusion DTCC; CKM reproduction/matched control; DTCC-2023; Euclidean k-means, k-Shape, R-Clustering, KASBA, FASA; TS2Vec + k-means. Full protocol/artifact gates apply. |
| Additional novelty obligation | APCL must be discussed; obtain its final protocol. If the headline becomes prototype/hard-assignment novelty, APCL quantitative comparison becomes required before that headline can be supported. |
| Metrics | **ARI primary**, arithmetic NMI secondary; Hungarian ACC and RI for context. Log AMI if possible at negligible evaluation cost, but do not multiply confirmatory claims across metrics. |
| Seeds | **5 new seeds 0–4** for every stochastic method; deterministic methods need one genuine deterministic fit, with that exception declared. k-means restarts are internal algorithm settings, not five independent reported seeds. |
| Reporting | Per-dataset per-method mean ± SD plus individual seed scores; paired effect sizes, failure counts and task sizes. No reuse of old variance or selective best seed. |
| Statistics | Dataset-level paired means: omnibus rank summary/Friedman where appropriate; predeclared Wilcoxon signed-rank comparisons to LoSTer with Holm correction; report median paired differences and wins/ties/losses. Bootstrap datasets for a descriptive CI, recognizing family dependence. |
| Efficiency | Total wall-clock fit and inference, pretraining/initialization included; parameters, peak GPU allocation/reservation and host memory; samples/sec; equal hardware/thread policy and fixed-batch versus max-feasible-batch distinction. |
| Controlled scaling | **10 unique (N,L) conditions**: length {512,1024,2048,4096,8192,16384} at N=512; size {128,512,2048,4096,8192} at L=1024. Their one shared condition is counted once. Fixed k, generator, seed, architecture-width policy and batch for a given track; report task-generation caveats. |
| Ablation datasets | **8 existing core tasks**: Beef, ECG200, SyntheticControl, ProximalPhalanxTW, ACSF1, EOGVerticalSignal, SemgHandMovementCh2, PigAirwayPressure. Four short/moderate and four long, with varied k/source families. |
| Ablations | Matched residual AE + post-hoc k-means; matched residual AE + CKM with no contrast; same AE + spectral objective; same AE + relaxed-forward assignments under identical regularization; plain dense AE counterpart; remove instance contrast; remove cluster contrast; remove entropy. Use these eight focused configurations, not a full factorial. The extra-softmax interpretation is a separate diagnostic. |
| Multivariate | Excluded; state univariate/fixed-length scope explicitly. |
| Forecasting Transformers | **Zero new adapters**; remove universal Transformer claims. |
| Retail | Neither required. Keep provenance concerns documented instead of treating saved arrays as trustworthy new benchmarks. |

Nominal primary workload is **40 × 5 × 15 = 3000 method–dataset–seed fits**, including LoSTer, before deterministic savings and any failures; this is a fit count, not a GPU-hour estimate. The eight ablation configurations add at most **8 × 5 × 8 = 320** fits and share already fitted identical controls where scientifically valid. Use model hashes/configuration equality to justify reuse, not approximate architectural similarity. The 10 scaling conditions are separate from the 40 real datasets; profile a representative competitor panel rather than every method at every extreme.

Five modern named papers alone would not cover the additional eligible diffusion method found here. Conversely, adding every newly found small study is not automatically necessary: prioritize direct mechanistic overlap and distinct strong alternatives. UCL-TSC, DETC and spectral masking are secondary for the minimum, with explicit discussion rather than assumed irrelevance.

### Strong revision — recommended practical destination

| Dimension | Recommendation beyond the minimum |
|---|---|
| Scientific question | Is the observed quality–cost tradeoff stable across domains, sequence lengths and evaluation protocols, and which component explains it? |
| Datasets | **44 UCR**: the same core 40 + CinCECGTorso, StarLightCurves, MixedShapesRegularTrain, MixedShapesSmallTrain. Rerun the same primary methods on all four additions before quoting a 44-task aggregate. |
| Seeds | **10 fresh seeds 0–9**. Initial five-seed runs can be extended with seeds 5–9 when code, data and configuration hashes are unchanged; this is not a second incompatible campaign. |
| Baselines | Minimum 14 plus **RandomNet, APCL, and one PatchTST adapter**: **17 baselines + LoSTer**. APCL final-paper/artifact gate remains. If APCL cannot support the common task, keep a clearly separate author-faithful/adaptive-k study and do not count it as a matched 44-task oracle-k result. |
| Metrics/statistics | Same primary/secondary hierarchy and adjusted paired tests; family-stratified robustness; intervals and effect sizes. Analyze natural long versus short strata without creating post-hoc favorable subgroups. |
| Efficiency | Complete timings for all 44; detailed memory, throughput, peak batch and time–quality curves on all **11 L≥1024** tasks; controlled length/collection scaling. Compare CPU baselines as CPU systems, not forced neural GPU implementations. |
| Held-out track | **8 predeclared tasks**, the same ablation list: TRAIN fit/TEST predict with frozen preprocessing fitted only where needed on TRAIN. Run LoSTer, CKM control, CDCC, FCACC, TFMCC, R-Clustering, KASBA and TS2Vec + k-means. This separate track must not be merged with pooled scores. |
| Ablations | Extend the same eight-task component study to ten seeds. Add extra-softmax versus justified direct-assignment/probability handling and a temperature-schedule sensitivity; these are labeled methodological variants, not repairs. Equal-batch PatchTST control tests the architectural claim. |
| Convergence | Save unlabeled training loss, assignment changes and validation-independent stopping traces. Label metrics may be plotted after the fixed training policy, but cannot choose the winning epoch. Report pretraining and joint-stage time separately. |
| Multivariate/retail | Still not required. One reconstructed M5 case is optional only after membership/data provenance and all comparator reruns are resolved. |

If all 18 methods support the common task, the nominal accuracy workload is **44 × 10 × 18 = 7920** fits. This is expensive enough that two-stage continuation, reuse of unchanged first-five runs, and avoiding duplicate forecast adapters matter. The eight-method held-out track adds **8 × 10 × 8 = 640** fits. A partial APCL track changes these counts and must be reported transparently. Do not fill missing rows by citing unmatched paper means.

**Smallest practical strong-submission variant:** finish the 44-task, five-seed common campaign first, the eight-task mechanistic ablations and detailed length/quality–cost study; extend to ten seeds only after a stable protocol and scientifically interpretable signal. One PatchTST adapter and full adaptive-k APCL study should follow the actual narrative. If no credible positive contribution survives the matched controls, more seeds/datasets will not manufacture novelty.

### Ideal / expensive revision

| Dimension | Recommendation |
|---|---|
| Scientific question | Does the empirical finding generalize across the eligible univariate archive and to a genuinely specified multivariate setting? |
| Datasets | **112 fixed-length, finite UCR tasks** under the KASBA eligibility rule, plus the exact **15 EMTC UEA tasks**: **127 archive tasks**, with separate uni/multi tables. The exact UCR 112-ID manifest must be recovered and checked before execution; this audit did not claim to verify all array missingness. |
| Scope gate | Multivariate work is conditional on a separately approved and validated extension; no implementation is designed here. Full 128 UCR would require additional unequal-length/missingness scope and is a different undertaking. |
| Baselines | Strong common univariate panel, plus **TimesURL + k-means, UCL-TSC, DETC, spectral-mask method and DTCR** when faithful artifacts/protocols are available. These are breadth/robustness additions, not retroactively mandatory for the minimum. |
| Multi panel | LoSTer extension, **EMTC, MUFASA**, TS2Vec + k-means and Euclidean k-means as a five-method starting panel; additional author-faithful multi methods only after their protocols are verified. TFEC remains optional preprint context unless final acceptance/artifact status is resolved. |
| Metrics/seeds | Same ARI/NMI/ACC hierarchy; **10 seeds for all principal stochastic methods**; **20 only on the predeclared eight-task ablation/convergence subset**, not indiscriminate doubling of every fit. |
| Statistical analysis | Dataset-level adjusted paired tests separately by scope/task; hierarchical/family sensitivity and effect CIs. Unknown-k evaluation is a separate task with predicted k and cluster-quality tradeoffs, not the oracle-k leaderboard. |
| Efficiency | Archive-wide resource/failure table plus natural and controlled scaling; parameter/storage growth, total training and inference budgets, reproducible CPU/GPU policies. No hiding very costly or failed classical/deep methods. |
| Ablations | Reuse matched models/representations, examine assignment-gradient and architecture effects on eight tasks, and validate the actual additional multi mechanism. No full factorial across 127 tasks. |
| Retail | At most a verified reconstructed application study if useful to the question; Store Sales remains excluded until its preprocessing is genuinely recovered. |

Ten runs on 112 univariate datasets with 23 methods would imply **25760 fits**, plus **750** for the five-method 15-UEA starting panel. Those counts are conditional on support/artifact availability and exclude the extra selected-seed diagnostics. They are a workload ceiling illustration, not a recommended immediate job queue or an estimate of acceptance value.

### Efficiency and inference contract for all tiers

A valid efficiency table must include both AE pretrains, augmentations, k-means initialization, joint training, any required graph/neighbor/pretext stages, model selection allowed by the declared protocol, and final full-pool embedding/inference. Distinguish startup/compilation from steady-state training and report both where relevant. CUDA timing needs synchronization and a stated warm-up/repetition policy. The historical Store Sales epoch plot and StarLightCurves pass table measure different things and cannot establish one total speedup.

Measure quality at fixed wall-clock budgets and report total time to a fixed, label-independent stopping policy. Epoch counts alone are not equivalent compute budgets. Same batch, representation width and parameter count cannot always all be equal simultaneously; separate controls explain which is matched. Contrastive batch-size effects need a fixed-batch track and a resource-limited maximum-batch track. Timeouts/OOM must be recorded with N,L,k,hardware and attempted settings; no selective exclusions from macro-averages.

For statistical comparisons, seeds are nested within datasets. Treating 40 datasets × five seeds as 200 independent problem instances exaggerates significance. Related axes/families require sensitivity analysis, and synthetic scaling conditions are not additional independent application domains. True class counts are benchmark oracle k, not a universally correct intrinsic-cluster definition. APCL's adaptive-k setting needs its own fair comparison and cannot silently inherit the oracle k of other methods.

## Defensible Revised Narratives

| Narrative | Central scientific question | Actual possible novelty | What must not be claimed | Required evidence | Relative strength |
|---|---|---|---|---|---|
| **1. Efficient global dense temporal clustering** | When does a global residual dense embedding plus joint clustering yield useful quality at lower total cost? | A measured practical quality–cost tradeoff for this integration | New residual blocks; first efficient temporal clustering; universal superiority over attention/RNNs; arbitrary-length support | 44-task fresh comparison, modern direct + random/elastic baselines, complete fit/inference timing, natural/controlled scaling, matched dense controls | **Strongest conditional path**; depends on measurable advantage rather than rhetoric |
| **2. Hard assignments under matched temporal representations** | Does CKM-style hard-forward training improve stability/quality relative to spectral/soft alternatives under the same encoder and regularization? | A controlled empirical answer and a specific useful integration, not a new estimator | First Gumbel k-means; exact/unbiased gradients; necessary superiority of hard clustering; new dual contrast | Matched CKM/spectral/soft controls, component losses, temperature sensitivity, label-independent stopping, APCL context/faithful comparison | **Moderate**, potentially strong if a clear reproducible mechanism survives controls |
| **3. Reproducible component and cost study of deep temporal clustering** | Which ingredients actually justify their complexity once protocols, records and total costs are controlled? | A trustworthy empirical characterization, including negative results and reproducible artifacts | A newly invented method because the original ingredients lack novelty; SOTA from unmatched paper averages | Correct records/metrics/checkpoints, 40–44 matched tasks, explicit unavailable-artifact limitations, ablations and resource frontier | **Honest fallback, weaker method novelty**; venue fit may favor empirical/reproducibility contribution |

A retail-centered fourth narrative is not currently defensible: M5 selection needs verification and Store Sales preprocessing is absent. It could become an application contribution after reconstruction, but cannot be used now to rescue methodological novelty.

## Recommended Path

Proceed with narrative 1 and use narrative 2 as its mechanistic test. The revision should describe an **integration of established mechanisms** for **fixed-length univariate, known-k clustering**, and ask whether it earns its added complexity relative to CKM, simple AE features, random features and strong elastic clustering.

The decision sequence after this analysis is:

1. Recover missing comparator papers/protocols and freeze exact data IDs, normalization, selection policy and software SHAs.
2. In a separately authorized implementation phase, repair record loading and scientific artifact saving; keep method variants separate and preserve all historical artifacts.
3. Run the 40-core five-seed campaign and eight-task matched controls with total-cost logging. Stop escalating the benchmark if the results do not support the intended scientific question; report negative findings.
4. Add four predeclared datasets for 44, then extend unchanged settings to ten seeds for the strong campaign. Add one efficient forecast backbone only when it tests the actual architecture claim.
5. Draft new claims only after those results exist. Do not rewrite the immutable rejected manuscript, reuse unrecoverable old variance, or call a source-only release reproducible.

This audit establishes what claims and comparisons are scientifically defensible. It does not establish that LoSTer already improves on 2026 methods, that missing full protocols can be faithfully reconstructed from abstracts, or that the recommended campaign will secure a journal acceptance.

## Verified Bibliography / URLs / DOIs

All records were checked against primary proceedings/publisher/author sources or publisher-deposited DOI metadata. A code URL is an author-linked artifact unless explicitly labeled third-party. “DOI not found” is not proof no DOI exists. Access labels refer to the evidence inspected in this audit, not a universal access status. No preprint is substituted for a known final publication.

### Core lineage and architecture

1. **Boyan Gao, Yongxin Yang, Henry Gouk, Timothy M. Hospedales.** *Deep Clustering with Concrete K-Means*. **ICASSP 2020**, pp. 4252–4256; published, peer reviewed. DOI **10.1109/ICASSP40776.2020.9053265**. [Conference record](https://cmsworkshops.com/ICASSP2020/Papers/ViewPaper.asp?PaperNum=4196), [author institution record](https://www.research.ed.ac.uk/en/publications/deep-clustering-with-concrete-iki-means/), [accepted final manuscript](https://www.research.ed.ac.uk/files/134695608/Deep_Clustering_with_Concrete_GAO_DOA24012020_AFV.pdf). Official code not identified. **FULL-PAPER method-passage evidence**; binary accepted PDF blocked. Section 3/Algorithm 1, not an unrelated aggregator, establishes the hard/relaxed optimization comparison.

2. **Yunfan Li, Peng Hu, Zitao Liu, Dezhong Peng, Joey Tianyi Zhou, Xi Peng.** *Contrastive Clustering*. **AAAI 2021**, 35(10), 8547–8555; published, peer reviewed. DOI **10.1609/aaai.v35i10.17037**. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/17037), [official code](https://github.com/XLearning-SCU/2021-AAAI-CC). **FULL-PAPER + CODE EVIDENCE**. Peng Hu is part of the verified author list.

3. **Naveen Sai Madiraju, Seid M. Sadat, Dimitry Fisher, Homa Karimabadi.** *Deep Temporal Clustering: Fully Unsupervised Learning of Time-Domain Features*. **2018 arXiv preprint**, arXiv:1802.01059. [Primary record](https://arxiv.org/abs/1802.01059); **10.48550/arXiv.1802.01059** is the preprint identifier, not a conference-publication DOI. Final peer-reviewed publication not verified. [FlorentF9 implementation](https://github.com/FlorentF9/DeepTemporalClustering) is **third-party**, not established as author official code. **ABSTRACT-LEVEL preprint evidence** with historical implementation context.

4. **Qianli Ma, Jiawei Zheng, Sen Li, Gary W. Cottrell.** *Learning Representations for Time Series Clustering*. **NeurIPS 2019**, Advances in Neural Information Processing Systems 32; published, peer reviewed. DOI not found. [Proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/1359aa933b48b754a2f54adb688bfa77-Abstract.html), [final paper](https://papers.neurips.cc/paper/8634-learning-representations-for-time-series-clustering.pdf), [official code](https://github.com/qianlima-lab/DTCR). **FULL-PAPER indexed methods/experiment evidence + README**. Main evaluation is 36 TRAIN/TEST tasks and five runs, not the pooled 17-task LoSTer protocol.

5. **Ying Zhong, Dong Huang, Chang-Dong Wang.** *Deep Temporal Contrastive Clustering*. **Neural Processing Letters 55 (2023)**, 7869–7885; published, peer reviewed, online 6 May 2023. DOI **10.1007/s11063-023-11287-0**. [Publisher](https://link.springer.com/article/10.1007/s11063-023-11287-0), [final PDF](https://link.springer.com/content/pdf/10.1007/s11063-023-11287-0.pdf), [official code](https://github.com/07zy/DTCC). **FULL-PAPER + CODE EVIDENCE**. This is DTCC-2023 throughout; it is not the later diffusion paper.

6. **Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K. Mathur, Rajat Sen, Rose Yu.** *Long-term Forecasting with TiDE: Time-series Dense Encoder*. **Transactions on Machine Learning Research, 2023**; published, peer reviewed. DOI not found. [Final OpenReview record](https://openreview.net/forum?id=pCbC3aQB5W), [Google Research publication catalog](https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/), [official code](https://github.com/google-research/google-research/tree/master/tide). Google catalog uses the variant title “Long Horizon Forecasting…” and an abbreviated author display; final indexed citations/code identify the six-author record. **CODE EVIDENCE** is operative for the architecture; final PDF access was blocked.

### Direct modern competitors and additional eligible methods

7. **Furong Peng, Jiachen Luo, Xuan Lu, Sheng Wang, Feijiang Li.** *Cross-Domain Contrastive Learning for Time Series Clustering*. **AAAI 2024**, 38(8), 8921–8929; published, peer reviewed, 24 March 2024. DOI **10.1609/aaai.v38i8.28740**. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/28740), [final PDF](https://ojs.aaai.org/index.php/AAAI/article/download/28740/29426), [official code](https://github.com/JiacLuo/CDCC). **FULL-PAPER + CODE EVIDENCE**.

8. **Sangho Lee, Chihyeon Choi, Youngdoo Son.** *Deep time-series clustering via latent representation alignment*. **Knowledge-Based Systems 303 (2024)**, 112434; published, peer reviewed. DOI **10.1016/j.knosys.2024.112434**. [Publisher](https://www.sciencedirect.com/science/article/pii/S0950705124010682), [author lab publication list](https://sites.google.com/view/dgudslab/JOURNAL). Official code not identified. **ABSTRACT-LEVEL / indexed-introduction evidence**; exact counts/protocol remain unknown.

9. **Qin Zhang, Zhuoluo Liang, Alladoumbaye Ngueilbaye, Han Liu, Hong Zhou, Joshua Zhexue Huang.** *Graph-augmented contrastive clustering for time series data*. **Knowledge-Based Systems 330, Part B (2025)**, 114602; published, peer reviewed, November 2025 issue. DOI **10.1016/j.knosys.2025.114602**. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0950705125016417), [official code](https://github.com/zololululu/TSGCC). **ABSTRACT-LEVEL / indexed-section + CODE EVIDENCE**; full final text not recovered.

10. **Congyu Wang, Mingjing Du, Xiang Jiang, Yongquan Dong.** *Fuzzy cluster-aware contrastive clustering for time series*. **Pattern Recognition 173 (2026)**, 112899; published, peer reviewed; online 9 December 2025. DOI **10.1016/j.patcog.2025.112899**. [Publisher/DOI](https://doi.org/10.1016/j.patcog.2025.112899), [author-hosted final publisher PDF](https://dumingjing.github.io/files/paper-21_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series/2026_PR_Fuzzy%20cluster-aware%20contrastive%20clustering%20for%20time%20series.pdf), [official code](https://github.com/Du-Team/FCACC). **FULL-PAPER + CODE EVIDENCE**. Final tables have nine baseline columns while the setup prose says eight; use actual table identities, not the inconsistent prose count.

11. **Congyu Wang, Mingjing Du, Xiang Jiang.** *Time-Frequency Augmented Multi-level Contrastive Clustering for Time Series*. **AAAI 2026**, 40(31), 26142–26150; published, peer reviewed, 14 March 2026. DOI **10.1609/aaai.v40i31.39817**. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/39817), [final PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39817/43778), [official code](https://github.com/Du-Team/TFMCC). **FULL-PAPER + CODE EVIDENCE**. Author train.py, not the separate loader signature, verifies split pooling.

12. **Yu Zhou, Meng Liu, Shi Chen, Zhikui Chen.** *Deep time series contrastive clustering with cross-view reliable cluster diffusion*. **Expert Systems with Applications 330 (2026)**, 133028; published, peer reviewed, **advance online 28 May 2026**, issue 1 December 2026. DOI **10.1016/j.eswa.2026.133028**. [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417426019391), [author-institution publication/date record](https://researchportal.hkust.edu.hk/en/publications/deep-time-series-contrastive-clustering-with-cross-view-reliable-/), [official code](https://github.com/qinqinhan/DTCC). **ABSTRACT-LEVEL / indexed-introduction + README EVIDENCE**. Eligible by online date; acronym collision is explicitly resolved.

13. **Bo Cao, Qinghua Xing, Ke Yang, Xuan Wu, Longyue Li.** *Unsupervised Contrastive Learning for Time Series Data Clustering*. **Electronics 14(8) (2025)**, 1660; published, peer reviewed, 19 April 2025. DOI **10.3390/electronics14081660**. [Publisher](https://www.mdpi.com/2079-9292/14/8/1660). Official code not identified. **FULL-PAPER indexed publisher passage/table evidence**; complete browser retrieval intermittently failed. Twelve UCR datasets and loss variability are verified; loss SD is not accuracy SD.

14. **Hossein Abbasimehr, Ali Noshad.** *Deep time-series clustering via evolutionary learning and graph-based manifold learning*. **Information Processing & Management 63(2), Part A (2026)**, 104409; published, peer reviewed. DOI **10.1016/j.ipm.2025.104409**. [Publisher](https://www.sciencedirect.com/science/article/pii/S0306457325003504), [publisher-linked artifact](https://anonymous.4open.science/r/Deep-Evolutionary-Time-Series-Clustering-DETC-DD2D/README.md). **ABSTRACT-LEVEL / indexed-section evidence**; artifact contents not verified. Fifteen public datasets is not a verified UCR/UEA breakdown.

15. **Zexi Tan, Xiaopeng Luo, Yunlin Liu, Yiqun Zhang.** *Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering*. **AAAI 2026**, 40(30), 25787–25795; published, peer reviewed, 14 March 2026. DOI **10.1609/aaai.v40i30.39777**. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/39777), [final PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39777/43738), [official code](https://github.com/yueliangy/EMTC). **FULL-PAPER main-section + README EVIDENCE**. The final paper explicitly sends some tests/ablation results to an extended version; this audit does not assert those supplement results were fully inspected.

16. **Wei Li.** *Adaptive Prototypical Contrastive Learning for Time Series Clustering*. **Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, 2026**; published, peer reviewed. DOI **10.1145/3770855.3817773**; publisher-deposited record dates August 2026. [ACM](https://dl.acm.org/doi/10.1145/3770855.3817773), [author publication page](https://www.weili.space/about/publications/), [official code](https://github.com/William-Liwei/APCL), [assignment source](https://github.com/William-Liwei/APCL/blob/main/models/soft_assignment.py). **METADATA + CODE EVIDENCE** only: do not impute final benchmark counts/results from the synthetic demo.

17. **Zhixuan Wang, Xiufang Xu, Sen Xu, Naixuan Guo, Xuesheng Bian, Shanliang Yao, Tian Zhou, Yuyang Shen.** *Spectral-adaptive masking and hierarchical contrastive learning for time series clustering*. **Intelligent Data Analysis: An International Journal, 2026 OnlineFirst**; published, peer reviewed, 2 February 2026. DOI **10.1177/1088467X251413367**. [Publisher](https://journals.sagepub.com/doi/10.1177/1088467X251413367). Official code not identified. **ABSTRACT-LEVEL EVIDENCE**; no invented volume/pages or method acronym.

18. **Haojun Li, John Paparrizos.** *MUFASA: Fast and Accurate Multivariate Time-Series Clustering*. **Proceedings of the ACM on Management of Data 4(3), SIGMOD, Article 213 (June 2026)**, 1–29; published, peer reviewed. DOI **10.1145/3802090**. [Final author-hosted publisher PDF](https://paparrizos.org/papers/LiSIGMOD26.pdf), [official FASA/MUFASA code](https://github.com/TheDatumOrg/MUFASA). **FULL-PAPER method/setup passage + CODE EVIDENCE**. FASA is the univariate algorithm in this publication; MUFASA is its multivariate companion.

### Strong non-deep/representation baselines and comparative context

19. **Jorge Marco-Blanco, Rubén Cuevas.** *Time series clustering with random convolutional kernels*. **Data Mining and Knowledge Discovery 38 (2024)**, 1862–1888; published, peer reviewed, online 1 April 2024. DOI **10.1007/s10618-024-01018-x**. [Publisher](https://link.springer.com/article/10.1007/s10618-024-01018-x), [official R-Clustering code](https://github.com/jorgemarcoes/R-Clustering). **FULL-PAPER + README EVIDENCE**.

20. **Xiaosheng Li, Wenjie Xi, Jessica Lin.** *Randomnet: clustering time series using untrained deep neural networks*. **Data Mining and Knowledge Discovery 38 (2024)**, 3473–3502; published, peer reviewed, online 22 June 2024. DOI **10.1007/s10618-024-01048-5**. [Publisher](https://link.springer.com/article/10.1007/s10618-024-01048-5), [official code](https://github.com/Jackxiini/RandomNet). **FULL-PAPER + README EVIDENCE**.

21. **Christopher Holder, Anthony Bagnall.** *Rock the KASBA: blazingly fast and accurate time series clustering*. **Data Mining and Knowledge Discovery 40 (2026)**, Article 21; published, peer reviewed, 25 February 2026. DOI **10.1007/s10618-026-01189-9**. [Publisher](https://link.springer.com/article/10.1007/s10618-026-01189-9), [aeon](https://github.com/aeon-toolkit/aeon), [author-linked reproduction notebook](https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/publications/clustering/kasba/kasba.ipynb). **FULL-PAPER + dependency metadata EVIDENCE**. Later final publication replaces its older preprint as operative evidence.

22. **John Paparrizos, Sai Prasanna Teja Reddy Bogireddy.** *Time-Series Clustering: A Comprehensive Study of Data Mining, Machine Learning, and Deep Learning Methods*. **Proceedings of the VLDB Endowment 18(11) (2025)**, 4380–4395; published, peer reviewed. DOI **10.14778/3749646.3749700**. [Final author-hosted publisher PDF](https://paparrizos.org/papers/PaparrizosVLDB25.pdf), [official benchmark code](https://github.com/TheDatumOrg/TSB-Clustering). **Final abstract/setup-context evidence**, not a complete 84-method artifact audit. Full author name comes from the final paper; the shorter author-site name is not substituted.

23. **Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, Bixiong Xu.** *TS2Vec: Towards Universal Representation of Time Series*. **AAAI 2022**, 36(8), 8980–8987; published, peer reviewed. DOI **10.1609/aaai.v36i8.20881**. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/20881), [official code](https://github.com/zhihanyue/ts2vec). **Publisher metadata + CODE/README EVIDENCE**. Adding k-means is an explicitly identified clustering adaptation.

24. **Jiexi Liu, Songcan Chen.** *TimesURL: Self-Supervised Contrastive Learning for Universal Time Series Representation Learning*. **AAAI 2024**, 38(12), 13918–13926; published, peer reviewed. DOI **10.1609/aaai.v38i12.29299**. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/29299), [official code](https://github.com/Alrash/TimesURL). **Publisher metadata / representation-context evidence**; full implementation not audited. The chosen clustering head must be stated rather than inherited from a different paper's FCM or k-means adaptation.

25. **John Paparrizos, Luis Gravano.** *k-Shape: Efficient and Accurate Clustering of Time Series*. **ACM SIGMOD 2015**, 1855–1870; published, peer reviewed. DOI **10.1145/2723372.2737793**. [Official conference contents](https://www.sigmod2015.org/toc_sigmod.shtml), [author final PDF](https://www.paparrizos.org/papers/PaparrizosSIGMOD15.pdf), [author Python repository, redirected](https://github.com/thedatumorg/kshape-python). Author code MIT; tslearn alternative must be identified separately. This older canonical baseline remains scientifically relevant in contemporary comparisons.

26. **François Petitjean, Alain Ketterlin, Pierre Gançarski.** *A global averaging method for dynamic time warping, with applications to clustering*. **Pattern Recognition 44(3) (2011)**, 678–693; published, peer reviewed. DOI **10.1016/j.patcog.2010.09.013**. [Publisher](https://www.sciencedirect.com/science/article/pii/S003132031000453X), [author DBA code](https://github.com/fpetitjean/DBA), GPL-3.0. The proposed optional k-DBA experiment uses a disclosed tslearn implementation; no author code was run.

### Forecasting / task-distant architecture records

27. **Dazhao Du, Bing Su, Zhewei Wei.** *Preformer: Predictive Transformer with Multi-Scale Segment-Wise Correlations for Long-Term Time Series Forecasting*. **ICASSP 2023**, 1–5; published, peer reviewed. DOI **10.1109/ICASSP49357.2023.10096881**. [Publisher DOI](https://doi.org/10.1109/ICASSP49357.2023.10096881), [official code](https://github.com/ddz16/Preformer). Title/authors/venue verified against publisher-deposited Crossref metadata and author README. No final preprint substitution was used.

28. **Peng Chen, Yingying Zhang, Yunyao Cheng, Yang Shu, Yihang Wang, Qingsong Wen, Bin Yang, Chenjuan Guo.** *Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting*. **ICLR 2024**, poster; published, peer reviewed. DOI not found. [Proceedings/OpenReview](https://openreview.net/forum?id=lJkOCMP2aW), [official code](https://github.com/decisionintelligence/pathformer). Final indexed record/author-code evidence; full model reproduction not audited.

29. **Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long.** *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. **ICLR 2024**; published, peer reviewed. DOI not found. [Proceedings/OpenReview](https://openreview.net/forum?id=JePfAI8fah), [official code](https://github.com/thuml/iTransformer). Final indexed record + author README evidence.

30. **Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam.** *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. **ICLR 2023**; published, peer reviewed. DOI not found. [Final proceedings PDF](https://openreview.net/pdf/2e4e6db8733d24f382a7e57c9b3d53d7e0061ade.pdf), [official PatchTST code](https://github.com/yuqinie98/PatchTST). Final published-title/author passage verified; the anonymous under-review PDF was not treated as the final record.

31. **Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin.** *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. **ICML 2022**, PMLR 162, 27268–27286; published, peer reviewed. DOI not found. [Official proceedings](https://proceedings.mlr.press/v162/zhou22g.html), [official code](https://github.com/MAZiqing/FEDformer). Proceedings + README evidence.

32. **Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, Song Han.** *Lite Transformer with Long-Short Range Attention*. **ICLR 2020**; published, peer reviewed. DOI not found. [Author project/publication](https://hanlab.mit.edu/projects/lite-transformer), [official code](https://github.com/mit-han-lab/lite-transformer). Primary author-project evidence; NLP/mobile scope, not a verified native temporal forecaster.

### Emerging preprint and archive metadata

33. **Zexi Tan, Tao Xie, Haoyi Xiao, Baoyao Yang, Yuzhu Ji, An Zeng, Xiang Zhang, Yiqun Zhang.** *TFEC: Multivariate Time-Series Clustering via Temporal-Frequency Enhanced Contrastive Learning*. **arXiv preprint, 2026**, submitted 12 January 2026, arXiv:2601.07550. [Primary record](https://arxiv.org/abs/2601.07550), identifier **10.48550/arXiv.2601.07550**. Final publication/ICASSP acceptance not verified; comments indicate submission, not acceptance. Code is advertised by the preprint but the target repository was not resolved here. **ABSTRACT-LEVEL EVIDENCE**; optional emerging context.

34. **UCR Time Series Classification Archive, 2018 release.** [Official archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), [official DataSummary.csv](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv). Administrative dataset metadata, not an additional publication record. Canonical names, TRAIN/TEST sizes, class counts and numeric/Vary lengths were checked. Missing-value eligibility and actual array hashes require a later authorized data-validation phase.

### Completion and preservation

Only this report was created during Stage B. No scientific code, immutable manuscript/Drafts, dataset, environment dependency, old result, checkpoint or experiment was altered. No training or benchmark was run. Audit 3 was not committed or pushed. Completion verification compares all **299 pre-existing non-Git files** against Stage A's path/content/size/mtime manifest, excluding this report, and checks that HEAD remains e09b9db462b165ac16af100ab5d06ded64bb151c. The required final Git status is a single untracked report under analysis/literature.


Verification passed: all 299 pre-existing files matched the preservation manifest (content, size, path and modification timestamp); all 19 required sections and Markdown table column counts passed; all local document links resolved. HEAD and origin/revision-2026 remain at the Audit 2 commit. Final status: only this Audit 3 report is untracked.
