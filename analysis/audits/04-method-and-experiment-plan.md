# LoSTer 2026 Method and Experiment Plan

Audit date: 2026-10-05. Scope: methodological decisions and experiment design only.

Evidence: [Audit 1](01-paper-code-audit.md), [Audit 2](02-provenance-recovery-audit.md), [Audit 3](../literature/03-literature-novelty-benchmark-audit.md), the supplied implementations, immutable manuscript tables, and the primary sources cited below. Three focused read-only reviews used GPT-6.1 Sol with Extra High reasoning. Recommendations and numerical screening tolerances below are prospective design judgments, not experimental findings.

Stage A completed: Audit 3 was committed as **5db542335b40112bf50eb1bf28642663ce74f0b7**, message **Add 2026 literature novelty and benchmark audit**, and pushed only to **origin revision-2026**. Clean status was verified before Stage B. This report remains uncommitted. No scientific code, manuscript, legacy results, package environment or dataset was changed; no model was executed.

## Executive Decision

1. **Preserve the legacy mathematical method as the control; alter the final method only if a predeclared pilot earns the change.** The extra softmax lacks a strong theoretical justification, but removing it is not an automatic repair. Independent centroid initialization presents a correspondence risk, not a demonstrated bug.
2. Pilot exactly **LoSTer-Legacy-Clean**, **LoSTer-NoResoftmax**, and **LoSTer-AlignedInit**. These isolate the extra-softmax and initialization questions. Do not combine them into a fourth candidate or add shared encoders, new weights, balancing or fashionable augmentation.
3. Use **SyntheticControl, Beef, ECG200, OSULeaf, ShapesAll, SemgHandMovementCh2, CinCECGTorso, StarLightCurves**, each with five paired seeds **{0,1,2,3,4}**. ARI is primary; arithmetic NMI is secondary. Correct canonical TSV ingestion and control initialization, both augmentation snapshots and RNG state across arms.
4. Adopt a change only after the eligibility, quality/stability, consistency, collapse and simplicity gates in the selection rule. A tiny macro gain is insufficient. Broad ARI improvement or a substantial stability/collapse improvement with limited quality loss can qualify; comparable legacy performance favors retaining Legacy-Clean.
5. Freeze **EXTENDED-44**, consisting of the exact CDCC **CORE-40** plus four historical long-series tasks. Do not choose datasets using LoSTer scores. Eight tasks are development tasks; the other **36** support the primary comparisons after method freezing. The full 44-task aggregate is development-inclusive and descriptive.
6. Minimum Tier 1: **Euclidean k-means, k-Shape, CKM, DTCC-2023, CDCC, FCACC, TFMCC, TS2Vec + k-means, R-Clustering, KASBA, FASA**. Eleven comparators cover distinct scientific challenges. Artifact/source gates must be resolved before their numbers enter the main table; an incomplete implementation does not become a faithful reproduction by naming it.
7. Test the claim that **a reproducibly specified integration of established dense reconstruction, hard assignment and contrastive mechanisms offers useful clustering quality at competitive complete-pipeline cost on fixed-length, univariate, known-k tasks**. Do not claim invention of the components, unrestricted superiority, or general Transformer inadequacy. Exclude multivariate work; retain forecasting adapters as historical/context evidence. **M5: REBUILD, conditionally as an appendix application. Store Sales: DROP from new empirical claims.**

The exact frozen method is necessarily conditional before evidence exists: Legacy-Clean is the default; only the two specified alternatives can replace it under the rule below. This is one bounded decision strategy, not a promise that a revised formulation will win.

## Legacy Method Specification

**LoSTer-Legacy means the executed historical algorithm**, including discrepancies with the printed equations. Legacy-Clean uses the same mathematical objective on correctly ingested canonical records; it cannot reproduce historical means exactly when record populations change.

Let N be pooled record count, L sequence length, k the known class count, B the training minibatch size, d=256, and v in {o,a} index original and augmented views.

**Preprocessing and views.** Historical UCR code concatenates TRAIN and TEST, then independently standardizes each series by its mean and population standard deviation:
\[
x_i=(r_i-\operatorname{mean}(r_i))/\operatorname{std}_{ddof=0}(r_i).
\]
Its default pandas header inference discards the first record of each standard headerless split. Audit 2 confirmed this ingestion defect; historical score impact is unknown. No finite/constant-series guard exists. M5 instead uses historical selection and weekly STL trend-plus-seasonal reconstruction, followed by per-series min-max scaling; its filtering issue is discussed separately.

The augmentation is sign inversion, equal-segment permutation, then cubic time warp, in that order. Signs are equiprobable ±1 per series/channel; the segment count is uniform on {1,2,3,4}; warp multipliers are Normal(mean 1, standard deviation 0.2) at six knots. The spline interpolates perturbed coordinates, rescales the endpoint, clips to [0,L−1], and uses NumPy interpolation. No monotonicity constraint exists.

Two loader constructions independently precompute fixed augmentation snapshots. Denote them A_train and A_init. Original-view training and pretraining use X; augmented training and pretraining use A_train. Augmented KMeans initialization uses A_init. There is no epoch-wise augmentation resampling. See [loader](../../models/LoSTer/data/load_data.py) and [augmentation](../../models/LoSTer/data/augmentation.py).

**Networks and initialization.** Two dense residual autoencoders have independent weights and independent pretraining. An ordinary block computes
\[
h'=\operatorname{LN}\{\operatorname{Dropout}[W_2\operatorname{ReLU}(W_1h+b_1)+b_2]+W_sh+b_s\}.
\]
The encoder has three blocks, the first mapping L to d and the rest d to d. The decoder has two ordinary blocks and a final prediction residual block mapping d to L without dropout or LayerNorm. Ordinary-block dropout is 0.1; RevIN and the global AE residual are disabled. The architecture is fixed-length; d exceeds L on some short tasks.

Each view receives 50 reconstruction-only epochs using Adam with learning rate 0.001. Separate full-pool KMeans fits initialize C^o and C^a from ordered original and A_init embeddings, respectively. Initialization is k-means++ with the run seed. Under the pinned sklearn 1.4 behavior, the omitted n_init means one k-means++ initialization; the clean contract will specify n_init=1 explicitly. Active trainable centers belong to the two KMeansLoss modules. The wrapper also registers dormant initial-center copies. See [main](../../models/LoSTer/main.py) and [networks/pretraining](../../models/LoSTer/net/model.py).

**Assignments and estimator.**
\[
D^v_{ij}=\|z_i^v-c_j^v\|_2^2,\qquad
\ell_i^v=\log\operatorname{softmax}(-D_i^v/\sigma^2),\qquad \sigma=1.
\]
With independent standard Gumbel noise G,
\[
Y_i^v=\operatorname{softmax}((\ell_i^v+G_i^v)/\tau_e),\quad
H_i^v=\operatorname{onehot}(\arg\max Y_i^v),\quad
Q_i^v=\operatorname{stopgrad}(H_i^v-Y_i^v)+Y_i^v.
\]
Q is hard in the forward pass from the first batch; its backward path follows Y. This is a biased straight-through surrogate, not exact differentiation of a discrete objective. For fixed logits and positive tau, argmax sampling has categorical probabilities exp(ell), independent of tau; annealing changes the backward relaxation. The schedule is
\[
\tau_e=\max(10\cdot0.65^e,0.01)
\]
for zero-based epoch e. Sigma is a separate RBF bandwidth. See [assignment utilities](../../models/LoSTer/utils/gumbel.py) and [joint training](../../models/LoSTer/experiment.py).

**Losses and reductions.** Define
\[
R^v=\frac{1}{BL}\sum_i\|\hat x_i^v-x_i^v\|_2^2,\qquad
K^v=\frac{1}{Bd}\sum_i\|z_i^v-Q_i^vC^v\|_2^2.
\]
For paired vectors u_i,w_i, define the directional contrastive mean
\[
T(u,w)= -\frac1m\sum_{i=1}^m\log
\frac{\exp(\cos(u_i,w_i)/t)}
{\sum_{j=1}^m\exp(\cos(u_i,w_j)/t)+
 \sum_{j\ne i}\exp(\cos(u_i,u_j)/t)}.
\]
The positive cross-view pair is included in the denominator; the same-view self-pair is masked. Actual code approximates this exclusion by subtracting 10^9 from self logits.

Instance contrast uses row-normalized embeddings, m=B, t=1:
\[
I=T(z^o,z^a)+T(z^a,z^o).
\]
Cluster contrast first performs the extra row softmax
\[
S^v=\operatorname{softmax}(Q^v).
\]
It treats the k columns of S as vectors in sample space, m=k, t=1:
\[
C=T((S^o)^\top,(S^a)^\top)+T((S^a)^\top,(S^o)^\top).
\]
Column cosine uses the library's norm floor. The entropy term is
\[
E=\sum_j\bar p_j^o\log\bar p_j^o+\sum_j\bar p_j^a\log\bar p_j^a,\qquad
\bar p_j^v=B^{-1}\sum_iS_{ij}^v.
\]
It has the correct negative-entropy sign. Both contrastive directions are added, not half-averaged. Entropy is summed across clusters and views. The actual complete objective is
\[
\mathcal L=R^o+R^a+\frac{\alpha}{2}(K^o+K^a)+I+C+E,\qquad\alpha=1.
\]
Thus “unit coefficients” retains the explicit half factor inside the combined k-means term and every original reduction; it does not mean changing all terms into vector-sum means. See [losses](../../models/LoSTer/utils/losses.py).

**Optimization, stopping and inference.** Joint training uses SGD at 0.01, without supplied momentum/weight decay, and StepLR multiplying learning rate by 0.1 every five epochs. Historical training has shuffled B=128 batches with drop_last=True. Evaluation uses the complete ordered pool, dropout disabled. After every joint epoch, predictions are original-view deterministic nearest-center assignments:
\[
\hat y_i=\arg\max_j\ell^o_{ij}=\arg\min_j D^o_{ij}.
\]
Stop at the first comparison, starting after the second epoch, satisfying
\[
N^{-1}\sum_i{\bf1}[\hat y_i^{(e)}\ne\hat y_i^{(e-1)}]<0.001,
\]
or at 100 joint epochs. No warm-up/patience/collapse gate is added to this stopping rule. Report the last stopped epoch. Historical best-RI logging does not select the returned model. Final inference uses only the original encoder and active original centers, without Gumbel noise. Known k comes from class metadata; labels do not enter the training loss, although historical evaluation computes metrics every epoch.

## Engineering vs Methodological Changes

Categories: **A** = engineering/reproducibility without intended objective change; **B** = faithful documentation of executed behavior; **C** = changed model, objective, initialization policy, stopping or data transformation requiring evidence. A repair can still change the cohort or stochastic trajectory. “Engineering” does not authorize treating new scores as recovered old results.

| Audit 1 finding | Classification and prescribed consequence |
|---|---|
| F01 ST estimator | B: write the explicit surrogate and qualify exact-gradient claims. A different estimator is C. |
| F02 sigma | B: record sigma=1 and distinguish it from tau. Tuning/changing it is C. |
| F03 contrastive masks | B: correct printed denominators to match code. Changing code to the wrong printed masks is C. |
| F04 n/k bound | B: cluster sums run over k, instance sums over B. |
| F05 reductions/alpha | B: document BL/Bd means, summed directions and alpha. Rescaling/reweighting is C. |
| F06 extra softmax | B if described faithfully; removal/replacement is C and must be a named arm. |
| F07 entropy notation | B: fix grouping/sign notation; no implemented sign bug is established. |
| F08 final decoder block | B: disclose the dropout/LayerNorm exception. Adding them is C. |
| F09 centroid correspondence | A for measurement; C for sharing, matching or coordinated initialization. Risk, not confirmed defect. |
| F10 checkpoint/path defect | A: save both views, active centers, optimizer/scheduler/RNG/config and seed-specific artifacts. Missing historical states cannot be recreated by editing save code. |
| F11 TSV parser | A: header=None, canonical counts and row IDs. Audit 2 confirms standard-input loss of two records. New common-cohort comparisons require new results. |
| F12 normalization guards | A: assert finite input/output and diagnose zero variance/range. Imputation, deletion or an alternate constant-series mapping is a separately declared data-policy change, not a silent repair. |
| F13 permutation distribution | B: uniform 1–4 equal segments. Changing distribution is C. |
| F14 warp monotonicity | A: coordinate diagnostics; B: disclose risk; monotone repair is C. |
| F15 two augmentation snapshots | B: disclose both fixed arrays. Consolidating them or resampling online is C. |
| F16 temperature schedule | B: exponential schedule with floor. A different schedule is C. |
| F17 stopping | B/verified: retain ordered deterministic full-pool change rule. Additional patience, warm-up or utilization gating is C/protocol change. |
| F18 collapse | A: monitor hard utilization; B: no entropy guarantee. Reseeding, balancing or new penalties are C. |
| F19 M5 filter | A in repair intent, but potential cohort/k changes require a traced data decision and matched reruns if adopted. |
| F20 M5 transformations | B: weekly STL trend+seasonal and per-series min-max. A new transform is C. |
| F21 Store Sales provenance | A: seek original recipe/artifacts. An invented replacement pipeline defines a new experiment; do not call it recovery. |
| F22 timings/convergence | A: fresh reproducible records; B: qualify old scalar/curve evidence and axis issues. No missing uncertainty can be inferred. |
| F23 NMI mismatch | B: describe actual arithmetic implementation; A: explicitly record future convention. Historical runtime convention is not fully proven. Recompute metrics from predictions if available; metric definition is distinct from model training. |
| F24 omitted ARI/std | B: report available metrics and uncertainty accurately; A: persist new per-seed records. Historical ARI/std are unrecovered. New sample SD uses ddof=1; old aggregation used ddof=0. |
| F25 percentage claims | B: distinguish percentage points from relative percent change. |
| F26 Transformer exception name | B: ECG5000, not EOGVerticalSignal, is the second favorable iTransformer table case. |
| F27 dense scaling/bottleneck | B: linear in L at fixed width, length-dependent, fixed-length; d is not always smaller than L. Variable-length redesign is C. |
| F28 scope/independent weights | B/verified: univariate and independent views. Shared encoders/multivariate extension is C. |
| F29 supplied settings | B/verified: record defaults; A: serialize exact runtime config. Hyperparameter changes are C/protocol changes. |
| F30 oracle k | B/verified: known-k evaluation; A: isolate labels from fitting/checkpointing. Unknown-k evaluation is a new task. |
| F31 inference | B/verified: original-view deterministic assignment. View averaging or stochastic inference is C. |
| F32 pretraining | B: exactly 50 epochs, rather than a convergence loop. New pretraining stopping is C. |

**Additional planning observation, not a retroactive Audit 1 finding:** CORE-40 contains tasks smaller than B=128; Beef has N=60. With drop_last=True the inherited loader produces zero batches and training cannot proceed. Predeclare **B_eff=min(128,N)**, retaining drop_last=True for all three arms. This preserves B=128 on every historical task but extends the protocol for small-N tasks. Batch cardinality affects contrastive denominators and occupancy; disclose it rather than claiming universal numeric equivalence. Assert at least one training batch.

Legacy-Clean engineering preparation also includes an environment manifest, explicit sklearn initialization semantics, record/checksum manifests, two saved augmentation snapshots, run-local RNG state, independent output paths and an inference round-trip check against the live active centers. Keep dormant-center removal optional: it must not accidentally shift initialization/RNG consumption. None of these changes is implemented in this audit.

## Extra-Softmax Decision

Put D=e+k−1, c=e−1, a=e/D, b=1/D and delta=c/D. In a hard forward row, the selected class receives
\[
a=\frac{e}{e+k-1},
\]
and every unselected class receives
\[
b=\frac{1}{e+k-1}.
\]
The selected/unselected ratio is exactly e. Both forward probabilities are independent of tau. At large k, the selected probability approaches e/k: S0 is not a confident hard-assignment representation.

For hard occupancy f_j=B^−1 sum_i H_ij,
\[
S=b{\bf1}+\delta H,\qquad
\bar p_j=b+\delta f_j=(1-\delta)/k+\delta f_j.
\]
This mixes occupancy strongly toward uniform, with delta approximately (e−1)/k. Empty hard clusters still have positive mass b.

**Column geometry.** With original/augmented hard column counts n_j^o,n_r^a and co-assigned intersection m_jr,
\[
\cos(S_j^o,S_r^a)=
\frac{B+c(n_j^o+n_r^a)+c^2m_{jr}}
{\sqrt{[B+(e^2-1)n_j^o][B+(e^2-1)n_r^a]}}.
\]
Distinct same-view columns have m=0. For balanced occupancy their cosine is
\[
\frac{k+2(e-1)}{k+e^2-1}\longrightarrow1.
\]
It is approximately 0.948 at k=50. Two hard-empty columns are identical nonzero constants with cosine 1. If every sample chooses one center, all columns are constant across samples and every column cosine is 1. S0 can therefore make negative columns geometrically very similar, especially at high k; no performance consequence is established without testing.

**Entropy.** Every row has the same entropy
\[
H(S_i)=\log D-e/D
\]
regardless of assignment confidence. The marginal entropy difference between uniform hard occupancy and complete hard collapse is only
\[
\log k-\log D+e/D\sim1/k.
\]
S0 marginal entropy can consequently look almost maximal while hard assignments have completely collapsed. Log hard entropy separately. The implemented negative entropy encourages balance; the smoothing weakens its interpretation as hard occupancy.

**Gradients.** The actual S0 path is
\[
\frac{\partial L_{\rm cluster}}{\partial\ell_i}
=J_Y^\top J_S^\top\frac{\partial L_{\rm cluster}}{\partial S_i},
\quad J_Y=(\operatorname{diag}Y-YY^\top)/\tau,\quad
J_S=\operatorname{diag}S-SS^\top.
\]
It continues through RBF distances to encoder weights and active centers. K-means also has direct residual gradients. A center unused by hard assignments can receive indirect assignment gradients.

The affine formula b+delta H describes hard forward values only. Replacing the actual softmax by this affine expression changes backward semantics; J_S is not delta I. Its nonzero eigenvalues are b (multiplicity k−2) and kab, but cosine normalization can compensate scale. Do not infer a universal reduction in total gradient magnitude from the outer Jacobian alone. At very low tau, the relaxed derivative can saturate or become concentrated; annealing does not progressively harden an already-hard forward pass.

**Lineage.** Official Contrastive Clustering has one softmax in its learned cluster projector and consumes those probabilities directly in the loss; it does not softmax hard Gumbel assignments again. [CC network](https://github.com/XLearning-SCU/2021-AAAI-CC/blob/main/modules/network.py), [CC loss](https://github.com/XLearning-SCU/2021-AAAI-CC/blob/main/modules/contrastive_loss.py).

Official DTCC-2023 applies softmax inside cluster_loss, but its inputs are spectral indicator matrices updated through SVD/orthogonality, not hard one-hot Gumbel outputs. The local [DTCC losses](../../models/DTCC/utils/losses.py) and experiment preserve this spectral-input pattern. LoSTer reuses a DTCC-style operation with different input semantics. Historical copying intent is not established. [Official DTCC source](https://github.com/07zy/DTCC/blob/main/dtcc.py).

S0 avoids zero column norms and log(0), providing a numerical smoothing rationale. No strong theory establishes the particular ratio e, its k-dependent smoothing, or an optimality guarantee.

| Formulation | Concept, gradients and entropy | Hard narrative, stability and novelty | Pilot decision |
|---|---|---|---|
| S0 softmax(Q_ST) | Legacy biased ST plus outer softmax; entropy is smoothed occupancy, not hard occupancy. | Hard k-means/inference but smoothed cluster regularizer; finite columns/logs; no primitive novelty. | Required control. |
| S1 Q_ST directly | Cluster vectors represent actual hard memberships; only ST relaxation mediates this branch. Entropy is hard occupancy with declared zero convention. | Most direct hard-column interpretation; empty columns require finite surrogate derivatives. Removal is a methodological simplification, not a new primitive. | Include. |
| S2 relaxed Gumbel Y | Differentiable stochastic soft-column loss; depends on tau in forward and backward. Keep hard Q for k-means to isolate this branch. | Mixed hard/soft regularization; low-tau underflow/saturation remain possible. Related to established relaxed assignment methods. | Reserve; no initial run. |
| S3 RBF p=exp(ell) | Deterministic soft probabilities; direct distance gradients, no Gumbel noise in cluster branch; entropy is RBF marginal. | Hard k-means with soft contrast; distance saturation remains possible. Close to prior probability-head contrast, weakening a fully-hard objective description. | Reserve; no initial run. |

**S1 must be completely specified before running.** Simply deleting two softmax calls makes current entropy produce NaN at zero occupancy and can expose enormous empty-column cosine derivatives. Define
\[
E_\epsilon=\sum_{v,j} f_j^v\log\max(f_j^v,10^{-8}),\quad
s(x,y)=\frac{x^\top y}{\max(\|x\|_2,1)\max(\|y\|_2,1)}.
\]
For a hard column, any nonempty norm is at least 1 and positive occupancy at least 1/B. Thus these conventions preserve ordinary occupied-column cosine and hard-forward negative entropy, with 0 log epsilon=0, and bound zero-column derivatives. The count-unit norm floor replaces the inappropriate default tiny floor for this candidate. Declare these surrogate conventions as part of S1; do not silently skip empty anchors, add pseudocounts, reseed centers or introduce balancing. Retain all k anchors and identical coefficients.


## Centroid Alignment Decision

The column loss pairs index j with index j across views. KMeans label IDs are arbitrary; independent fits need not give equivalent groups the same number. The instance objective and joint training may establish correspondence, so initial mismatch is plausible rather than a confirmed failure. Final RI/ARI/NMI are permutation-invariant and use only the original view; they do not themselves require view matching.

| Design | Coherence, freedom and identity | Cost, complexity and decision |
|---|---|---|
| C0 independent/no matching | Historical design; allows distinct latent geometries but initially assumes same-index correspondence. | No extra operation. Required control; monitor agreement and occupancy. |
| C1 one shared center matrix | Enforces shared center IDs and coordinates. Sharing removes independent center-row permutations, but does not by itself align independently pretrained latent geometries or guarantee corresponding semantic groups. Constrains geometry and removes kd active parameters. | Little arithmetic overhead but greater initialization/design drift. A reserve alternative, not an initial arm. |
| C2 independent + matching | Retains each view's geometry while making initialization IDs correspond according to paired records. Matching based on cross-view membership is valid without comparing coordinates in incompatible spaces. | One contingency and one assignment solve are inexpensive. Include C2-once only; repeated matching changes semantics and complexity. |
| C3 coordinated/copied initialization, independent updates | Same seed alone already provides weak coordination. Copying original center coordinates into another pretrained latent basis need not align semantics. A coordinated record partition with separate view-specific mean centers is more coherent but changes fitted initialization. Copying encoder weights as well introduces another change. | Numerically cheap; more confounded than a permutation-only test. Exclude initially. |

**C2-once contract.** After both ordinary KMeans fits, form deterministic nearest-center labels on corresponding X and A_init rows. Construct M_jr as the number of rows with original label j and augmented label r. Solve
\[
\pi=\arg\max_{\pi\in\mathcal S_k}\sum_j M_{j,\pi(j)}.
\]
Reorder the augmented center rows according to pi, then freeze this correspondence for the entire joint fit. Record M, pi, agreement before/after, and a deterministic tie convention with fixed solver version/row ordering. No ground-truth labels are used. Do this before joint optimizer construction; there are no accumulated optimizer states to permute. Reorder any corresponding diagnostic/dormant copies consistently.

The operation costs O(N+k^3) after the existing assignment calculation. It changes initialization, not the algebraic form or number of centers; it remains Category C because the coupled loss is not invariant to permuting just one view. It preserves the historical two independent encoders and two independent updates.

Do not use Euclidean center-to-center matching across unaligned latent bases. Do not put Hungarian matching naively into the differentiable loop: the optimal permutation is discrete, switches at ties, and changes the chosen positives. A piecewise objective could be defined, but an unrelated contingency match has no ordinary smooth gradient and cannot be portrayed as differentiable alignment. Repeated optimizer-state permutations and adaptive positive routing are additional methodological choices.

Only **C2-once** is recommended for the initial centroid pilot. C1 is the single reserve option if evidence later warrants another design. No C1/C3 arm, online matching, or S1+C2 combination is in the present three-method pilot.

## Collapse Diagnostics

Prefer **monitoring**. Existing negative entropy encourages balance but provides no guarantee; S0 occupancy obscures unused hard centers. No evidence yet justifies reseeding, forced minimum sizes, extra balancing, changing k or rejecting batches.

At every joint epoch, in eval mode, compute original-view full-pool assignments and augmented-view assignments on the fixed A_train pool. At initialization also record A_init. Do not generate another augmentation. Log separately for each view:

- Counts n_j for all k centers, occupied count, utilization occupied/k, minimum count including zeros, minimum positive count, maximum fraction max_j n_j/N.
- Hard marginal entropy −sum f_j log f_j with 0 log 0=0, normalized entropy H/log k, and effective cluster count exp(H). Also log S0 and RBF marginal/row entropy under separate names.
- Per-center cumulative hard use, dead-center durations, center norms/displacements, nearest-center distances and minimum pairwise center separation. Indirect gradients do not imply hard utilization.
- Consecutive deterministic full-pool assignment-change fraction; first appearance and duration of empty centers; exact stopping trigger and epochs/updates.
- Original/augmented same-index agreement, contingency/permutation-invariant agreement for diagnostics, and sampled hard batch occupancy. Diagnostic matching must not affect training.
- Component losses, encoder/decoder/center gradient norms, total norm, finite checks and numerical failures. Record tau, learning rate, effective batch size and processed records/steps.

A minibatch-empty column is expected at large k and is not a failure gate. Full-pool emptiness is different. Define **complete collapse** as one occupied center at the final deterministic check on either view, for these k>1 tasks. Also flag maximum fraction ≥0.98 as near-collapse for investigation, without automatically rejecting it: class imbalance can be real. Record all empty full-pool centers rather than claiming utilization must equal k.

Monitoring does not delay historical stopping. A stable collapsed solution may stop early and must remain visible. The selection rule assesses such events after fitting. Reseeding, center deletion, balancing, utilization-conditioned stopping and retries until a good seed appears are interventions and are excluded.

## Augmentation Decision

Choose **A: preserve historical augmentation exactly in the initial method-selection pilot**, including both distinct fixed snapshots. This isolates S1 and C2 instead of confounding them with a new data transformation. Sign inversion and segment permutation are not assumed label-preserving across every domain; their empirical adequacy remains a limitation to investigate.

Add generation-time diagnostics without changing the sampled coordinates: finite outputs, endpoint scale, minimum consecutive coordinate difference, number/fraction of non-increasing steps, fraction of series affected, and clipping frequency. Preserve the arrays and coordinate summaries for each seed. Gaussian knot perturbations and cubic interpolation do not enforce increasing coordinates; clipping does not establish strict monotonicity. This is a real mathematical risk, but its frequency in the future arrays is not yet measured. NumPy documents the increasing-coordinate requirement. [NumPy interp](https://numpy.org/doc/1.22/reference/generated/numpy.interp.html).

**B: minimal monotone warp** becomes a focused, separately declared validity sensitivity if invalid coordinates occur and temporal-warp claims are retained. A repair must explicitly define positive increments, endpoint mapping and interpolation, preserve the other transforms, and be labeled Category C. It must not be introduced under Legacy-Clean by sorting, resampling until acceptable or silently consolidating snapshots. If the validity problem prevents a defensible freeze, report that the initial pilot is insufficient and amend the plan before final training. A later augmentation ablation is not permission to invent another initial S/C candidate.

**C: redesign** is not justified now. No jitter/cropping/mixing family is added.

An exhaustive augmentation grid is **not necessary** for the bounded efficiency/integration claim. A transform-removal sensitivity is desirable, especially for sign-sensitive physiology. A minimal validity comparison becomes necessary if the observed interpolation inputs violate the intended warp and the chosen paper claims a valid temporal deformation. Strong claims that these augmentations are universally beneficial would also require evidence; remove those claims if that cost is not incurred.

## Normalization Decision

Retain UCR per-series z-normalization and M5 per-series min-max after historical preprocessing. Describe them as **dataset preprocessing within the complete protocol**, not a novel LoSTer layer. Record dtype, population standard deviation, transformation order and checksums; fit a new model per fixed-length dataset.

These transforms remove positive affine level/amplitude information. That fact alone does not establish harmful information loss or warrant changing the method. Beef spectroscopy and ECG signals give plausible amplitude concerns, but neither domain name establishes that class separation depends on absolute amplitude. Do not label them verified amplitude-sensitive cases.

No normalization arm enters the initial pilot. An amplitude-preserving ablation is necessary only for a verified amplitude-dependent use case or an amplitude-related generality claim; otherwise it is desirable and optional. Any such study must use the same records and apply its declared base preprocessing to all compared methods. Do not exploit target labels to choose a normalization per final dataset.

Assert nonfinite inputs, missing values and zero variance/range before fitting. The initial contract is **fail with an explicit diagnostic** for undefined normalization, not silently remove rows, impute, or map constants to zero. Metadata eligibility does not prove all arrays satisfy these checks. If an edge case occurs, resolve and preregister a common data policy before the affected comparison; never quietly reduce the fixed benchmark list.

## Loss-Weight Decision

Choose **A: preserve historical coefficients, reductions and alpha=1**. Keep alpha explicit in the specification and configuration; it is not tuned.

| Option | Decision |
|---|---|
| A historical unit coefficients | Initial pilot and default final contract. Smallest search burden and clearest continuity. |
| B new lambda_rec/km/inst/cluster | Exclude: adds several degrees of freedom without evidence they are needed. Naming fixed coefficients for exposition is harmless; tuning them is not. |
| C new reductions/normalization | Exclude: changes relative influence with L,d,B,k and breaks the isolated pilot. |
| D tune existing alpha | Reserve only if logged evidence identifies a material problem; no optimization now. |

Heterogeneous reductions are not inherently erroneous. Their empirical relative scales and gradient contributions matter; scalar loss magnitudes alone do not prove dominance.

On one fixed, already-existing training batch at zero-based epochs **0,1,4,9,19,39,59,79,99**, when reached, obtain component gradients from the same stochastic forward graph using autograd.grad with graph retention. Components are R, alpha*K, I, cluster contrast C, and entropy E, with K already including the half factor. Group parameter norms by original/augmented encoder, decoder and active centers; treat unused component/group pairs as zero and distinguish nonfinite gradients.

Record component values, group L2 gradient norms, total norm and, if useful, gradient cosines/cancellation. Then perform the ordinary total backward/optimizer update. Do not consume another dropout/Gumbel draw, alter .grad during measurement or take an extra step. Keep sparse diagnostics lightweight, record overhead, and disable them in the separately instrumented efficiency measurement. No inference about an optimal weight follows before observing these logs.

## Candidate Pilot Variants

Exactly three total methods:

| Candidate | Exact delta from Legacy-Clean | Hypothesis, continuity and naming |
|---|---|---|
| **LoSTer-Legacy-Clean** | S0/C0; correct canonical ingestion, complete artifacts/logging, explicit runtime settings and declared small-N batch extension only. No intended objective change. | Historical mathematical control on new correct cohorts. Can remain the main LoSTer method. New scores are not recovered rejected-paper numbers. |
| **LoSTer-NoResoftmax** | S1/C0: cluster loss/entropy use Q_ST directly with the stated entropy log floor and count-unit cosine norm floor. K-means, instance loss, networks, centers, coefficients, views and schedules unchanged. | Tests whether direct hard membership gives useful quality/utilization/stability without arbitrary k-dependent smoothing. Preserves the hard-assignment integration; if selected, call it revised LoSTer and disclose the change. No claim of a new contrastive primitive. |
| **LoSTer-AlignedInit** | S0/C2-once: label-free contingency matching reorders augmented centers once; independent view geometries/updates remain. No shared weights or centers. | Tests whether arbitrary initial label correspondence materially harms joint optimization. High continuity, but explicitly a revised initialization policy if selected. No differentiable matching claim. |

There is no relaxed/RBF/shared-center arm, combined S1+C2 arm, changed augmentation, weight search or collapse intervention. Negative or null outcomes remain useful: they justify a documented legacy method rather than adding unsupported machinery.

## Pilot Dataset Set

Pooled N means complete TRAIN+TEST records, not the N−2 buggy-parser cohort. Metadata comes from the [official UCR summary](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv). Modern memberships were established in Audit 3 against final publication tables/author code. Historical numbers are rounded manuscript table evidence, not verified new runs; ARI was not recoverable.

| Dataset | N | L | k | Scientific inclusion reason | Historical LoSTer evidence | Verified recent overlap |
|---|---:|---:|---:|---|---|---|
| SyntheticControl | 600 | 60 | 6 | Very short simulated series; favorable historical NMI but an RI exception. | NMI .7862; RI .8698, below historical DTC .8763. | CDCC, FCACC, TFMCC |
| Beef | 60 | 470 | 5 | Tiny-N spectroscopy; stresses full-batch behavior and possible amplitude concerns. | Not in original 17; no score assumption. | CDCC, TFMCC |
| ECG200 | 200 | 96 | 2 | Small-N, short, few-cluster physiological task. | Not in original 17; no score assumption. | CDCC |
| OSULeaf | 442 | 427 | 6 | Image-outline domain; historically low absolute clustering NMI. | NMI .2286, though stronger than old table competitors. | CDCC, TFMCC |
| ShapesAll | 1200 | 512 | 60 | Many clusters; stresses S0 smoothing and empty minibatch columns. | Not in original 17; no score assumption. | CDCC, FCACC, TFMCC |
| SemgHandMovementCh2 | 900 | 1500 | 6 | Long physiological series; historically weak absolute NMI. | NMI .2508; historical RI .7519. | CDCC, FCACC |
| CinCECGTorso | 1420 | 1639 | 4 | Long historical continuity task; low absolute NMI despite table-relative advantage. | NMI .2812; RI .6906. | None of these three fixed subsets |
| StarLightCurves | 9236 | 1024 | 3 | Large-N astronomy; prior adapter-favorable counterexample and evaluation-cost stress. | LoSTer NMI .6024 vs iTransformer .6349; RI .7661 vs .7695. | None of these three fixed subsets |

Sources for exact modern overlap: [CDCC final paper](https://ojs.aaai.org/index.php/AAAI/article/download/28740/29426), [FCACC final publication](https://doi.org/10.1016/j.patcog.2025.112899), [TFMCC final paper](https://ojs.aaai.org/index.php/AAAI/article/download/39817/43778). TSGCC reports full-128 coverage, but highlighted-36 membership/protocol remains unresolved. APCL, DEETO and diffusion-DTCC exact overlap is not established. Do not infer it.

This eight-task set spans N=60–9236, L=60–1639 and k=2–60 with several domains. Three new-to-LoSTer tasks prevent a purely favorable-history selection. Historical strong/weak and Transformer-favorable behavior is deliberately used for development coverage, as requested; it is not a final benchmark selection rule. Beef is only a plausible amplitude concern, not evidence that z-normalization harms it.

## Pilot Protocol

Use **five paired seeds {0,1,2,3,4} from the outset for every arm**. Three would cost 72 joint fits but offer weaker seed-stability evidence; five costs 120. Avoid staged score-driven seed/arm expansion and the associated complexity.

**Data and training contract.** Read canonical headerless TSVs with header=None; verify official TRAIN/TEST/pooled counts, fixed L, labels, row order and checksums. Concatenate the two splits for transductive clustering. Normalize each series as specified. Set k from declared class metadata; keep labels outside optimization and checkpoint selection. Use B_eff=min(128,N), shuffle=True/drop_last=True for training, and full ordered non-dropping evaluation. Record incomplete-batch row exposure across epochs; do not silently change drop_last to False.

Pretrain both independent views for 50 epochs at Adam 0.001. Joint ceiling 100 epochs with original SGD/StepLR/tau/alpha/stopping. No quality metric chooses an epoch, hyperparameter or retry. Save the last stopped state. Use explicit n_init=1. Do not substitute a ten-restart KMeans implementation.

**Paired initialization.** Per dataset/seed, generate and save X,A_train,A_init; initialize and pretrain independent AE views once; cache both pretrained states and both KMeans fits. Replay identical states, wrapper construction and joint-boundary RNG state for each arm. AlignedInit changes only the declared center permutation. Identical integer seeds alone do not guarantee identical dropout, Gumbel and shuffle streams when code consumes randomness differently; isolate/replay streams and document any unavoidable numerical divergence. Preserve the historical two-snapshot order rather than unifying arrays for convenience.

**Metrics.** Primary ARI. Secondary
\[
\mathrm{NMI}_{arith}=2I(y;\hat y)/(H(y)+H(\hat y)),
\]
using an explicit library convention, including its degenerate-label handling. RI is compatibility/descriptive evidence, not the main selector because pair-count agreement is sensitive to cluster structure. Evaluation-only ACC is worthwhile:
\[
\mathrm{ACC}=N^{-1}\max_\pi\sum_i{\bf1}[y_i=\pi(\hat y_i)].
\]
Its ground-truth Hungarian assignment is distinct from label-free centroid matching and never enters fitting. Record per-seed predictions/contingencies and metrics at the stopped epoch. Report mean ± **sample SD, ddof=1**, all seed values, medians and failure counts; never infer missing historical ARI/SD.

**Diagnostics and resources.** Log the collapse/gradient/warp fields above, convergence trajectories, stop reasons, finite failures, effective batches, epochs and updates. Record trainable/deployed parameters, wall-clock full fit and stage times, peak GPU allocated/reserved memory and host memory. A failed run stays in the record; do not replace seeds until favorable. Distinguish transient/minibatch emptiness from complete full-pool collapse.

Full-pairwise silhouette computation is historical telemetry, can be expensive on StarLightCurves, and does not affect stopping. It may be omitted or fixed-sample measured in a disclosed telemetry configuration without consuming the training RNG; retain the exact mathematical/stopping behavior. Do not confuse telemetry-inclusive legacy runtime with algorithm time in a revised comparison.

Cache reuse yields **80 view-specific pretraining jobs** (8×5×2) plus **120 joint jobs**. Each candidate conceptually has a complete fit; publish standalone cost including both pretrains and initialization, not the amortized cost of running three cached pilot arms. The pilot produces method-selection evidence, not final-publication seed statistics.

## Predeclared Selection Rule

These are prospective **practical screening thresholds**, not a powered superiority/noninferiority test or universal literature standard. A one-score-point broad gain is a minimum useful effect for this pilot; a three-point task loss is a material warning. Register them before seeing pilot scores.

For each alternative define
\[
\Delta_d=\frac15\sum_s(\mathrm{ARI}_{candidate,d,s}-\mathrm{ARI}_{legacy,d,s}),\quad
\overline\Delta=\frac18\sum_d\Delta_d.
\]
Every dataset receives equal weight; large N does not dominate. Apply this sequence:

1. **Eligibility and conceptual gate:** all five declared runs on all eight tasks have complete artifacts and finite required computations, and the observed implementation is exactly the named candidate. This completeness requirement first applies to Legacy-Clean: if any required legacy score/artifact is unavailable, paired deltas and stability comparisons are uninterpretable; make no adoption decision and resolve the failed protocol in a newly declared step, never use only surviving pairs. An alternative's numerical failure disqualifies its adoption in this pilot. Its improvement must address its stated hypothesis, not an undocumented workaround.
2. **Safety/consistency gate:** no new final complete-collapse event on either view relative to the paired legacy run; no dataset has Delta_d < −0.03; macro arithmetic NMI loss is no worse than 0.01. Record underutilization/near-collapse but do not equate expected class imbalance or minibatch zeros with failure.
3. **Require one substantive route:**
   - **Quality:** macro Delta at least +0.01, positive Delta on at least 6/8 tasks, and Delta at least +0.01 on at least three tasks.
   - **Stability:** macro Delta at least −0.01, together with either (a) lower ARI sample SD on at least 6/8 tasks and mean SD reduction at least 0.01, or (b) elimination of at least two paired complete-collapse events across at least two tasks without new events. The NMI and per-task gates still apply.
4. **Cost and simplicity:** report standalone fit-time medians, peak memory and parameters. Reject domination by legacy only when the alternative has no greater macro ARI, no smaller mean ARI SD, no fewer collapsed fits and no lower median complete-fit time, with at least one strict disadvantage. A qualifying useful improvement that costs more is a measured tradeoff, not subject to an invented universal 20% runtime veto.
5. **Resolve the candidate choice deterministically:** with a valid legacy control, if neither alternative passes steps 1–4, retain Legacy-Clean; if exactly one passes, select it. If both pass, compare in this fixed order and stop at the first decisive item: (a) fewer dataset-seed fits with complete collapse on either view, counting each fit once; (b) larger macro ARI if the difference is at least 0.01; (c) lower mean per-dataset ARI SD if the difference is at least 0.01; (d) more datasets with positive paired Delta_d; (e) lower median complete-fit time if the slower/faster ratio is at least 1.05; (f) otherwise select NoResoftmax as the simpler removal of unexplained smoothing. The comparable bands for (b),(c),(e) are explicitly differences below those thresholds. Cost is a tie-break after substantial quality/stability evidence. A tiny gain without either route in step 3 therefore retains legacy. Do not construct a combined winning arm.
6. Freeze the selected code/config/data/seeds and all failed-run records before examining the other 36 tasks. Do not select again, tune a threshold or fall back based on their rankings.

Display sensitivity of the broad mean-effect criterion at 0,0.005,0.02 as a robustness description, without changing the registered decision. Five seeds/eight tasks give exploratory selection evidence only. Historical scores or previous reviewer expectations are not additional gates.

If an alternative is selected, require a fresh Legacy-Clean reference on the **36 non-pilot tasks × five final seeds** to test whether the change generalizes. This is 180 additional fits, not a fourth method. Its results do not trigger adaptive method reselection; poor transfer must be reported.

## CORE-40 / EXTENDED-44

**CORE-40 is exactly CDCC's fixed 40-dataset selection verified in Audit 3**, with official canonical UCR names. This provides a reproducible direct-competitor anchor, varying sample sizes, cluster counts and domains, without inspecting LoSTer outcomes.

| 1–10 | 11–20 | 21–30 | 31–40 |
|---|---|---|---|
| ACSF1 | ECG200 | GunPointOldVersusYoung | PowerCons |
| Adiac | ECGFiveDays | HouseTwenty | ProximalPhalanxTW |
| ArrowHead | EOGVerticalSignal | LargeKitchenAppliances | SemgHandMovementCh2 |
| Beef | FaceFour | MiddlePhalanxOutlineAgeGroup | ShapeletSim |
| Car | FiftyWords | MiddlePhalanxTW | ShapesAll |
| CBF | Fish | OSULeaf | SwedishLeaf |
| CricketX | Fungi | PigAirwayPressure | SyntheticControl |
| CricketY | GunPointMaleVersusFemale | PigArtPressure | ToeSegmentation1 |
| CricketZ | DistalPhalanxOutlineAgeGroup | PigCVP | Trace |
| DiatomSizeReduction | DistalPhalanxTW | Plane | WordSynonyms |

Columns are only a display arrangement: the exact set is the 40 unique names above. Canonical alphabetical order may be used in the execution manifest. The source list in Audit 3 is authoritative; do not infer an execution order from the table.

**EXTENDED-44 = CORE-40 union {CinCECGTorso, StarLightCurves, MixedShapesRegularTrain, MixedShapesSmallTrain}.** These four additions extend historical long-series continuity; they were not selected for favorable scores. The set is fixed before the pilot and cannot be enlarged selectively after results.

Seven CORE tasks have L≥1024: ACSF1, EOGVerticalSignal, HouseTwenty, the three Pig pressure tasks, and SemgHandMovementCh2. Four additions raise this to eleven of 44, while short tasks remain legitimate stress tests for the scope of the dense representation. All 40 core metadata entries are fixed-length numerical tasks; array-level finite/constant checks remain a future prerequisite. No silent exclusion/substitution is permitted after seeing scores.

CORE contains **6 of the original 17**: EOGVerticalSignal, SemgHandMovementCh2, OSULeaf, MiddlePhalanxOutlineAgeGroup, ProximalPhalanxTW, SyntheticControl. EXTENDED contains **10 of 17**, not all 17. The seven absent names are NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, UWaveGestureLibraryX, UWaveGestureLibraryY, ECG5000, Symbols and ProximalPhalanxOutlineAgeGroup. Historical evidence can be discussed separately; rerunning the other seven is optional continuity work, never an implicit enlargement of the required campaign.

Six pilot tasks are in CORE; two are additions. Thus CORE alone would leave **34** non-pilot tasks, while EXTENDED leaves **36**. Prefer EXTENDED-44 and do not change to CORE after unfavorable addition scores. All44 is development-inclusive; non-pilot36 is the primary fixed comparative panel. Public/historical knowledge and related task families mean even that panel is not a pristine unseen population sample.

Why not all 128? Full-archive evaluation increases source adaptation, missing/variable-length handling and fit costs while exceeding the fixed-length claim. Forty-four yields direct modern overlap, long-series continuity and a manageable protocol. It does not justify universal archive superiority. FCACC/TFMCC have different 40-task sets; no claim is made that their published tables match this exact set. APCL overlap remains unresolved; TSGCC's full-128 claim alone is not an artifact/protocol verification.


## Baseline Tiers

The tiers deliberately narrow Audit 3's broader literature-driven mandatory list. Scientific coverage and reproducibility matter more than the number of rows. This revision cannot claim superiority over excluded modern graph, topological, diffusion or adaptive-k methods.

**Tier 1: required main-panel comparisons, subject to source/protocol validation.**

| Baseline | Distinct question it answers | Artifact/protocol gate and reproduction burden |
|---|---|---|
| Euclidean k-means | Does learned representation earn its complete fit cost over the direct known-k reference? | Freeze standardized pooled-input representation, explicit initialization/restarts and stopping. LOW. |
| k-Shape | Does dense latent clustering improve over an established shift-invariant shape method? | Pin implementation, SBD/normalization/initialization; check constant-series validity. LOW–MODERATE. |
| CKM | Does the contrastive/reconstruction integration improve on the inherited Concrete/hard-assignment clustering mechanism? | Verify equations against the temporal implementation; official author code was not identified. Label as our temporal reproduction or matched CKM control, never official reproduction by implication. MODERATE. |
| DTCC-2023 | Does hard centroid clustering improve on the closest dual-contrastive spectral temporal predecessor? | Full source/config validation; local PyTorch code is an adaptation, official artifact is TF1. Avoid ground-truth best-epoch selection; document common-protocol departures. HIGH. |
| CDCC | Do explicit time/frequency and cross-domain features outperform the dense time-domain integration? | Official author code; canonical-record adapter, pooled normalization and stated stopping. Old Torch environment requires isolation. MODERATE. |
| FCACC | Is fuzzy membership and cluster-aware contrast preferable to hard latent assignment? | Official author code; preserve fuzzy/pair-selection mechanics, audit separate TRAIN/TEST global normalization and override transparently for the common protocol. MODERATE. |
| TFMCC | Is contemporary hierarchical temporal/frequency contrast stronger at acceptable cost? | Official author code/full paper; clarify suspicious dependency pins and long training recipe before implementation is called faithful. HIGH until resolved. |
| TS2Vec + k-means | Is joint clustering needed, compared with strong generic self-supervised representation plus a fixed clustering head? | Official encoder implementation; freeze instance pooling, training budget and KMeans head before final labels. This is an explicitly defined two-stage baseline, not a clustering score from the original classification paper. MODERATE. |
| R-Clustering | Can inexpensive random convolutional features match learned representations? | Official artifact, PCA/feature settings and pooled protocol frozen; retain randomness and licensing. LOW–MODERATE. |
| KASBA | Does efficient elastic clustering outperform neural integration in quality or CPU cost? | Pin verified aeon implementation; adapt the paper's main TRAIN-fit/TEST-predict setting to pooled clustering explicitly, using its pooled evidence where applicable. MODERATE. |
| FASA | Can efficient shape-based centroid updates offer a better quality/cost tradeoff? | Author FASA/MUFASA artifact, correct univariate FASA branch, preprocessing/init/thread settings; resolve code-use permissions separately from the article license. MODERATE. |

Primary artifact references: [DTCC-2023](https://github.com/07zy/DTCC), [CDCC](https://github.com/JiacLuo/CDCC), [FCACC](https://github.com/Du-Team/FCACC), [TFMCC](https://github.com/Du-Team/TFMCC), [TS2Vec](https://github.com/zhihanyue/ts2vec), [R-Clustering](https://github.com/jorgemarcoes/R-Clustering), [KASBA final paper](https://doi.org/10.1007/s10618-026-01189-9), [FASA/MUFASA](https://github.com/TheDatumOrg/MUFASA). Audit 3 records source dates, publication verification, implementation details and unresolved gates. These links establish provenance, not successful reproduction.

CKM and DTCC are lineage controls, not interchangeable: one isolates the inherited hard-assignment primitive, the other the closest spectral/dual-contrastive temporal design. CDCC, FCACC and TFMCC represent different temporal/frequency/fuzzy mechanisms. TS2Vec uniquely tests whether joint learning is worthwhile. R-Clustering, KASBA and FASA provide distinct low-cost feature/elastic/shape challenges.

A failed Tier-1 gate blocks a claim of a complete required panel. Resolve the source or clearly amend the preregistered scope before scores are inspected; do not quietly omit a difficult competitor. A matched control can be useful but cannot substitute unnoticed for a faithful paper reproduction.

**Tier 2: important secondary evidence, not automatic full44 jobs.**

| Candidate | Reason and promotion condition |
|---|---|
| DTCR | Older reconstruction/spectral/recurrent lineage. Useful secondary comparison; TF1 restoration is costly and DTCC already provides the closest main lineage challenge. |
| TSGCC | Modern weighted-neighborhood/graph contrast. Official legacy multistage artifact exists, but its Fish-specific runner and full-archive recipe need resolution before broad quantitative use. |
| DEETO | Topology/eigenalignment contrast challenges a different structure assumption. Verified complete methods/official code are unavailable; abstract-driven reconstruction is unacceptable. |
| diffusion DTCC-2026 | Distinct from DTCC-2023: reliable frequency/diffusion neighbor selection and orthogonality. Exact 26-task list/full protocol remain unresolved. |
| APCL | Essential contemporary prototype/hard-assignment discussion. Partial demo/component code is not full-paper reproduction. Promote to required quantitative evidence only if the revised headline claims novel prototype/hard-assignment methodology. Unknown-k/model-selection results need a separate track unless an author-supported known-k mode is verified. |

[TSGCC author artifact](https://github.com/zololululu/TSGCC), [DEETO publication](https://doi.org/10.1016/j.knosys.2024.112434), [diffusion DTCC publication](https://doi.org/10.1016/j.eswa.2026.133028), [APCL author artifact](https://github.com/William-Liwei/APCL). No new unknown-k claim is proposed here, so APCL remains required discussion/source resolution rather than a speculative costly main reimplementation.

**Tier 3: optional/context only.** DTC, IDEC and DNFCS retain historical context; they add less unique modern coverage than the chosen main set. DTC's author implementation/final publication status requires care; inherited dense IDEC/DNFCS variants should not be represented as identical to every paper implementation.

Preformer, Pathformer and iTransformer are inherited forecasting-encoder **adapters**, not native clustering methods. Their old scores remain contextual. Pathformer's length truncation and iTransformer's one-variate attention semantics require controlled interpretation; do not revive a universal attention claim from these results. Zero adapter reruns are required for the preferred narrative. One preregistered PatchTST architecture control is optional if a specific temporal-architecture claim is retained. RandomNet is optional inexpensive-representation context, with substantial Windows/environment effort and overlap with R-Clustering; it is not another automatic Tier-1 job.

Multivariate MUFASA belongs to a different scope from univariate FASA. No UEA track or multivariate extension is required.

## Fairness Policy

| Evidence type | Main quantitative/rank panel? | Required presentation |
|---|---|---|
| Official paper numbers | No, unless the exact raw cohort/protocol/evaluator/run artifacts are verifiably matched; default is separate context. | Attribute the original source, metric, dataset selection and uncertainty. No mixed-protocol ranks or timing ratios. |
| Newly run official implementations | Yes, after source and common-protocol validation. | Label reproduced official code under the stated common protocol; identify loader/normalization/stopping departures and artifact commit. |
| Our reimplementations/adaptations | Yes if equation-to-code validated and explicitly identified. | State what is reproduced, changed or unverifiable; distinguish a matched component control from the full published method. |
| Historical LoSTer-manuscript numbers | Context only. | Preserve unchanged, explain parser/provenance/statistics limits; no invented SD/ARI or aggregation with new runs. |

**Common task.** Use exactly the same full canonical TRAIN+TEST records and IDs, base per-series normalization, known k, evaluation pool and evaluator. This is transductive pooled clustering, not held-out classification or train-to-test prediction. Do not compare KASBA's original split protocol or other paper cohorts as though they were this task. Preserve algorithm-specific internal FFT/wavelet/elastic transforms, learning objectives and augmentations rather than converting every method into LoSTer.

**Label discipline.** Ground truth determines declared k and final evaluation only during fitting. It cannot select checkpoints, retry seeds, choose per-dataset preprocessing or select the final configuration. Remove/disable official best-label-epoch selection under a clearly named common evaluation protocol. Native unsupervised stopping can differ by algorithm, but must be frozen and documented.

The pilot uses ARI/NMI to select a method and therefore is explicitly label-informed development across tasks. Do not call the complete development process label-free. For every method the final-target-label tuning budget is **zero**. Published/default global settings are preferred. If baseline adaptation needs tuning, apply a uniform baseline-adaptation ceiling of **three global configurations × eight development tasks × three fixed seeds {0,1,2}** per baseline; no task-specific score search and no final36 inspection. This optional ceiling is distinct from the five-seed LoSTer method decision, not a claim of equal historical development effort. Publish actual search fits/costs and freeze any global baseline choice before final results.

**Seeds and runtime.** Use five final seeds **{100,101,102,103,104}**, common across stochastic methods. Equal numbers do not create paired initialization across unrelated algorithms; paired dataset means remain valid. Verify determinism before reducing a method to one fit. Never duplicate deterministic outputs to manufacture an SD. Record run seeds and relevant backend/worker RNG settings. Numerical reproducibility is an environment-specific property to verify, not a promise from setting a seed.

**Batches/resources.** Do not force B=128 on every algorithm. Report effective batch, negative population, gradient accumulation, total steps, epochs and resource constraints. Accumulation does not automatically reproduce a larger contrastive denominator. Document method-native batch selection/OOM handling before final scores, using only development tasks; no silent favorable batch search.

Use the same documented host and accelerator for comparable neural runs, fixed thread counts for CPU-heavy methods and isolated compatible environments. Pin versions and commits; do not “fix” old dependency recipes silently. Hard resource ceilings and any dataset failure policy must be declared before the final campaign based on infrastructure/development observations. An OOM/timeout is a reported failure, not a zero clustering score, nor permission to omit an unfavorable task.

**Persistence.** Each new run gets separate cohort/config/seed paths, input/augmentation hashes, both model views and active centers, optimizer/scheduler/RNG states, stopping metadata, raw predictions and per-seed metrics. Preserve all historical artifacts. No test-set checkpoint selection. Restore-and-infer agreement and a small provenance/loader smoke check are prerequisites to later scientific execution; they are not performed in this planning audit.

## Efficiency Measurements

The primary claim concerns **complete quality–cost tradeoffs**, not dense parameter count alone.

| Measurement | Main-paper role | Appendix/supporting detail |
|---|---|---|
| Complete pipeline fit wall time | Required with ARI; stage totals and all essential preprocessing/initialization/pretraining included. | Augmentation, each AE pretrain, KMeans init, joint/staged fit, compilation/start-up and data-loading breakdown. |
| Original-pool inference wall time | Required where meaningful; includes encoder/feature extraction and assignment after fitting. | Batch sweeps, warm/cold distinction, repeated-pass dispersion and record counts. |
| Peak GPU allocated/reserved memory | Required for GPU methods, with batch/device settings. | Per-stage peaks, framework/background effects and inference memory. |
| Host peak memory | Required supporting cost for CPU methods; include in main comparison if it changes practical feasibility. | Sampling method, process tree and peak RSS details. |
| Trainable parameters | Required descriptive model size for neural methods, including both AEs and both active center matrices. | Registered versus objective-active counts, dormant tensors, optimizer/model bytes; deployed original-view count. |
| Epochs/steps to stopping | Supporting convergence descriptor. | Stage-specific epochs/steps/losses and cap-hit fractions. Epochs are not commensurate across algorithms. |
| Samples/sec | Supporting throughput at disclosed B/device. | Training/inference batch timing; never use it instead of total fit cost. |
| FLOPs/MACs | Optional, only for well-defined comparable forward operators. | Scope convention, counted shapes and omitted KMeans/SVD/elastic/preprocessing work. Not a universal pipeline ranking. |

Compute both **registered trainable** and **objective-active** parameter counts. Dormant wrapper centers are registered but do not receive the loss gradients; counting them as active scientific capacity is misleading. Memory measurements still include any allocated dormant tensors. Report complete two-view training capacity and original-view deployed capacity separately. A non-neural method's zero neural parameters is not zero complexity.

Measure full-fit/stage time and memory on every final task/seed. Emphasize all eleven L≥1024 tasks for the long-series claim, and all44 quality/cost summaries. A fixed detailed efficiency panel is those eleven long tasks plus **Beef, ECG200, SyntheticControl, ShapesAll** (15 unique tasks), chosen by size/length/k stress before observing timing. It supports detailed repeated inference/throughput measurements, not replacement of all44 fit-time accounting.

**Timing contract.**

- Use a monotonic wall clock; synchronize the accelerator immediately before and after each timed GPU region. Avoid repeated per-operation synchronization that changes the pipeline being measured.
- Warm the backend in a disposable calibration operation, outside timing, preserving/replaying the measured fit's weights/RNG. Genuine per-fit initialization/compilation overhead belongs in end-to-end cost; separately label steady-state kernel results.
- Five standalone final training seeds provide repeated fit measurements. Do not repeat costly whole CPU fits solely for clock precision when deterministic execution is established; report the available dispersion and method.
- For inference on the fixed panel, use eval mode, three untimed warm-up batches and **five complete timed passes**, synchronizing boundaries. Report median and spread, full-pool milliseconds, per-sample throughput and B; do not mix first-call cold timing with warmed methods.
- Include both 50-epoch pretrains, augmentation and KMeans initialization in LoSTer fit cost. Include every mandatory stage for SCAN/graph/SSL baselines. Shared pilot pretraining is a development saving; stage-summed pilot cost estimates must be labeled as estimates, not fresh standalone observed runtimes.
- Standardize lightweight required logging. Exclude online dashboards, pairwise silhouette and sparse gradient diagnostics from algorithm-only timing while separately recording telemetry-inclusive operational cost. No extra stochastic diagnostic forward may change the fit.
- Record CPU/GPU model, VRAM/RAM, precision, OS, framework/library versions, batch, CPU thread limits and seed. Model-loading/startup and common disk-read time can be separated, but all essential method-specific work remains in full-pipeline cost.

CPU-heavy R-Clustering/KASBA/FASA/k-Shape run on the same documented host with fixed thread policy; do not force them onto GPUs or force neural baselines onto CPUs. Report native-device cost with the resource label. CPU seconds and GPU seconds are practical observations on those devices, not device-invariant algorithm constants or power/cost equivalence. CPU peak memory is not missing just because GPU allocation is zero.

Show per-task quality-versus-fit-time plots, normalized cost ratios with explicit reference and medians, and failure/feasibility counts. Avoid averaging unmatched historical timings, quoting unqualified “under two seconds,” universal complexity claims from one L, training speed from inference-only counts, or MAC totals that omit dominant non-network stages. Do not claim dense superiority over every Transformer from three inherited adapters.

## Statistical Analysis

Freeze this plan before final scores. **ARI is the primary outcome**. Arithmetic NMI is secondary; RI and evaluation-only ACC are descriptive. State NMI arithmetic normalization and SD convention in every new table.

For each stochastic method/task publish all five seed values and mean ± sample SD. The statistical unit is the **dataset**, using seed-mean scores, not 36×5 independent observations. Seed variability and cross-dataset variation answer different questions.

**Primary comparative panel:** 36 non-pilot EXTENDED tasks, all 12 methods (selected LoSTer + eleven Tier-1 baselines), on common complete support. If an alternative was selected, include the required Legacy-Clean reference in this panel, giving 13 methods. Run a Friedman omnibus rank test at alpha=0.05. Follow an omnibus rejection with the **eleven predeclared two-sided LoSTer-versus-baseline Wilcoxon signed-rank comparisons**, correcting that family with Holm at 0.05; the changed-method branch adds the legacy contrast to the same family, giving twelve tests as specified below. Use a fixed zero/tie policy: Wilcox removes exactly zero paired differences; use average ranks for ties, a tie-aware implementation/permutation calculation with a frozen implementation/random seed, and report effective paired n. If all differences are zero, report no evidence of difference rather than an undefined significance claim.

This is a planned control-versus-baselines post-hoc procedure, not every possible pairwise comparison. Report raw and corrected p-values, effect estimates and actual coverage. Do not turn repeated significance tests over every metric or seed into a large uncorrected claim set.

For each contrast, report mean and median paired ARI difference, wins/ties/losses with exact-score ties, and matched-pairs rank-biserial effect size with a declared sign. Provide a dataset-level paired bootstrap interval for mean difference (10,000 resamples, fixed analysis seed), clearly acknowledging task dependence. Inference is conditional on the selected public benchmark, not a random sample from all real-world time series.

**Secondary analyses:** descriptive full44 and pilot8 tables; non-pilot arithmetic NMI effect/rank summaries; historical-versus-new-to-LoSTer subsets; source-family-balanced/leave-one-family-out sensitivities. If formal NMI tests are retained, repeat the registered control comparisons with a separate Holm-corrected secondary family, labeled exploratory and unable to rescue an unsuccessful primary ARI claim. ACC/RI receive no additional primary p-value family.

Related tasks, e.g. Cricket axes, Phalanx variants, Pig pressure channels, Shapes subsets and EOG variants, weaken an independent-task interpretation. Define source families using archive documentation before results; show how conclusions change when correlated families receive equal weight or are removed together. The 36 tasks were not used in the current pilot selection, but public/historical task knowledge persists. Fresh seeds do not erase dataset/family selection history.

If a methodological alternative is selected, its paired final Legacy-Clean comparison on36 is **one separately identified method-change validation**, not another baseline selected after scores. Always report its paired ARI effect/interval. Perform its predeclared two-sided Wilcoxon contrast only after rejection of the 13-method Friedman omnibus, alongside the eleven baseline contrasts, as a **12-test Holm family**. If the omnibus is not rejected, report descriptive effects without post-hoc significance claims. Freeze this branch with the pilot choice before final scores; do not adaptively switch back after seeing confirmation.

A critical-difference/rank diagram is optional. A generic Nemenyi critical-distance bar does not represent Holm-corrected Wilcoxon conclusions; if using Nemenyi, label it as a distinct secondary all-pairs procedure. Prefer average-rank plots and an explicit corrected-comparison table for the registered primary analysis.

If a method fails, publish per-task failures/OOMs/timeouts and feasibility counts. Do not silently discard failed tasks or assign fictional ARI=0. Report descriptive available-case results plus a clearly named conditional common-support analysis; list excluded tasks and changed n. A common-support panel smaller than planned is incomplete primary coverage, not the unchanged 36-task result. Do not repair that limitation by importing unmatched paper scores.

## Retail / Real-World Dataset Decision

| Dataset | Decision | Scientific reason and next evidence gate |
|---|---|---|
| **M5** | **REBUILD**, conditional optional appendix; excluded from the mandatory44 workload. | Historical 1150-row selection/29 labels is recoverable, but the zero-day filter includes store_id and omits the final sales day. Missing raw/boundary data prevents quantifying changed membership. |
| **Store Sales** | **DROP** from new empirical claims; preserve historical files/numbers as historical evidence. | Preparation pipeline, exact selected arrays and run provenance are unavailable. A newly invented recipe would be a new task, not recovery. |

M5 next step is data-only: obtain the relevant raw daily-sales/boundary columns and immutable selected IDs; compare the historical and corrected filtering membership/labels; record checksums, retained sizes, class counts and temporal span; define weekly STL and scaling exactly. Do not assume the filter bug already changed the subset or invalidated published scores.

If the membership is identical, **F19 alone does not require retraining**. Nevertheless, new comparisons and uncertainties may need fresh runs because complete historical model/per-seed artifacts are absent. If membership changes and the corrected definition is adopted, all retained comparators must use that new cohort, with a separately labeled historical-versus-corrected data description. Replace hard-coded size only in later authorized implementation.

Missing M5 raw data means omission from new appendix evidence, without delaying the reproducible UCR core. It is not a universal real-world validation claim. Retail k uses category/store metadata; min-max plus STL studies shape/trend structure rather than proving economic segmentation quality.

No replacement dataset is recommended now and no candidates are researched. Rebuild M5 only if it adds interpretable application evidence at reasonable cost. Do not keep Store Sales merely to retain a “real-world” paragraph, and do not edit its historical assets.

## Final Recommended Campaign

One preferred sequence:

1. Prepare reproducible canonical-data loading, artifacts, explicit runtime settings and sparse diagnostics; resolve Tier-1 source/environment gates using static inspection and later small authorized smoke checks.
2. Run exactly the three named methods on the eight named development tasks, five paired seeds {0..4}, with historical augmentation/coefficients and declared S1/C2 semantics. No new objective grid.
3. Apply the registered eligibility, consistency, quality/stability, collapse and simplicity rule. Resolve any observed warp-validity/data edge case openly before final freeze. Legacy-Clean remains the default; a change needs evidence.
4. Freeze EXTENDED-44, the eleven Tier-1 comparators, all global configurations/evaluators and five final seeds {100..104}. Run the selected method and main baselines on all44; analyze non-pilot36 as primary and full44 as development-inclusive. If a change was selected, also run Legacy-Clean on36 for paired change validation.
5. Complete full-pipeline efficiency measurements and registered corrected statistics, report failures/negative findings, and then rewrite the revision manuscript around observed evidence.

**Scope:** fixed-length, univariate, transductive known-k clustering. No multivariate campaign, unknown-k claim or mandatory forecasting-adapter rerun. Transformers remain limited historical context; at most one optional architecture control needs a specific surviving claim. M5 is conditional appendix rebuilding; Store Sales is excluded from new empirical claims.

**Narrative:** a transparent integration and evaluation contribution, testing whether a simple dense model with hard assignment and contrastive regularization gives an attractive quality–cost tradeoff against contemporary temporal/fuzzy/frequency and efficient non-neural approaches. Evidence may instead show that the integration offers limited benefit or costs more than alternatives; report that outcome. Do not restore rejected broad novelty/superiority claims by changing datasets, metrics or competitors.

## Estimated Workload

Fit counts describe workload structure, not predicted GPU hours. No actual runs or runtime estimates were produced here.

| Work item | Fixed planned work | Workload |
|---|---|---|
| Infrastructure preparation | Canonical loader/manifests, complete checkpoints, RNG replay, environments, metric/timing contract and source validation. No scientific fits in this audit. | **HIGH** |
| Pilot | 3×8×5 = **120 joint fits**, plus **80 view-specific pretrains** cached across arms; 120 conceptual complete pipelines. | **MODERATE** |
| Final selected LoSTer | 44×5 = **220 complete fits**; add **180 Legacy-Clean reference fits** if a changed method is selected. | **HIGH** |
| Tier-1 reproductions | 11×44×5 = **2420 complete fits** at nominal five runs per method/task; old frameworks/source gates dominate engineering effort. | **VERY HIGH** |
| Statistical analysis | Provenance/failure checks, paired summaries, corrected tests, effects and family sensitivity; no new model fits. | **MODERATE** |
| Manuscript rewrite | Exact methods, bounded novelty/scope, fresh figures/tables and negative-result interpretation. Later authorized work only. | **HIGH** |

Nominal final campaign is **2640 complete fits** (220+2420), or **2820** when the selected alternative requires the 180-fit legacy reference. Including the pilot gives **2760** or **2940** conceptual fits. The 80 cached pretraining jobs are a stage-count explanation, not another 80 complete pipelines to add to those totals.

A method proven deterministic before scores can reduce the nominal final count by 44×(5−1)=176; do not fabricate seed SD. No deterministic reduction is assumed in the headline budget. Ten final seeds would double the main campaign to5280 and the conditional36-task reference to360; this is not the preferred plan and must never be triggered by borderline significance.

Optional baseline development has the shared ceiling of72 fits per baseline (3 configurations×8 tasks×3 seeds); all eleven using the ceiling would add792 development fits. Actual use, not the ceiling, must be reported. Optional single-arm eight-task/five-seed ablations add up to40 complete fits each, with reuse only for truly identical states/inputs. Tier-2 promotions, seven missing historical tasks, a Transformer control and M5 are separate scopes with explicit additional cost; none is hidden inside the mandatory budget.

Stop after this report. Audit 4 is not committed or pushed. Scientific implementation, training, manuscript edits and publication require the next task.

