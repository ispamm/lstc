# LoSTer Paper-Code Audit

## Executive Summary

Audit started 2026-10-04 and completed 2026-10-05 (Europe/Rome). Audited HEAD: `0c83687c03d854817112f61dd29dde58903a8ff1`; inherited code tag `original-code`: `b594b650f68ebed80436b8de7313fa1536d3d95c`. This is a static, read-only audit. Only this report was created; no source, manuscript, historical material, or results were changed, and nothing was committed or pushed.

**A. Confirmed implementation/manuscript errors.** The manuscript's instance and cluster contrastive denominators mask the positive cross-view pair rather than the same-view self-pair (F03); the cluster denominator also uses sample count `n` instead of cluster count `k` (F04). Loss reductions do not match the equations: reconstruction is divided by sequence length, k-means by latent dimension, and directional contrastive means are added without the paper's factor 1/2 (F05). The final checkpoint excludes the active trained centroids and retains unused centroid copies (F10). The M5 zero-day filter includes `store_id` and omits the last sales day (F19); whether correcting it changes the historical subset is not established. The printed NMI equation uses geometric normalization, while the pinned scikit-learn call defaults to arithmetic normalization (F23). Historical score provenance needs verification before claiming which formula produced the paper tables.

**B. Scientifically questionable but non-bug design choices.** Independently pretrained networks and centroid sets have no explicit cluster-label matching; the column-wise contrastive objective can encourage matching, so permutation ambiguity is a plausible risk rather than a demonstrated bug (F09). Cluster loss applies a second softmax to hard one-hot assignments, changing occupancy statistics and cluster representations (F06). Sign inversion, segment reordering, potentially non-monotone time warping, amplitude-removing normalization, and the absence of a collapse guarantee merit targeted investigation, not automatic rejection (F12/F14/F18; architecture and preprocessing sections).

**C. Overstatements/incomplete descriptions.** The manuscript already describes hard forward/soft backward behavior, but omits the explicit straight-through construction and overstates freedom from surrogate gradients (F01). Sigma exists and is fixed to 1, but its value is omitted from the manuscript (F02). The final decoder block has no dropout or LayerNorm (F08). Permutation sampling, two separately precomputed augmentation sets, exponential annealing, M5 STL preprocessing, and percentage-point comparisons need more accurate documentation (F13/F15/F16/F20/F25). Store Sales and timing/convergence run provenance are incomplete (F21/F22). Dense scaling is linear in `L` at fixed width, but length-dependent and not intrinsically variable-length (F27).

**D. Correctly implemented behavior.** Three encoder and three decoder blocks, separate view weights, two active trainable centroid matrices, the RBF/Gumbel assignment construction, the supplied training hyperparameters, deterministic full-dataset stopping, original-view deterministic inference, RI, and oracle class-count use agree with the relevant manuscript descriptions (F17/F28-F31 and the detailed sections). ARI and run standard deviations already exist in the code.

The consistency table contains 32 findings/checks: **0 CRITICAL, 8 MAJOR, 12 MODERATE, 7 MINOR, 5 NONE**. Primary action counts: **2 CODE + RERUN, 16 PAPER ONLY, 2 EXPERIMENT NEEDED, 7 FURTHER INVESTIGATION, 5 NO ACTION**. Rerun recommendations are conditional where explained below; no reruns were performed. Severity measures consequence if the documented discrepancy/risk applies, not proof that published scores are wrong.

## Scope and Evidence

### Manuscript identification and evidence convention

`paper/original_r1/elsarticle-template-num-revised.tex` is the only full LoSTer manuscript in the supplied source tree: 861 lines, the LoSTer title at line 112, methods at 217-390, experiments at 392-826, and revised limitations at 835-840. Its SHA-256 is `6401d3c01eef52f6f99006327ffaecdcbbcb6bc0a769473418d714605e6e0d5a`. `paper/original_r1/elsarticle-template-num-names.tex` is a 284-line generic Elsevier template, with an empty title at line 81 and an example section at 147; it is not the scientific manuscript. Identification as the supplied latest/full R1 relies on the user's archive provenance; the actual submission version/date is **Not established from static inspection.** No separate R1 manuscript PDF is available locally.

Throughout this report, **M** denotes exactly `paper/original_r1/elsarticle-template-num-revised.tex`. `M:331`, for example, is a source-line reference to that file, not a rendered PDF line. Equation labels (`rbf`, `gumbel`, `instance_contrastive_loss`, `cluster_contrastive_loss`, `cluster_entropy`, `kmeans_o`, `kmeans_a`) are used where available; unlabeled equations are cited by source line. No LaTeX compilation was performed to infer rendered equation numbers. Notebook references use **zero-based cell indices and one-based source lines within the cell**.

**FACT** means a directly observed definition, argument, call site, saved notebook output, or algebraic consequence of those definitions. **INFERENCE** means a proposed interpretation, possible consequence, or design risk. Saved notebook outputs are historical evidence, not a rerun or proof of current raw data. Statements about the exact historical training environment and table provenance remain **Not established from static inspection.**

### Sources inspected

- All Python modules in `models/LoSTer/`, plus `run.sh`, `results.py`, and nested ignore rules; AST parsing and call-site inspection without importing the project or scientific libraries.
- `M5_EDA.py`, source cells and existing outputs of `M5_EDA_cat_store_id.ipynb`, and `plots_training.ipynb`; four existing `csv_logs/` files inspected for schema/row counts only.
- `README.md`, `requirements.txt`, repository file inventory, and relevant Git history. Raw UCR TSVs, M5 arrays, Store Sales data, trained checkpoints, and per-seed final metric CSVs are absent from this checkout. `datasets/` contains only its ignore file.
- Only official, version-relevant library documentation/source was consulted to establish semantics: [PyTorch 1.13.1 functional source](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/functional.py#L1699-L1751), [Linear initialization](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/modules/linear.py#L94-L102), [LayerNorm](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/modules/normalization.py), [scikit-learn 1.4 NMI](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.normalized_mutual_info_score.html), [scikit-learn 1.4 KMeans](https://scikit-learn.org/1.4/modules/generated/sklearn.cluster.KMeans.html), [pandas 1.5.3 read_csv](https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.read_csv.html), and [NumPy 1.22 interp](https://numpy.org/doc/1.22/reference/generated/numpy.interp.html). These checks are implementation semantics, not literature/novelty research. The scikit-learn documentation is for the 1.4 series (page build 1.4.2); the repository pins 1.4.1.post1.

`requirements.txt:23,25,39,40,48,56` pins NumPy 1.22.4, pandas 1.5.3, scikit-learn 1.4.1.post1, SciPy 1.13.0, torch 1.13.1+cu117, and wandb 0.21.0. These are repository declarations, not evidence of installed packages or the environments that produced historical results. No dependencies were installed, upgraded, or imported for execution.

## Actual Execution Pipeline

```text
models/LoSTer/run.sh:8-25,28-54
  -> models/LoSTer/main.py (__main__:19-152)
     -> utils.config.get_arguments + utils.random_seed.random_seed
     -> data.load_data.get_loader twice (main.py:62-76)
        -> Dataset_UCR OR Dataset_M5
        -> per-series scaling + rotation -> permutation -> time_warp
        -> CustomDataset + train/test DataLoader over the same original pool
     -> two independent net.model.AE instances (82-85)
     -> pretrain_ae + pretrain_ae_augmented (87-97; Adam, separately)
     -> predict + predict_augmented (100-108; eval mode; all rows)
     -> separate sklearn KMeans.fit calls -> fitted centers
     -> two net.model.LoSTer wrappers reload the pretrained AE weights
     -> two utils.losses.KMeansLoss objects own the ACTIVE centroids
     -> SGD + StepLR (111-117)
     -> experiment.train -> train_epoch (120)
        -> reconstruction + hard straight-through assignments
        -> reconstruction/k-means/instance/cluster losses -> backward -> step
        -> full deterministic original-view evaluation -> metrics/silhouette
        -> assignment-change stopping -> final model-only checkpoint
     -> per-seed final metrics CSV + text cluster counts (124-152)

Separate aggregation: models/LoSTer/results.py -> per-seed CSV reads -> mean/std.
Separate convergence plots: plots_training.ipynb -> csv_logs/* -> figure PDFs.
```

**Launch assumptions (FACT):** `run.sh` invokes `python main.py` without changing directory (line 52), and dataset/checkpoint/log paths are relative to the process working directory (`main.py:28-30`; `data/load_data.py:20,39`). Launch is therefore intended from `models/LoSTer/`, not repository root. `main.py:16,100,105` hardcodes CUDA, with no CPU fallback. The tracked `models/LoSTer/datasets` entry is a Git symlink to `../../datasets` (mode 120000). In this Windows checkout, `core.symlinks=false` materializes it as a plain text file. This is a local launch prerequisite, not a scientific alteration. No attempt was made to run or repair it.

**Dataset paths:** the supplied `run.sh` covers the 17 listed UCR datasets and `m5-forecasting-accuracy`; seeds are 0-4. `get_loader:75-80` treats only that exact M5 name specially. Every other name is routed to UCR. There is no Store Sales loader or supplied Store Sales run script. M5 preparation is in the root script/notebook, not invoked by `main.py`.

**Convergence paths:** `plots_training.ipynb`, cells 3-15 and 18-30, consumes the four tracked RI/NMI CSVs for NonInvasiveFetalECGThorax1 and UWaveGestureLibraryY. All four have 100 data rows. These are W&B-style exported run columns, renamed by numeric column positions; the notebook neither trains models nor computes across-seed uncertainty. `experiment.py:49-55,99-110` is the current W&B logging source. No dedicated full-100-epoch convergence training override is supplied. Git commit `efb74ff` added these plots/exports; `a322caa` changed LoSTer's run-script/parser joint learning rate from 0.001 to 0.01. A causal mapping from each historical export to a code commit/config/seed is **Not established from static inspection.**

**Unused/optional code:** `utils/gumbel.py:4-25` defines a manual Gumbel sampler and straight-through wrapper, but the active controller imports only `softmax_logits` and calls `torch.nn.functional.gumbel_softmax` (`experiment.py:3,9,33-34`). `TemporalResidualBlock` (`net/model.py:112-125`) is never instantiated by AE. RevIN and the AE-wide residual are optional, disabled by the supplied script/call sites. The additional augmentation functions, including DTW-dependent functions, are not in the active three-transform pipeline. `main.py:28`'s `path_data`, `main.py:39-41`'s augmented checkpoint path, and `results.py:6`'s `path_results` are defined but unused.

## Architecture

### Active block structure and shapes (FACT)

`models/LoSTer/net/model.py:77-93` implements

\[
R(x)=\operatorname{LayerNorm}\bigl(\operatorname{Dropout}(W_2\operatorname{ReLU}(W_1x+b_1)+b_2)+W_sx+b_s\bigr).
\]

This matches the order in M:239-251: Linear -> ReLU -> Linear -> Dropout, add a learned **linear** skip, then LayerNorm. The skip is not an identity mapping. There is no activation after LayerNorm. Encoder: first block `L -> d -> d`, then two `d -> d -> d` blocks (`net/model.py:142-149`). With `d=256`, input `[B,L,1]` is squeezed to `[B,L]`; latent output is `[B,d]` (`163-176`).

Decoder: two ordinary `d -> d -> d` blocks, then `PredictionResidualBlock`, which is

\[
P(z)=W_2\operatorname{ReLU}(W_1z+b_1)+b_2+W_sz+b_s,
\]

with `d -> d -> L`, **no Dropout and no LayerNorm** (`96-109,151-158`). Reconstruction is `[B,L,1]`. The last-block exception is not stated in the general decoder description M:259; F08 is a documentation discrepancy, not evidence the output block must be changed.

`main.py:82-85` instantiates two different AEs before pretraining; they do not share weights and are not initialized by copying one another. Their architectures match, but parameters are independent. M:266 already states that the two identical architectures have different weights; this is correctly described. The project's code supplies no custom initialization: Linear weights/biases use PyTorch's uniform fan-in-based initialization, and LayerNorm affine parameters initialize to ones/zeros. The seeded RNG is consumed sequentially by the two network constructions. These defaults are verified in the official [Linear](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/modules/linear.py#L94-L102) and [LayerNorm](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/modules/normalization.py) sources. Joint wrappers later reload each view's own pretrained weights (`net/model.py:195-199`), rather than retaining their freshly initialized AE weights.

**RevIN:** `--use_revin` is a store-true flag, absent from `run.sh`, so false (`utils/config.py:15`). If enabled, AE creates `RevIN(num_features=1, affine=True, subtract_last=False)` and normalizes/denormalizes around the network (`net/model.py:140,166-175`; `net/revin.py:35-62`). RevIN is not part of the supplied experiment path; its presence in the repository is not an omitted active method. Likewise all main call sites pass `use_residual=False`; the optional global `Linear(L,L)` residual at `net/model.py:160-172` is inactive.

### Structural scaling and scope (FACT/algebra)

For the supplied three encoder/three decoder blocks, one AE, excluding centroids and inactive options, has

\[
P_{AE}=4Ld+14d^2+26d+2L.
\]

Derivation: the first encoder block has `2Ld+d^2+5d` parameters; each of the four internal ordinary blocks has `3d^2+5d`; the final prediction block has `d^2+2Ld+d+2L`. Two views double this. Active centroids add `2kd`; dormant wrapper copies add another `2kd` stored parameters. This count is derived from constructors, not measured by constructing a model.

At fixed block count, AE multiply-add work per view/batch is `O(B(Ld+d^2))`; parameters scale as `O(Ld+d^2)`. The implemented distance broadcast also creates a `[B,k,d]` intermediate (`utils/gumbel.py:29`), instance similarities cost `O(B^2 d)`, and cluster-column comparisons `O(B k^2)` (`utils/losses.py:37-43,86-92`). These terms qualify whole-method scalability beyond the AE alone.

**INFERENCE:** linear dependence on `L` at fixed `d` is a reasonable structural advantage over an `L x L` attention map, but it is not length-independent parameterization or a demonstrated speed/memory advantage in arbitrary settings. M:195,831-833 should qualify “scalable” and “arbitrarily long”: learned first/last layer shapes are tied to a particular `L`. No ragged batching, length mask, padding policy, or pooling-based length invariance is provided.

The active pipeline is univariate: loader arrays are `[N,L]`, main adds a singleton channel, AE squeezes exactly dimension 2, and RevIN is fixed to one feature. Linear's ability to accept extra leading dimensions does not establish a multivariate temporal model. A `[B,L,C]` input with `C>1` is not converted to the intended `[B,L]` mapping; the present call sites/decoder do not implement multivariate reconstruction. M:228 defines scalar sequences and M:836 explicitly acknowledges the univariate scope, so this is not an undisclosed multivariate capability or a proven violation of that limitation.

Also, `d=256` is not a lower-dimensional bottleneck on ECG5000 (`L=140`), the three phalanx datasets (`L=80`), or SyntheticControl (`L=60`), per M:443-459. The “lower-dimensional” description in M:233 is therefore not universal; F27 includes this qualification.

## Concrete / Gumbel k-means

### Actual estimator

For a view with batch embeddings `Z in R^{B x d}` and active centers `C in R^{k x d}`, `models/LoSTer/utils/gumbel.py:27-31` computes

\[
D_{ij}=\|z_i-c_j\|_2^2,\quad a_{ij}=-D_{ij}/\sigma^2,\quad
\ell_{ij}=\log p_{ij}=a_{ij}-\log\sum_{r=1}^k e^{a_{ir}}.
\]

Sigma **exists literally**, with default `sigma=1.0`. Every active call omits it (`experiment.py:31-32,66`), so it is fixed to 1. M's `rbf` equation at 295-300 agrees structurally; the missing experimental value is F02. Sigma is the distance/RBF bandwidth; it is **not** the annealed Gumbel temperature. Although both influence relaxed probabilities, replacing bandwidth by tau also changes the relative noise/logit scale and is not an equivalent training estimator.

The active PyTorch call has `hard=True` from the first batch onward. With independent `g_{ij} ~ Gumbel(0,1)`, the returned differentiable tensor is

\[
y_{ij}=\frac{\exp((\ell_{ij}+g_{ij})/\tau_e)}{\sum_r\exp((\ell_{ir}+g_{ir})/\tau_e)},\quad
h_{ij}=\mathbf1[j=\arg\max_r y_{ir}],\quad
Q^{ST}=\operatorname{stopgrad}(H-Y)+Y.
\]

**Forward:** `Q^{ST}=H` is one-hot. **Backward:** its assignment Jacobian is that of `Y`, namely

\[
\frac{\partial Q^{ST}_{ij}}{\partial\ell_{ir}}
=\frac{1}{\tau_e}y_{ij}(\mathbf1[j=r]-y_{ir}).
\]

Downstream loss gradients are evaluated using the hard forward tensor, then multiplied by this relaxed Jacobian; embeddings/centers also receive direct gradients through their other occurrences. This is the standard straight-through Gumbel-Softmax construction, not differentiation through ordinary argmax. The built-in implementation explicitly uses the detached-soft correction ([PyTorch 1.13.1 source](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/functional.py#L1737-L1749)). The project's unused manual wrapper at `utils/gumbel.py:25` is equivalent in this respect but is not the executed function.

**Algebraic consequence:** for fixed logits and positive tau, scaling all perturbed logits by tau does not change argmax. The hard sample's categorical law is `p`, independent of tau. Annealing changes the gradient relaxation; it does not switch the forward pass from soft to hard, nor make hard samples deterministic as tau tends to zero. Both views draw their own stochastic assignments in each training batch (`experiment.py:33-34`).

### Manuscript comparison

M:293 explicitly says stochastic hard forward assignment with gradients through soft assignment. Thus H1's strongest premise is false: the mechanism is already described verbally. M:302-307 gives the relaxed equation and rounding language, but neither names straight-through nor specifies detach/stop-gradient, and suggests discretization as tau decreases, whereas the code is always hard forward. Add the estimator formula and clarify that tau regulates the backward relaxation.

M:283,198 and 833 claim direct hard k-means optimization without a soft relaxation/surrogate. The **forward residual** really does use one-hot assignments, but the gradient through assignments is a relaxed straight-through surrogate; unbiasedness or optimization of the exact discrete expected objective is **Not established from static inspection.** This is a manuscript overstatement, not proof the implementation should be replaced. Inference is a separate deterministic rule, described later.

## Centroids and Two-View Alignment

**FACT:** there are **four stored `k x d` Parameter objects, of which two are active**:

| Location | Initialization | Used for assignment/loss? | Gradient role |
| --- | --- | --- | --- |
| `model.centroids` | Original-view fitted centers | No; `LoSTer.forward` returns only AE outputs | Dormant; no loss path |
| `model_augmented.centroids` | Augmented-view fitted centers | No | Dormant; no loss path |
| `criterion_kmeans.centroids` | Original-view fitted centers | Yes, original logits and residual | Trainable, in SGD |
| `criterion_kmeans_augmented.centroids` | Augmented-view fitted centers | Yes, augmented logits and residual | Trainable, in SGD |

Evidence: `models/LoSTer/net/model.py:179-205`; `utils/losses.py:6-16`; `main.py:101-116`; `experiment.py:31-40,66`. Wrappers and criteria construct separate tensors/Parameters from the center arrays; they are not tied. The two dormant copies are in the optimizer's model-parameter lists, but inclusion in an optimizer does not create a gradient when the loss never uses them.

Each active matrix is initialized after its own 50-epoch AE pretraining. `main.py:100-108` uses all ordered original rows and separately all ordered augmented rows with two `KMeans(init='k-means++', random_state=args.seed).fit(...)` calls. This is a full KMeans fit initialized with k-means++, not merely sampling seed points and stopping. `n_init` is not passed; for the pinned 1.4 API, `n_init='auto'` uses one initialization with k-means++ ([KMeans documentation](https://scikit-learn.org/1.4/modules/generated/sklearn.cluster.KMeans.html)). Historical runtime versions or alternative n_init values are not established.

The source ordering and RNG seed are the same for the two fits, which can correlate their initialization. The representations and fitted distance geometry differ; those shared inputs **do not guarantee** equal cluster identities or a common final column permutation. No matching/Hungarian step, shared-center object, or copied network weights exists. Whether this partial coordination effectively aligned labels in the historical runs is **Not established from static inspection.**

`ClusterContrastiveLoss` uses target indices `arange(k)` (`utils/losses.py:82-95`), treating column j from the original view and column j from the augmented view as positives. The objective itself can establish a correspondence during joint training; instance contrast also couples the latent spaces. Successful alignment, harmful local minima, and empirical permutation sensitivity are **Not established from static inspection.** Classification: **PLAUSIBLE RISK**, not a confirmed bug (F09). A permutation applied to one view alone changes that objective; RI/NMI themselves are invariant to cluster-label names and do not require cross-view matching for final original-view evaluation.

**Checkpoint defect (F10):** `experiment.py:135` saves only `model.state_dict()`. This includes the original AE and its unused initial centroid copy, not `criterion_kmeans.centroids`, the augmented network/active centers, or optimizer/scheduler state. `main.py:39-41` defines an augmented checkpoint path but never saves to it. Recovering the trained clustering rule from the written checkpoint alone is therefore not possible in general. Paths `model.pth`, `autoencoder.pth`, and `autoencoder_augmented.pth` omit the seed (`main.py:28-49`), so the script's sequential runs overwrite those checkpoints; metric CSV names do include the seed. This does not prove on-the-fly reported scores were wrong, but it is a confirmed model-artifact reproducibility defect.

## Loss Functions

Let `B` be the actual minibatch size, `L` sequence length, `d` latent dimension, and `k` cluster count. Superscripts `o,a` indicate views. The controller (`models/LoSTer/experiment.py:36-40`) minimizes

\[
L_{code}=R^o+R^a+\alpha\frac{K^o+K^a}{2}+I^o+I^a+C^o+C^a+E,
\]

with `alpha=1` in the supplied script. Formula details:

| Term / function | Implemented formula and reduction | Paper reference | View contribution / coefficient |
| --- | --- | --- | --- |
| Reconstruction, `nn.MSELoss()` | `R^v=(BL)^{-1} sum_i ||x_i^v-xhat_i^v||^2` | M:268-280 | Sum of two view means; coefficient 1 |
| `KMeansLoss.forward` | `K^v=(Bd)^{-1} sum_i ||z_i^v-Q_i^{ST,v} C^v||^2` | `kmeans_o/a`, M:309-320 | Half the sum of view means, multiplied by alpha |
| `InstanceContrastiveLoss.forward` | Two B-anchor cross-entropy means on normalized embeddings | `instance_contrastive_loss`, M:330-337 | Directional means added; temperature 1 |
| `ClusterContrastiveLoss.forward` | Two k-anchor cross-entropy means on columns of `S^v=softmax(Q^{ST,v},dim=1)` | `cluster_contrastive_loss`, M:342-355 | Directional means added; temperature 1 |
| Entropy inside cluster loss | `E=sum_j p_j^o log p_j^o + sum_j p_j^a log p_j^a`, `p_j^v=B^{-1}sum_i S_ij^v` | `cluster_entropy`, M:348-355 | Both views, coefficient 1; negative entropy |

Code evidence: `models/LoSTer/main.py:111-117`; `utils/losses.py:15,26-48,60-97`; `experiment.py:36-40`. Pretraining uses only one view's reconstruction MSE at a time (`net/model.py:8-48`). No additional executed loss was found. Dropout is active regularization, not a separate objective term; no weight decay or momentum is requested in the SGD call.

### Correct implemented contrastive denominators versus printed equations

For unit-normalized embeddings, the original-view per-anchor instance loss is

\[
i_i^o=-\log\frac{e^{sim(z_i^o,z_i^a)/\tau_I}}{
\sum_{j=1}^{B}e^{sim(z_i^o,z_j^a)/\tau_I}
+\sum_{j\ne i}e^{sim(z_i^o,z_j^o)/\tau_I}},
\quad I^o=B^{-1}\sum_i i_i^o.
\]

The reverse direction is analogous. Cross-view positives are **included** in the denominator; same-view self-similarity is suppressed by subtracting `1e9` from diagonal logits (`utils/losses.py:37-46`). M:331 instead places the exclusion indicator on the cross-view term while leaving the same-view self-term. This does not implement the displayed formula. The same reversal occurs in the cluster equation M:343 versus `utils/losses.py:86-95` (F03). Correct the paper masks to match the executed contrastive objectives; changing the code to the displayed masks would change the method and require reruns, not a harmless notation fix.

For cluster columns `s_j^v in R^B`, the analogous original anchor is

\[
c_j^o=-\log\frac{e^{cos(s_j^o,s_j^a)/\tau_C}}{
\sum_{r=1}^{k}e^{cos(s_j^o,s_r^a)/\tau_C}
+\sum_{r\ne j}e^{cos(s_j^o,s_r^o)/\tau_C}},
\quad C^o=k^{-1}\sum_j c_j^o.
\]

Tensor proof of the n/k error: inputs have shape `[B,k]`; transposition at `losses.py:79-80` yields `[k,B]`, `n_clusters=q_.size(0)` at 82 is k, target length is k, and cross-entropy logits have shape `[k,2k]`. The denominator in M:343 must sum over k cluster columns, not n sample rows (F04). In minibatch formulas, instance n should be defined as B; cluster k remains the same dataset class count.

### Extra softmax, entropy, and implicit weighting

The cluster loss receives the **hard-forward** tensor produced by Gumbel-Softmax, not RBF probabilities or the relaxed Y (`experiment.py:39`). It then computes `S=softmax(Q^{ST})` (`losses.py:64-65`). If H is the one-hot forward matrix,

\[
S_{ij}=b+(a-b)H_{ij},\qquad
b=\frac{1}{e+k-1},\quad a=\frac{e}{e+k-1}.
\]

Consequently the entropy marginal is `p_j=b+(a-b) f_j`, with `f_j=B^{-1}sum_i H_ij`. It is neither the hard occupancy f nor the RBF marginal. Even an empty hard-assigned cluster has positive soft mass b. The selected/unselected mass ratio is always e, independent of Gumbel tau. Gradients also include the Jacobian of this extra softmax at the hard forward values. This materially changes column similarities and entropy gradients (F06), but the code alone cannot determine whether it was deliberate smoothing or an unintended implementation choice. Documenting the implemented S could be manuscript-only; replacing/removing S changes training and requires reruns.

The entropy implementation has the correct **negative-entropy** sign for minimization of `contrastive - H`: encouraging larger entropy, not collapse. M:349 lacks clear brackets placing both view terms inside the summation and common leading minus, leaving the augmented term/sign/scope ambiguous. Write `H=-sum_j[p_j^o log p_j^o+p_j^a log p_j^a]`; do not claim the code has a sign bug (F07).

The printed reconstruction/k-means means in M:269/273 and 310/315 divide by n but sum squared vector norms. Code divides additionally by L or d. With the corrected masks and B=n, code instance directional summation is **twice** M:337's `1/(2n)` term. Cluster directional summation is likewise twice M:355's `1/(2k)` term, while the entropy coefficient is unchanged. These factors are not one global scaling that leaves the complete objective equivalent (F05). There is also an explicit configurable alpha for k-means, omitted from M:387, although the script passes 1 (`run.sh:32,52`; `utils/config.py:16`).

**Mathematical scale, not empirical measurement:** reconstruction and k-means are means per scalar coordinate, scaling like mean squared residual size rather than summed L/d-coordinate error. With equal similarities, the two-direction instance term is `2 log(2B-1)` and the cluster contrastive part is `2 log(2k-1)`; these are algebraic reference cases, not observations from training. Entropy lies in `[-2 log k,0]`, and is near its uniform offset when the marginals are near uniform. Combining these heterogeneous reductions with nominal unit coefficients does not mean equal influence. Whether any term or gradient actually dominates historical training is **Not established from static inspection.** H5 is only partly confirmed; empirical imbalance was not measured.


## Augmentations

**Active pipeline (FACT):** for every series, both `Dataset_UCR` and `Dataset_M5` construct

```text
scaled X [N,L] -> expand to [N,L,1]
 -> rotation -> permutation -> time_warp
 -> squeeze -> stored X_a [N,L]
```

Evidence: `models/LoSTer/data/load_data.py:25-27,44-46`; `data/augmentation.py:13-37,52-65`. These three transforms are always invoked; there is no probability gate or random choice of one transform. `CustomDataset.__getitem__` only reads stored tensors (`load_data.py:67-71`); no augmentation is generated online per batch/epoch. The construction is stochastic once per loader creation, seeded through NumPy, then fixed during that loader's lifetime. Shuffling/dropout/Gumbel sampling remain stochastic during training independently of these fixed augmented arrays.

| Transform | Actual operation / probability / parameters | Manuscript comparison |
| --- | --- | --- |
| Rotation, `augmentation.py:13-17` | Independent sign from {-1,+1}, probability 1/2 each, per series/channel; shuffle channel indices once. For C=1 that shuffle is the identity, so this is sign inversion only. | M:230 correctly describes random sign flips; no physical multivariate rotation is implemented. |
| Permutation, `19-37` | Segment count uniform on {1,2,3,4}, because `randint(1,5)` excludes 5. `seg_mode='equal'` uses `array_split`; segments preserve internal order but their block order is randomly permuted. One segment (probability 1/4) is unchanged; for s segments the identity block permutation is also possible. | M:230 instead says a positive Gaussian with std 5 is rounded to obtain the split count. This is a confirmed parameter/distribution mismatch (F13). |
| Time warp, `52-65` | Six evenly spaced knot times (`knot=4` means knot+2); independent multipliers from Normal(1,0.2), no clipping of these draws. CubicSpline fits knot time times multiplier, evaluates at integer times, rescales to end at L-1, clips coordinates to [0,L-1], then `np.interp` resamples the signal on the original grid. Output length is unchanged. | M:230 matches the six controls, Gaussian mean/std, and cubic interpolation, but omits endpoint rescaling, clipping, and final linear resampling. |

Permutation disrupts cross-segment chronological order and creates new boundaries; it does not shuffle individual time points within a segment. “Rotation” in this scalar pipeline is a mathematically well-defined sign inversion, not spatial rotation. Whether sign inversion and arbitrary segment order preserve the relevant class semantics—especially for nonnegative retail sales—is **Not established from static inspection.** This is a scientific augmentation assumption, not an implementation bug solely because of the function name.

**Time-warp risk (F14, INFERENCE backed by the formula):** Gaussian knot multipliers and cubic interpolation do not enforce monotone warped coordinates. Even positive multipliers can reorder knot coordinates; a spline can additionally overshoot. Rescaling and clipping do not establish strict monotonicity. The code does not check it. `np.interp` requires increasing sample coordinates; the [NumPy 1.22 documentation](https://numpy.org/doc/1.22/reference/generated/numpy.interp.html) does not guarantee meaningful interpolation for non-increasing coordinates. Occurrence/frequency and effect on the actual historical augmentations are **Not established from static inspection.** No sampling experiment was run.

**Two loader snapshots (F15, FACT):** `main.py:62-76` calls `get_loader` twice, and each call reconstructs its dataset and draws a fresh augmentation. The original data pool/order is the same, but training has one fixed augmented array and the ordered evaluation/centroid-initialization loader has a second. The augmented AE is pretrained on the first; its KMeans fit uses the second (`main.py:97,105-107`). Final metrics use only original rows and ignore evaluation augmentations. M:230/364 correctly says augmentation is precomputed, but does not specify this two-snapshot implementation. It should not be described as epoch-wise augmentation or a single globally reused augmented dataset.

`jitter`, `scaling`, `magnitude_warp`, `window_slice`, `window_warp`, `spawner`, `wdba`, and guided-warp functions elsewhere in `augmentation.py` are defined but not called by this pipeline. Their optional DTW functionality in `data/utils/dtw.py` is not an executed dependency of the supplied three-transform path.

## Training Procedure

The table distinguishes arguments actually supplied by `models/LoSTer/run.sh` from parser/library defaults and hardcoded controller choices. `run.sh:28-45,52` passes the same settings to all 17 UCR datasets and M5; no per-dataset override is present.

| Item | Actual supplied/executed value | Exact evidence | Manuscript |
| --- | --- | --- | --- |
| Latent/hidden width | 256 | `run.sh:28`; `net/model.py:142-158` | M:405,427 agrees |
| Encoder / decoder blocks | 3 / 3, last decoder is prediction block | `run.sh:29-30`; `net/model.py:142-158` | M:406-407 agrees in count, exception omitted |
| Dropout | 0.1 in ordinary residual blocks only | `run.sh:31`; `net/model.py:81-85,96-109` | M:408 agrees; qualify final block |
| AE pretraining | 50 epochs per view, sequentially, reconstruction only | `run.sh:38`; `main.py:87-97`; `net/model.py:8-48` | M:414,427 agrees |
| Pretraining optimizer / LR | Separate Adam optimizers, LR 0.001 | `main.py:91,96` (hardcoded, not args.lr) | M:416,427 agrees |
| Pretraining early stopping | None; loops over exactly epochs_pretrain | `net/model.py:10,32` | Algorithm at M:366-371 says while not converged; refine pseudocode (F32) |
| Load instead of pretrain | CLI epochs_pretrain=0 loads each AE checkpoint | `main.py:88-97` | Optional code branch, not run.sh's experiment |
| Centroid initialization | After both pretrainings; each KMeans fits all ordered view embeddings, k-means++ seeded | `main.py:99-108` | M:372,427 agrees broadly; full fit versus seed selection should be clear |
| Joint optimizer / LR | One SGD optimizer over both networks and loss-owned centers, initial LR 0.01 | `main.py:116`; `run.sh:40` | M:417,427 agrees |
| Joint LR decay | StepLR, factor 0.1 every 5 completed training epochs | `main.py:117`; `experiment.py:57`; `run.sh:41` | M:418,427 agrees |
| Maximum joint epochs | 100, unless assignment-change criterion fires | `run.sh:39`; `experiment.py:122-132` | M:415,427 agrees |
| Gumbel temperature | tau_e=max(10*0.65^e,0.01), e zero-based | `run.sh:33-34`; `experiment.py:21` | M:427 agrees as a closed-form schedule; M:390 incorrectly calls it linear |
| Minimum tau | 0.01, hardcoded; floor first applies at e=17 (18th epoch), if training reaches it | `experiment.py:21` | M:427 agrees |
| Instance / cluster temperatures | 1 / 1; default function arguments, never overridden | `utils/losses.py:26,60`; `experiment.py:38-39` | M:411-412 agrees |
| K-means coefficient alpha | 1.0 supplied; CLI can change it | `run.sh:32,52`; `utils/config.py:16`; `experiment.py:40` | M:387 omits configurable coefficient; unit value is consistent |
| Batch size | 128 | `run.sh:37,52`; `main.py:62-76` | M:413,429 agrees for main run script |
| Training data loader | shuffle=True, drop_last=True, 4 workers | `main.py:62-68`; `run.sh:43` | Minibatch/drop-last detail not specified |
| Evaluation loader | shuffle=False, drop_last=False; same original sample pool | `main.py:70-76` | Full-data clustering, not a held-out test protocol |
| RevIN / global AE residual | False / False | `config.py:15`; absent flag in run.sh; `main.py:82-85,103,108` | Neither is an active undocumented module |
| Seeds / repetitions | 0,1,2,3,4; five script processes per dataset | `run.sh:45,48-54`; `utils/random_seed.py:5-11` | M:393 agrees on five runs; seed values omitted |

`utils/random_seed.py:6-11` seeds Python, NumPy, Torch, and CUDA and sets deterministic cuDNN/disabled benchmark. This establishes an attempt at reproducibility, not a static proof of identical GPU results across platforms/library versions. The script does not record a raw-data checksum or exact code commit in its filenames. No undocumented dataset-specific hyperparameter override was found for UCR/M5. Store Sales cannot be audited to this level because its launch configuration is absent. M:824's Store Sales timing uses batch size 32, distinct from the supplied 128 script; this can be legitimate separate benchmarking, but its implementation/configuration is not supplied (F22).

The exponential schedule is recalculated from immutable `args.temperature`; it is not a recursive multiply of the previous epoch's already-decayed temperature. Distinguish the k-means weight `args.alpha` from the paper's learning-rate symbols alpha_1/alpha_2. The main script's hardcoded Adam LR is unaffected by changing the joint `--lr`.

## Stopping and Degenerate Solutions

### Exact stopping rule (FACT, matches the paper)

`models/LoSTer/experiment.py:59-74` switches the original model to eval mode and, under no_grad, predicts every row from the ordered, non-dropping evaluation loader. Assignment is `argmax(softmax_logits(z, criterion_kmeans.centroids.detach()))`, with dropout disabled and **no Gumbel noise**. `train:127-133` computes

\[
\delta_e=\frac{1}{N}\sum_{i=1}^{N}\mathbf1[\hat c_i^{(e)}\ne\hat c_i^{(e-1)}]
\]

once after every epoch and stops if `delta_e < 0.001`. This is full-pool, not minibatch, and measures deterministic original-view label changes. Epoch 0 only records predictions; comparison begins at epoch 1, so the initial empty list is not incorrectly compared. Augmented-view assignments, loss change, and gradient/parameter convergence are not used. M:427's less-than-0.1% description agrees (F17). Stable assignments alone do not prove convergence of every parameter or achievement of a good clustering solution.

`best_ri_score` at `experiment.py:118,124-126` is printed/tracked but does not select a checkpoint or determine stopping. Returned predictions/metrics are from the **last** executed epoch (`140`), not the epoch with maximum label-based RI.

### Empty/collapsed clusters

One-hot assignments can omit any cluster in a batch or across the evaluated pool. Neither `train_epoch` nor `train` reseeds empty centers, enforces minimum cluster size, or rejects a collapsed solution. The final count loop (`main.py:141-150`) reports empty/singleton counts; it does not repair them. Silhouette errors are caught and mapped to `-inf` (`experiment.py:75-78`), allowing training/evaluation to continue.

For a hard-assigned empty center, the **direct** k-means residual derivative with H fixed is zero because its assignment column is zero. Nevertheless, that center still enters the logits, so the straight-through assignment path and contrastive paths can supply indirect gradients; it is not correct to declare it permanently frozen. A zero derivative in the unused wrapper centers is a different issue. Whether a particular active empty center actually receives a nonzero update is **Not established from static inspection.**

The negative entropy regularizer encourages spread but operates on smoothed marginals S, where every column is positive even for a hard-empty cluster. There is no hard prevention of emptiness or one-cluster collapse. Classification: **PLAUSIBLE RISK** (F18). M:346 says entropy is incorporated to avoid degeneration; phrase this as a regularizing aim, not a demonstrated guarantee. No actual collapse in the historical results is established here.

## Dataset and Preprocessing

### UCR

**FACT:** `models/LoSTer/data/load_data.py:20-34` reads TRAIN and TEST TSVs and concatenates TRAIN before TEST. Both main loaders reconstruct this same pool. The labels are removed from X, individually standardized with `(x-mean(x))/std(x)`, encoded to contiguous zero-based integers, and counted to set k (`main.py:78`). No per-dataset/per-time-step scaler is fitted, no separate train-only scaling statistics exist, and no held-out generalization assessment is performed. This agrees with M:468's transductive clustering protocol.

`np.std` is the population-standard-deviation operation with its default ddof, applied over each individual 1-D sequence (`load_data.py:10-11,25-26`). Positive affine rescaling `x -> a*x+b` leaves the standardized sequence unchanged for a>0 and nonzero variance. Absolute level and amplitude scale are therefore removed. That can be appropriate shape-based preprocessing, as already described in M:468; harm to a particular task is not established (H8).

**Header risk (F11):** the TSV calls at lines 21-22 omit `header=None`. With no column names supplied, pandas consumes the first nonblank record as a header ([pandas 1.5.3 behavior](https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.read_csv.html)). For headerless inputs this drops one sample from each split, yielding `N_train+N_test-2`, and can also affect an oracle class count if an omitted record is unique. No local TSVs exist to check historical file headers or actual row counts. Header inference is confirmed; whether this affected the paper's actual data is **Not established from static inspection.** Check the original files before changing code or concluding the published sample counts were wrong.

**Missing values/constant series (F12):** there is no NaN imputation, finite check, or zero-variance guard. A constant series divides by zero; an ordinary missing value propagates through mean/std, augmentation, and potentially KMeans. Actual nonfinite/constant records in these benchmark runs are **Not established from static inspection.** Training batches omit labels in their loss path (`experiment.py:24`); label-derived k is explicit benchmark side information. Evaluation/RI tracking uses labels after each epoch but does not supervise the update, select the best model, or control stopping (F30).

### M5

**Preparation path and saved evidence (FACT):**

1. `M5_EDA.py:13-39` reads sales_train_validation, calendar (first 1913 days), and sell prices, merges daily prices, and processes a hardcoded 30490 series. The department filters at 15-16 are commented out. Its active save at 51 produces `sell_prices_individual.npy`; saves of raw sales/cat_id at 50/52 are commented out. This is an upstream notebook input, not the model's final sales array.
2. `M5_EDA_cat_store_id.ipynb` cell 3 reads the sales CSV; cells 5-7 merge the per-item prices. Cells 10/12 group by item_id, cat_id, store_id and aggregate sales by sum/prices by mean; missing **prices** are filled with zero in cell 12, line 1. Cell 14 merges the resulting tables. Saved outputs show 30490 rows; these are historical notebook observations, not validation of absent raw files.
3. Cell 16, lines 1-6 filters by zero_count<400; cell 17 applies the flags. Saved outputs report 1150 rows (858 FOODS, 256 HOUSEHOLD, 36 HOBBIES). There is no random subsampling at this step.
4. Cell 18, lines 1-3 constructs labels `cat_id + '_' + store_id`. Its saved value-count output has 29 categories. Cell 20, lines 1-7 performs `STL(data, period=7).fit()` and stores **seasonal+trend**, excluding the remainder; lines 16/23-36 hardcode 1150 iterations and export sales/labels. Saved output shows `(1150,1913)` sales and 1150 labels. This supports the manuscript's subset shape, but the final arrays/checksums are unavailable.
5. `models/LoSTer/data/load_data.py:39-50` loads that `sales.npy` and `cat_store_id.npy`, converts sales to float32, derives k from encoded labels, and **min-max normalizes each series separately** before the same augmentation pipeline. It does not load prices or calendar features. Additional notebook exports for prices/calendar are not model inputs.

M:789 says 1150 series, length 1913, and 29 category/store clusters; the saved notebook outputs are consistent. It does not explain the numerical zero threshold, STL remainder removal, or per-series min-max scaling (F20). The model therefore clusters processed univariate sales profiles; it does not use multivariate price/calendar inputs merely because preprocessing exports them. The exact subset consumed by each historical run, checksums, finite values, and k observed at runtime are **Not established from static inspection.**

**Confirmed filter-column discrepancy (F19):** before creating cat_store_id, the saved merged column order is `item_id, cat_id, store_id, d_1,...,d_1913,s_0,...`. Cell 16's `sales_items.values[:,2:2+1913]` selects `store_id` plus d_1 through d_1912, not the 1913 sales days. The last day is omitted from zero counting. This is a confirmed indexing defect for a full-series zero-day filter; interpreting all-sales-day counting as the intent is supported by the cell heading and sequence length, not an explicit threshold formula in M. It changes membership only for rows on the boundary: for example, 400 zeros across all days with d_1913=0 would be counted as 399 and retained. Whether such rows occur, whether the count/29 classes change, and whether rerunning affects results are **Not established from static inspection.** Verify before triggering a scientific rerun; do not silently preserve a hardcoded 1150 loop after changing the subset.

No explicit sales NaN handling or zero-range guard exists (`load_data.py:14-15,44-45`). Min-max normalization also removes positive affine level/amplitude changes for nonconstant series. The deterministic filter and STL are reconstructable from notebook source, but raw inputs, environment, object-array behavior, price joins, and hardcoded counts need a later reproducibility audit; saved outputs alone are not a complete provenance chain.

### Store Sales

M:791 specifies 1729 series of length 1684 with 33 product-family clusters; M:824 reports timings with batch size 32. No Store Sales preparation script, explicit dataset route, data, supplied launch configuration, or per-seed results was found (F21). A user could potentially convert external data into the generic UCR-style TSV route, but no such conversion is documented/provided. Filtering, sample selection, normalization, missing values, ground-truth construction, actual k, and seed aggregation for those reported runs are **Not established from static inspection.** The absent path does not establish that the reported experiment never happened.

## Inference and Metrics

### Assignment and evaluation

The active final assignment is original-view-only:

\[
\hat c_i=\arg\max_j\ell_{ij}=\arg\min_j\|z_i-c_j\|^2,
\]

using the trained **loss-owned** centers, eval mode, no_grad, and no Gumbel samples (`models/LoSTer/experiment.py:60-74`). Positive fixed sigma does not change argmax; tau is not used. The augmented network is not used for final prediction. This differs deliberately from stochastic hard training and matches Algorithm 1's final argmax (M:380; F31). Returned predictions are computed at the final executed epoch; saved checkpoint limitations are separate from the validity of this in-memory assignment path.

### RI, NMI, ARI, ACC

`models/LoSTer/utils/metrics.py:6-15` counts within-cluster and within-class pairs, joint true-positive pairs, and derives FP/FN/TN. Its returned `(TP+TN)/choose(N,2)` matches M:504-505. `evaluation:19` passes label/prediction in reversed named roles, but RI is symmetric, so this is not a correctness bug. Zero-based integer labels satisfy the bincount assumptions in this pipeline.

NMI is `sklearn.metrics.normalized_mutual_info_score(label,prediction)` with no `average_method` override (`metrics.py:21`). For the pinned 1.4 API the default is arithmetic, yielding `2*MI/(H_true+H_pred)`; M:508-511 instead uses `MI/sqrt(H_true*H_pred)`. These generally differ and agree when entropies are equal (F23). The default changed from geometric to arithmetic before this pinned version, as documented in the [official API](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.normalized_mutual_info_score.html). Confirm the historical runtime/call and table provenance. If arithmetic is the recorded metric, correcting the paper equation is sufficient; if geometric is intended, recompute from predictions, with new training only if those predictions cannot be recovered. This is not automatically a need to retrain every model.

ARI is **already active**, returned by `evaluation:20`, logged every epoch (`experiment.py:95,107`), written per seed (`main.py:131`), and aggregated by `results.py:45-46,60,68,75-76`. Silhouette is also computed/logged/written, but not aggregated by that results script. ACC/Hungarian-matched clustering accuracy was not found in repository Python source by targeted search; it is not an existing unused implementation. RI/NMI/ARI are invariant to cluster-label renaming; cross-view contrastive column matching is a separate training issue.

### Aggregation and provenance

`models/LoSTer/results.py:15-36` lists the 17 UCR datasets and seeds 0-4, not M5/Store Sales. It reads one final row per seed (`55-66`), computes means and `np.std` with default ddof=0 (`67-70`), stores mean/std columns, and prints the DataFrame and macro average over datasets (`72-88`). `path_results='./results.csv'` is unused: the script does not write an aggregate CSV despite its name. The paper's tables show means without these standard deviations (F24). Saved per-seed metric CSVs are absent, so reproducing those reported means/stds from source alone is not possible here. No script-level evidence of retail aggregation was found.

Convergence CSVs each contain 100 rows, but the column mapped to LoSTer by `plots_training.ipynb` has only **11 nonempty entries (epochs 1-11)** in all four exports: source column index 22, `worthy-yogurt-341` for NonInvasiveFetalECGThorax1 and `rare-tree-337` for UWaveGestureLibraryY. This is consistent with early stopping; 100 comparison-table rows do not mean LoSTer itself trained 100 epochs. The notebook selects single named run columns by position and plots them, rather than calculating a five-run average/error band. M:746/762's convergence discussion needs run/seed provenance; an RI/NMI plateau alone does not prove assignment stability or objective convergence (F22). No objective-magnitude estimates or new experimental statistics were computed.

### Percentage differences and a table/text mismatch

**FACT from the displayed numbers:** M:545-546/591 reports 0.8259 versus 0.7494 as +7.65%. The arithmetic is `(0.8259-0.7494)*100 = 7.65` **percentage points on the [0,1] scale**, not relative percent improvement. Relative improvement divides by 0.7494 (about 10.21%). M:581-582/591's 0.5181 versus 0.3231 similarly gives 19.50 percentage points, versus about 60.35% relative improvement. The difference rows in the Transformer and ablation tables use the same point-difference convention (M:619-620,650-651,697-698,733-734). Relabel units or explicitly define absolute point difference; do not alter the existing scores or interpret these as relative gains (F25).

M:664 says iTransformer surpasses LoSTer on StarLightCurves and **EOGVerticalSignal**. The displayed tables instead show StarLightCurves and **ECG5000** (M:609,615,640,646). That is a manuscript-only dataset-name error (F26), independent of score provenance or claims about competing architectures.


## Paper-Code Consistency Findings

Primary action labels describe how to resolve each finding **while preserving the current implementation as the historical reference**, unless an actual code defect is identified. `PAPER ONLY` means a faithful description/notation correction suffices; deciding instead to implement the presently printed equations can require code changes and reruns. `FURTHER INVESTIGATION` means the missing evidence or intended specification must be resolved first. `EXPERIMENT NEEDED` identifies an empirical design question, not authorization to experiment now. Conditional rerun requirements are detailed after the table.

| ID | Severity | Topic | Manuscript statement/equation | Implementation evidence | Verdict | Scientific consequence | Required action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | MODERATE | Straight-through estimator and exact-objective claim | M:283,293,302-307 (`gumbel`),198,833; hard optimization without soft relaxation | `models/LoSTer/experiment.py:31-34`; `models/LoSTer/utils/gumbel.py:25,27-31`; active PyTorch hard call | FACT: hard forward with relaxed backward; mechanism already described verbally, formula/name omitted | One-hot forward is real; exact discrete-gradient optimization and absence of surrogate gradients are overstated | PAPER ONLY |
| F02 | MINOR | RBF bandwidth | M:295-300 (`rbf`); sigma appears but no value/definition in hyperparameters at 395-425 | `models/LoSTer/utils/gumbel.py:27-30`; `models/LoSTer/experiment.py:31-32,66` | FACT: sigma exists, fixed default 1 at all active calls | Missing experimental scale impedes faithful reproduction; not an absent-code parameter | PAPER ONLY |
| F03 | MAJOR | Wrong contrastive mask placement | M:331 (`instance_contrastive_loss`),343 (`cluster_contrastive_loss`) exclude cross-view positive and retain same-view self | `models/LoSTer/utils/losses.py:37-46,86-95` masks same-view diagonal and includes all cross-view pairs | CONFIRMED manuscript/code mathematical mismatch; implemented positive-pair denominator is coherent | Reimplementing the printed equations gives a different optimization objective | PAPER ONLY |
| F04 | MODERATE | Cluster n/k indexing | M:343 sums cluster-column denominator to n, while 340 identifies columns as clusters | `models/LoSTer/utils/losses.py:79-95`: k by B transposes, k targets, k by 2k logits | CONFIRMED index error; bound must be k | Printed cluster formula is dimensionally wrong for n unequal to k | PAPER ONLY |
| F05 | MAJOR | Loss reductions and implicit relative weights | M:269/273,310/315,337,355,387 use sample-mean vector errors and half directional averages | `models/LoSTer/main.py:111`; `models/LoSTer/utils/losses.py:15,45-48,94-97`; `models/LoSTer/experiment.py:36-40` | CONFIRMED: extra 1/L and 1/d; contrastive directional means doubled; configurable alpha supplied as 1 | Complete objective differs by term-specific scales, not a harmless common multiplier | PAPER ONLY |
| F06 | MAJOR | Cluster assignments re-softmaxed | M:340-355 describes loss on assignment/probability columns without defining S=softmax(hard Q) | `models/LoSTer/experiment.py:33-39`; `models/LoSTer/utils/losses.py:64-72,79-97` | CONFIRMED extra smoothing; intention NOT ESTABLISHED | Cluster cosine/entropy uses transformed one-hot columns, with nonzero mass for empty clusters | FURTHER INVESTIGATION |
| F07 | MINOR | Entropy formula grouping | M:349 (`cluster_entropy`) lacks grouping of both view terms under the sum/common minus | `models/LoSTer/utils/losses.py:68-72,97` adds both negative entropies | CONFIRMED notation ambiguity; no demonstrated code sign bug | Printed expression can be read with asymmetric sign/summation scope | PAPER ONLY |
| F08 | MODERATE | Final decoder block | M:239-251,259 describes decoder as residual-block stack without output-block exception | `models/LoSTer/net/model.py:96-109,151-158` | CONFIRMED final block has no Dropout or LayerNorm | Architectural reconstruction from paper alone would differ | PAPER ONLY |
| F09 | MODERATE | Cross-view centroid permutation | M:323,326,340 assumes corresponding cluster columns | `models/LoSTer/main.py:82-108`; `models/LoSTer/utils/losses.py:82-95` | PLAUSIBLE RISK: no explicit matching; shared seed/order may coordinate partially; objective can establish correspondence | Sensitivity to independent label permutations/initialization is unproven, not a confirmed failure | EXPERIMENT NEEDED |
| F10 | MAJOR | Active centroids absent from checkpoint; seeds overwrite artifacts | M:323,380 and reproducible end-to-end clustering require trained centers | `models/LoSTer/net/model.py:201-205`; `models/LoSTer/utils/losses.py:10-15`; `models/LoSTer/main.py:28-49,112-116`; `models/LoSTer/experiment.py:66,135` | CONFIRMED: model-only save holds dormant centers, not active criteria; paths omit seed | Cannot generally reload the final clustering rule or all five seed models from written files | CODE + RERUN |
| F11 | MAJOR | UCR header inference | M:443-468 gives sample counts and concatenated splits | `models/LoSTer/data/load_data.py:21-23` uses read_csv without header=None; pandas defaults consume first record as header | CONFIRMED parser behavior; historical headerless omission NOT ESTABLISHED without raw TSVs | Headerless files lose two rows per dataset and may alter the oracle k/data population | FURTHER INVESTIGATION |
| F12 | MODERATE | Nonfinite/constant-series handling | M:468,789 gives normalized input protocol but no invalid-series policy | `models/LoSTer/data/load_data.py:10-15,25-27,44-46` has no variance/range guard or imputation | PLAUSIBLE RISK; actual affected records NOT ESTABLISHED | Invalid normalized values can contaminate augmentation, embeddings, or KMeans | FURTHER INVESTIGATION |
| F13 | MODERATE | Permutation distribution | M:230 says rounded positive Gaussian with std 5 for split count | `models/LoSTer/data/augmentation.py:19-36` draws uniform 1-4 and equal-size segments | CONFIRMED manuscript parameter mismatch | Different augmentation distribution would result from following the paper | PAPER ONLY |
| F14 | MODERATE | Time-warp monotonicity | M:230 describes smooth temporal distortion | `models/LoSTer/data/augmentation.py:56-64` spline/rescale/clip lacks monotonicity check | PLAUSIBLE RISK; no guarantee, historical violations NOT ESTABLISHED | Non-increasing interpolation coordinates need not be valid chronological warping | FURTHER INVESTIGATION |
| F15 | MODERATE | Two precomputed augmentation sets | M:230,364 describes advance augmentation without specifying two loader constructions | `models/LoSTer/main.py:62-76,97,105-107`; `models/LoSTer/data/load_data.py:27,46,74-81` | FACT: fixed training and separately generated initialization/evaluation augmentations | Augmented centroids initialize on a different fixed view from pretraining/joint batches | PAPER ONLY |
| F16 | MINOR | Annealing description | M:390 says linear; M:427 states exponential schedule and floor | `models/LoSTer/experiment.py:21` uses initial tau times beta^epoch, floor 0.01 | CONFIRMED internal wording contradiction; detailed setup matches code | Incorrect schedule if linear wording is followed | PAPER ONLY |
| F17 | NONE | Stopping criterion | M:427 specifies less than 0.1% assignment changes between epochs | `models/LoSTer/experiment.py:59-74,121-133` full ordered deterministic predictions, compare only after first epoch | NO ISSUE in specified criterion | Correctly measures global original-view assignment stability, not stochastic minibatch changes | NO ACTION |
| F18 | MODERATE | Empty/collapsed clusters | M:346 says entropy is incorporated to avoid degeneration | `models/LoSTer/utils/losses.py:64-72`; `models/LoSTer/experiment.py:75-78,127-132`; `models/LoSTer/main.py:141-150` | PLAUSIBLE RISK: regularizer, not guarantee; no reseeding/collapse rejection | Stable collapsed assignments could satisfy stopping; actual collapse NOT ESTABLISHED | EXPERIMENT NEEDED |
| F19 | MAJOR | M5 zero-count slice | M:789 states sparse-series filtering and 1913-step data; notebook heading says drop many-zero sales | `M5_EDA_cat_store_id.ipynb` cells 14,16:1-6,17; cell 10 saved column order | CONFIRMED: selected slice includes store_id and omits d_1913; subset impact NOT ESTABLISHED | Rows at the threshold boundary can enter a different training/evaluation subset | CODE + RERUN |
| F20 | MODERATE | M5 preprocessing disclosure | M:789 omits weekly STL remainder removal and per-series min-max scaling | `M5_EDA_cat_store_id.ipynb` cell 20:2-7,32-36; `models/LoSTer/data/load_data.py:40-46` | CONFIRMED undocumented preprocessing, not inherently a bug | Clusters are learned from denoised/scaled profiles rather than raw daily sales | PAPER ONLY |
| F21 | MAJOR | Store Sales reproducibility | M:791,793,824 reports 1729x1684, 33 classes, scores/timing | `models/LoSTer/run.sh:8-25`; `models/LoSTer/data/load_data.py:74-81`; repository script/data inventory | NOT ESTABLISHED: no supplied preparation/explicit route/run artifacts | Cannot verify or reproduce a complete reported dataset experiment from this checkout | FURTHER INVESTIGATION |
| F22 | MODERATE | Convergence and timing provenance | M:746,762,824,826 discusses convergence over 100 epochs and separate batch-32 timing | `plots_training.ipynb` cells 3-15,18-30; `csv_logs/*`; `models/LoSTer/experiment.py:49-55,99-110`; `models/LoSTer/run.sh:37,52` | FACT: LoSTer columns have 11 entries, compatible with stopping; seed/config/timing linkage NOT ESTABLISHED | Single-run curves/absent timing procedure do not establish replicated convergence or performance attribution | FURTHER INVESTIGATION |
| F23 | MAJOR | NMI normalization | M:508-511 uses geometric entropy mean | `models/LoSTer/utils/metrics.py:21`; `requirements.txt:39`; default arithmetic in official 1.4 API | CONFIRMED definition/call mismatch; actual historical metric provenance NOT ESTABLISHED | Formula and reproducible score convention differ; historical tables need attribution before recomputation | FURTHER INVESTIGATION |
| F24 | MINOR | Available ARI and standard deviations | M:393 and result tables report mean RI/NMI only | `models/LoSTer/main.py:125-136`; `models/LoSTer/results.py:43-50,67-80` | FACT: ARI and population std already computed; publication omits them | Existing per-run artifacts could support uncertainty reporting without retraining if recovered | PAPER ONLY |
| F25 | MINOR | Percent versus percentage points | M:546,582,591,620,651,698,734 reports scaled absolute score differences as percentages | Displayed mean-score arithmetic in M; `models/LoSTer/results.py:67-88` supplies macro means, no relative-percent calculation | CONFIRMED point-difference convention | Relative improvements have a different denominator and interpretation | PAPER ONLY |
| F26 | MINOR | iTransformer exception dataset name | M:664 says EOGVerticalSignal; M:609,615,640,646 shows ECG5000 and StarLightCurves | No runtime evidence needed: contradiction within the manuscript's own tables | CONFIRMED manuscript text/table inconsistency | Wrong benchmark named as an exception; numerical scores need not change | PAPER ONLY |
| F27 | MODERATE | Scaling, fixed L, and non-universal compression | M:195,233,253,831-833 calls dense method scalable/lower-dimensional/arbitrarily long | `models/LoSTer/net/model.py:142-176`; M:443-459 lengths; constructor parameter algebra | FACT: O(Ld+d^2), fixed learned input/output length; d>L on five benchmarks | Qualify scalability and bottleneck claims; does not refute structural linear-in-L scaling | PAPER ONLY |
| F28 | NONE | Univariate scope and view independence | M:228,266,836 describes scalar inputs, different weights, univariate limitation | `models/LoSTer/main.py:82-85`; `models/LoSTer/net/model.py:140,163-176`; `models/LoSTer/data/load_data.py:24,62` | NO ISSUE with those explicit descriptions; no intrinsic multivariate pipeline established | Current scope is narrow but acknowledged; extensions are methodological changes | NO ACTION |
| F29 | NONE | Supplied training hyperparameters | M:405-429 gives blocks, width, epochs, LR, temperatures, decay, batch size | `models/LoSTer/run.sh:28-45,52`; `models/LoSTer/main.py:87-117`; `models/LoSTer/experiment.py:21,57` | NO ISSUE for the supplied UCR/M5 configuration | No evidence of contradictory dataset overrides; historical actual invocations still need provenance | NO ACTION |
| F30 | NONE | Oracle k and labels in updates | M:228,362 supplies k; M:468 combines train/test | `models/LoSTer/data/load_data.py:23-34,50,61`; `models/LoSTer/main.py:78`; `models/LoSTer/experiment.py:24,124-132` | FACT: k comes from true labels; losses discard labels; best RI not used for selection | Known-k benchmark protocol, not class-supervised gradient training; disclose oracle choice explicitly | NO ACTION |
| F31 | NONE | Deterministic final assignment | Algorithm at M:380 outputs argmax probabilities | `models/LoSTer/experiment.py:60-74,140` original-view argmax without Gumbel | NO ISSUE; stochastic training and deterministic evaluation intentionally differ | Final in-memory predictions match the paper's stated inference rule | NO ACTION |
| F32 | MINOR | Pretraining pseudocode loop | Algorithm at M:366-371 says while not converged; M:414/427 says 50 epochs | `models/LoSTer/net/model.py:10,32` fixed epoch loops | CONFIRMED coarse pseudocode/actual-loop mismatch; experimental setup is accurate | Clarify fixed pretraining budget instead of implying a separate convergence test | PAPER ONLY |

## Verification of H1-H10

| Hypothesis | Verdict | Evidence and qualification |
| --- | --- | --- |
| H1: paper fails to explicitly describe straight-through although code uses it | PARTIALLY CONFIRMED | `models/LoSTer/experiment.py:33-34` uses the built-in hard estimator. M:293 already explicitly describes hard forward/soft backward, so absence of the mechanism is refuted. The name/detach formula and precise role of annealing are missing; see F01. |
| H2: sigma is undefined or absent from code | PARTIALLY CONFIRMED | Absence from code is REFUTED by `models/LoSTer/utils/gumbel.py:27-29` (`sigma=1.0`). Active calls omit override. M:296/300 and the hyperparameter table do not define/report its fixed value; paper-description part is confirmed (F02). |
| H3: cluster equation sums over n instead of k | CONFIRMED | M:343 uses n; `models/LoSTer/utils/losses.py:79-95` forms k anchors from k columns and k targets. See F04. |
| H4: independent encoders and centroids create possible permutation ambiguity | CONFIRMED | `models/LoSTer/main.py:82-108` creates separate networks/fits; active criteria have separate matrices at 112-113. No explicit matching exists. The ambiguity is a PLAUSIBLE RISK, not a confirmed bug; shared seed/order and the contrastive objective can help establish correspondence (F09). Four centroid copies are stored, only two are active. |
| H5: unweighted heterogeneous objective has substantially different mathematical scales | PARTIALLY CONFIRMED | `models/LoSTer/experiment.py:36-40` has nominal unit weights in run.sh but a configurable alpha. Reduction factors 1/L,1/d and doubled contrastive directions are confirmed; code is not uniformly equivalent to M's sum. Empirical magnitude/gradient dominance is Not established from static inspection. See F05/F06 and mathematical reference scales. |
| H6: dense architecture has O(L*d) dependence and scalable needs qualification | CONFIRMED | `models/LoSTer/net/model.py:142-158` yields one-AE parameters `4Ld+14d^2+26d+2L`, i.e. O(Ld+d^2) with fixed block count; O(Ld) describes only L-dependence at fixed d. Full losses add B/k-dependent costs. No benchmark inference was made (F27). |
| H7: fixed-length/univariate implementation despite broader LSTC wording | PARTIALLY CONFIRMED | Fixed-length and univariate implementation is confirmed by `models/LoSTer/net/model.py:140,142-176` and loaders. However, M:228/836 already explicitly limits the problem/study to scalar series; a hidden claim of demonstrated multivariate support is refuted. Broad scalability language still needs fixed-L qualification (F27/F28). |
| H8: per-series normalization may remove amplitude information | CONFIRMED | `models/LoSTer/data/load_data.py:10-15,25-26,44-45` removes absolute level and positive affine amplitude scale for nonconstant sequences. Mathematical invariance is established; empirical harm is not. M:468 already states UCR per-series normalization. |
| H9: true cluster count is obtained from labels | CONFIRMED | `models/LoSTer/data/load_data.py:34,50,61`; `models/LoSTer/main.py:78` supply the number of encoded true classes. Labels do not enter training losses. Paper should identify k as oracle class count, not estimated unknown cluster count (F30). |
| H10: final assignment differs from stochastic training | CONFIRMED | `models/LoSTer/experiment.py:33-34` samples during training; 66-67 uses deterministic original-view argmax at evaluation. This is explicitly consistent with Algorithm 1 at M:380 and is NO ISSUE (F31). |

## Changes That Require Rerunning Experiments

No change below was made or authorized for execution in this audit.

1. **F10 — complete recoverable checkpoints and per-seed artifact paths.** Save the active original/augmented centroids, both trained view networks if full training recovery is needed, and configuration/seed/commit/environment; capture optimizer/scheduler state for resumability. Fixing serialization does not itself change the training objective or prove old metric scores invalid. **Rerunning historical runs is required to recover exact missing trained model states if no external archive of those active states exists.** If such states are found, migration/serialization may suffice without retraining. A final prediction array alone cannot in general recover the exact active centroids.
2. **F19 — full sales-day filter.** Correcting the counted columns changes the data preparation definition. **First compare the intended full-day subset with the archived subset.** If membership is unchanged, no result rerun is needed solely for this correction. If membership/labels/count differ, rerun all affected M5 comparisons on a documented common subset; do not regenerate only LoSTer and compare it against old baselines with different data. Remove or parameterize the fixed 1150 loop only in a separately authorized implementation change.

Other findings that could lead to reruns **after investigation**, rather than mandatory code changes now:

- F11: if historical TSVs were headerless and the executed parser lost rows, fixing header handling changes the sample population; affected benchmark comparisons must be rerun consistently.
- F06: if the extra softmax was unintended, removing it or substituting RBF/relaxed assignments changes column contrast and entropy gradients; rerun affected training. If it was deliberate, document S without changing legacy results.
- F05/F03: implementing the currently printed loss scalings/masks rather than documenting the executed ones changes optimization and requires reruns. Manuscript fixes describing current code do not.
- F14/F12: changing warping or invalid-series handling requires reruns only where behavior/data actually changes. Missing safeguards alone do not establish affected historical results.
- F09/F18: shared centroids, explicit column matching, collapse prevention, or altered balancing are methodological choices and require controlled new experiments before replacement of results.
- F23: changing the NMI convention requires **metric recomputation**, not intrinsically retraining. Retraining/regeneration may be needed only when original assignments cannot be recovered.

## Changes That Are Manuscript-Only

The 16 primary PAPER ONLY rows are F01-F05, F07-F08, F13, F15-F16, F20, F24-F27, and F32. They can be resolved without changing numerical scores or scientific implementations by accurately documenting the existing method:

- State the straight-through estimator and its surrogate backward gradient; define sigma=1 separately from tau; correct positive/self masks and n/k bounds.
- Give reductions per coordinate, actual contrastive directional factors, entropy grouping, and the configurable alpha with supplied value 1.
- Identify the last decoder block exception, uniform 1-4 segment distribution, two fixed augmentation snapshots, exponential closed-form annealing, and fixed 50-epoch pretraining.
- Describe M5 weekly STL seasonal+trend reconstruction and per-series min-max scaling; qualify fixed-length scaling and the non-compressive short-sequence cases.
- Relabel absolute improvements as percentage points and correct ECG5000 in the iTransformer exception sentence.
- State that ARI and ddof=0 standard deviations are computed in the code. Adding actual uncertainty values to tables still requires recovered per-seed evidence; they cannot be invented from means.

For F30 (primary NO ACTION), explicitly stating known/oracle k is advisable clarification, although the paper already supplies k as an algorithm input and the implementation is not erroneous. For F23, a paper-only NMI formula correction is appropriate **once** the historical score convention is verified. These clarifications do not authorize editing the immutable `paper/original_r1/`; any future manuscript edits belong in `paper/revision_2026/`.

## Unresolved Questions

1. Which exact code commits, raw-data checksums, package versions, CLI invocations, and seeds produced the paper tables, retail results, convergence curves, and speed measurements? Current pins and saved notebook outputs do not establish this provenance.
2. Did historical UCR TSVs contain headers? What was the actual concatenated sample count/class set after parsing? No raw TSV is present for inspection.
3. Are active trained centroids, complete per-seed model states, assignment arrays, metric CSVs, or W&B artifacts archived outside this repository? This determines whether repair requires retraining versus artifact recovery.
4. Was the extra softmax of one-hot assignments deliberate? Should cluster contrast/entropy use hard, relaxed, RBF, or the current S assignments? The code establishes what happened, not which alternative authors intended.
5. How robust is implicit cluster-column alignment to independent initialization/permutation? What collapse/empty-cluster behavior actually occurred? These are empirical questions not settled by the two-centroid design.
6. Do any generated time-warp coordinates violate monotonicity? Were nonfinite/constant inputs present? This audit did not generate augmentations or inspect absent raw arrays.
7. Does counting d_1913 change the M5 filter boundary, 1150 series, or 29 classes? Was the source notebook rerun exactly as saved, and did all models use the same processed arrays?
8. Where is the Store Sales extraction/filter/label/normalization pipeline, its 33-class configuration, and its batch-32 timing invocation? Generic loader compatibility is not a supplied preprocessing recipe.
9. Was historical NMI arithmetic or geometric? Can final assignments support recomputation without training? Which metric convention did external baseline score sources use? No comparison-method literature was researched here.
10. Are the 11-entry LoSTer convergence columns single selected runs, and which seeds/configs do they represent? The tracked plotting notebook has no across-seed aggregation or direct assignment-change evidence.

For each missing factual answer above: **Not established from static inspection.** No uncertainty was silently resolved by assuming intended behavior or by assuming the paper's numbers must have come from the current code.

## Recommended Next Audit

Prioritize a **read-only data/run-provenance and artifact-recovery audit**: locate the original datasets/headers and per-seed outputs, recover active model/centroid states if possible, identify the score convention, and obtain the missing Store Sales/timing recipes. This can resolve F10/F11/F19/F21-F23 before changing scientific behavior. Separately authorized small semantic checks could then investigate the extra-softmax estimator, warping validity, cluster alignment, and collapse sensitivity. Do not begin training or baseline changes on the basis of this report alone.

Novelty, comparisons with CKM/TiDE/CDCC or 2025/2026 methods, statistical-significance methodology, and new experiment design were not audited. Any literature/novelty review is a separate task; the present recommendations do not perform it.

## Concise Audit Summary

- CRITICAL: **0**
- MAJOR: **8**
- CODE + RERUN: **2**, with the recovery/subset conditions above
- PAPER ONLY: **16**
- H1 PARTIALLY CONFIRMED; H2 PARTIALLY CONFIRMED; H3 CONFIRMED; H4 CONFIRMED; H5 PARTIALLY CONFIRMED; H6 CONFIRMED; H7 PARTIALLY CONFIRMED; H8 CONFIRMED; H9 CONFIRMED; H10 CONFIRMED.

Most important five findings:

1. **F10:** the saved checkpoint does not include the centroids used by trained assignment; per-seed model files are overwritten.
2. **F03:** instance/cluster equations mask the wrong denominator terms, giving a different objective from execution.
3. **F05:** per-coordinate reductions and directional factors change relative loss scaling from the manuscript.
4. **F06:** cluster loss re-softmaxes hard assignments, changing the columns and entropy statistics; intent needs investigation.
5. **F23:** code defaults to arithmetic NMI while the paper defines geometric NMI; historical score provenance must be verified.


## Safety Verification

Final `git status --short`:

```text
?? analysis/audits/01-paper-code-audit.md
```

The report is the only new/modified workspace file and remains untracked/unstaged.
All 297 pre-existing non-Git workspace files matched the audit-start inventory:
file paths, byte sizes, modification timestamps, and SHA-256 fingerprints were
unchanged. This includes every file in `models/` (246), both manuscript trees
(24), `Drafts/` (3), `csv_logs/` (4), and legacy `figures/` (2), as well as the
other pre-existing files. `git diff -- models paper csv_logs figures` was empty.
HEAD remains `0c83687c03d854817112f61dd29dde58903a8ff1` on `revision-2026`.

The report's 32 finding rows, permitted severity/action labels, H1-H10 coverage,
required sections, and full-path evidence line bounds were programmatically
checked. No scientific project module was imported for execution. No training,
new experiment, package installation, dependency update, manuscript edit, bug
fix, refactoring, LaTeX compilation, commit, or push was performed. Historical
correspondence was not reproduced. Stopped after producing this audit.
