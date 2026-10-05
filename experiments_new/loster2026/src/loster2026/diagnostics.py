"""Extends legacy experiment.py monitoring, F06/F18 and Audit 4.
No balancing/reseeding/stopping intervention.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .protocol import changed_fraction

def occupancy(predicted, k, previous=None):
    counts = np.bincount(np.asarray(predicted, dtype=int), minlength=k)
    f = counts / counts.sum()
    positive = f[f > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    occupied = int((counts > 0).sum())
    return {"counts": counts.tolist(), "occupied_clusters": occupied,
            "utilization": occupied / k, "minimum_cluster_size": int(counts.min()),
            "minimum_positive_size": int(counts[counts > 0].min()),
            "maximum_cluster_fraction": float(f.max()), "hard_entropy": entropy,
            "normalized_hard_entropy": entropy / math.log(k) if k > 1 else 0.0,
            "effective_cluster_count": math.exp(entropy),
            "assignment_change_fraction": changed_fraction(predicted, previous),
            "complete_collapse": occupied == 1 and k > 1}

def view_diagnostics(z, centers, predicted, k, previous, sigma):
    import torch
    from .assignments import rbf_logits
    z, centers = z.detach(), centers.detach()  # Monitoring creates no autograd graph.
    result = occupancy(predicted, k, previous)
    p = rbf_logits(z, centers, sigma).exp()
    hard = torch.nn.functional.one_hot(torch.as_tensor(predicted, device=z.device), k).to(z.dtype)
    s0 = hard.softmax(1)
    def entropy(v):
        return -(v * v.clamp_min(1e-30).log()).sum(-1)
    result["rbf_mean_row_entropy"] = float(entropy(p).mean().item())
    result["rbf_marginal_entropy"] = float(entropy(p.mean(0)).item())
    result["s0_mean_row_entropy"] = float(entropy(s0).mean().item())
    result["s0_marginal_entropy"] = float(entropy(s0.mean(0)).item())
    result["center_norms"] = centers.norm(dim=1).detach().cpu().tolist()
    result["mean_nearest_squared_distance"] = float(
        ((z[:, None, :] - centers[None, :, :]) ** 2).sum(2).min(1).values.mean().item())
    result["minimum_center_separation"] = float(torch.pdist(centers).min().item()) if k > 1 else None
    return result

def correspondence(original, augmented, k):
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (original, augmented), 1)
    r, c = linear_sum_assignment(-counts)
    return {"contingency": counts.tolist(),
            "same_index_agreement": float(np.mean(np.asarray(original) == augmented)),
            "permutation_invariant_agreement": float(counts[r, c].sum() / len(original))}

def component_gradients(terms, model, alpha):
    # Audit 4: same graph, no extra stochastic forward or writes to .grad.
    import torch
    named = list(model.named_parameters())
    params = [p for _, p in named]
    groups = {}
    for index, (name, _) in enumerate(named):
        if name.startswith("centers_"):
            key = name
        else:
            view, part = name.split(".", 2)[:2]
            key = view + "." + ("encoder" if part == "dense_encoder" else "decoder")
        groups.setdefault(key, []).append(index)
    result = {}
    for term_name, loss in [("reconstruction", terms["reconstruction"]),
                            ("alpha_kmeans", alpha * terms["kmeans"]),
                            ("instance", terms["instance"]),
                            ("cluster_contrast", terms["cluster_contrast"]),
                            ("entropy", terms["entropy"])]:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        result[term_name] = {}
        for group, indices in groups.items():
            squared = sum(float(grads[i].detach().square().sum().item())
                          for i in indices if grads[i] is not None)
            result[term_name][group] = math.sqrt(squared)
    return result

def total_gradient_norm(model):
    return math.sqrt(sum(float(p.grad.detach().square().sum().item())
                         for p in model.parameters() if p.grad is not None))
