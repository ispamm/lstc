"""Exact active legacy utils/losses.py and experiment.py:36-40 (F03-F07).
Only NoResoftmax changes the column representation and specified zero conventions.
"""
import torch
from torch.nn import functional as F

def kmeans_loss(z, q, centers):
    return torch.mean(torch.square(z - q @ centers))

def _directions(aa, bb, ab, ba, mask_value):
    m = aa.shape[0]
    mask = torch.eye(m, device=aa.device) * mask_value
    target = torch.arange(m, device=aa.device)
    return (F.cross_entropy(torch.cat([ab, aa - mask], dim=1), target) +
            F.cross_entropy(torch.cat([ba, bb - mask], dim=1), target))

def instance_loss(z, za, config):
    z = F.normalize(z, dim=1, eps=config.instance_norm_eps)
    za = F.normalize(za, dim=1, eps=config.instance_norm_eps)
    t = config.instance_temperature
    return _directions(z @ z.T / t, za @ za.T / t,
                       z @ za.T / t, za @ z.T / t, config.self_mask)

def cluster_representation(q, config):
    return q if config.variant == "LoSTer-NoResoftmax" else F.softmax(q, dim=1)

def cluster_terms(q, qa, config):
    s, sa = cluster_representation(q, config), cluster_representation(qa, config)
    p, pa = s.sum(0), sa.sum(0)
    p, pa = p / p.sum(), pa / pa.sum()
    direct = config.variant == "LoSTer-NoResoftmax"
    if direct:
        entropy = (p * p.clamp_min(config.noresoftmax_entropy_eps).log()).sum()
        entropy = entropy + (pa * pa.clamp_min(config.noresoftmax_entropy_eps).log()).sum()
    else:
        entropy = (p * p.log()).sum() + (pa * pa.log()).sum()
    a, b = s.T, sa.T
    def cosine(x, y):
        if direct:
            floor = config.noresoftmax_count_norm_floor
            return (x @ y.T) / (x.norm(dim=1).clamp_min(floor)[:, None] *
                                y.norm(dim=1).clamp_min(floor)[None, :])
        return F.cosine_similarity(x[:, None, :], y[None, :, :],
                                   dim=2, eps=config.legacy_cosine_eps)
    t = config.cluster_temperature
    contrast = _directions(cosine(a, a) / t, cosine(b, b) / t,
                           cosine(a, b) / t, cosine(b, a) / t, config.self_mask)
    return contrast, entropy

def objective(x, xa, xr, xar, z, za, q, qa, co, ca, config):
    rec = F.mse_loss(xr, x) + F.mse_loss(xar, xa)
    km = 0.5 * (kmeans_loss(z, q, co) + kmeans_loss(za, qa, ca))
    inst = instance_loss(z, za, config)
    contrast, entropy = cluster_terms(q, qa, config)
    total = rec + config.alpha * km + inst + (contrast + entropy)
    return {"total": total, "reconstruction": rec, "kmeans": km,
            "instance": inst, "cluster_contrast": contrast, "entropy": entropy}
