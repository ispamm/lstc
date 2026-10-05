"""Legacy utils/gumbel.py softmax_logits; experiment.py F.gumbel_softmax (F01/F02)."""
import torch
from torch.nn import functional as F

def rbf_logits(z, centers, sigma=1.0):
    if sigma <= 0 or z.shape[1] != centers.shape[1]:
        raise ValueError("Positive sigma and matching latent widths required")
    distances = (z[:, None, :] - centers[None, :, :]).norm(2, dim=2).square()
    return F.log_softmax(-distances / sigma ** 2, dim=1)

def hard_assignments(logits, tau):
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

def deterministic_assignments(z, centers, sigma=1.0):
    return rbf_logits(z, centers, sigma).argmax(dim=1)
