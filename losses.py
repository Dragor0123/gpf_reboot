import torch
import torch.nn.functional as F


def compute_ce_loss(logits, labels):
    """
    Standard cross-entropy loss for classification.
    """
    return F.cross_entropy(logits, labels)


def compute_kl_divergence(p_logits, q_logits, reduction='batchmean'):
    """
    KL divergence between two probability distributions from logits.
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction=reduction)


def gaussian_kernel(x, y, sigma=1.0):
    """
    RBF (Gaussian) kernel between two batches of vectors.
    """
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma ** 2))


def compute_mmd_loss(x_samples, y_samples, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between x_samples and y_samples.
    """
    K_xx = gaussian_kernel(x_samples, x_samples, sigma)
    K_yy = gaussian_kernel(y_samples, y_samples, sigma)
    K_xy = gaussian_kernel(x_samples, y_samples, sigma)

    m = x_samples.size(0)
    n = y_samples.size(0)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def total_loss(ce=None, kl=None, mmd=None, weights={'ce':1.0, 'kl':0.1, 'mmd':0.1}):
    """
    Aggregate loss function with optional CE, KL, and MMD terms.
    """
    loss = 0.0
    if ce is not None:
        loss += weights['ce'] * ce
    if kl is not None:
        loss += weights['kl'] * kl
    if mmd is not None:
        loss += weights['mmd'] * mmd
    return loss
