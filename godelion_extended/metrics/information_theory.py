import numpy as np
from typing import Optional


def estimate_entropy(data: np.ndarray, bins: Optional[int] = None) -> float:
    if len(data) < 2:
        return 0.0

    if bins is None:
        bins = min(int(np.sqrt(len(data))), 20)

    data = data.flatten()
    hist, _ = np.histogram(data, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def estimate_mutual_information(x: np.ndarray, y: np.ndarray, bins: Optional[int] = None) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    if bins is None:
        bins = min(int(np.sqrt(len(x))), 10)

    c = np.clip
    x_bins = np.digitize(c(x, x.min(), x.max()), np.linspace(x.min(), x.max() + 1e-10, bins)) - 1
    y_bins = np.digitize(c(y, y.min(), y.max()), np.linspace(y.min(), y.max() + 1e-10, bins)) - 1

    n = len(x_bins)
    p_x = np.bincount(x_bins, minlength=bins) / n
    p_y = np.bincount(y_bins, minlength=bins) / n

    p_xy = np.zeros((bins, bins))
    for xi, yi in zip(x_bins, y_bins):
        p_xy[xi, yi] += 1
    p_xy /= n

    mi = 0.0
    for xi in range(bins):
        for yi in range(bins):
            if p_xy[xi, yi] > 0 and p_x[xi] > 0 and p_y[yi] > 0:
                mi += p_xy[xi, yi] * np.log2(p_xy[xi, yi] / (p_x[xi] * p_y[yi]))

    return mi


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, bins: Optional[int] = None) -> float:
    if bins is None:
        bins = min(int(np.sqrt(len(p))), 20)

    p_hist, _ = np.histogram(p.flatten(), bins=bins, density=True)
    q_hist, _ = np.histogram(q.flatten(), bins=bins, density=True)

    p_hist = p_hist / (p_hist.sum() + 1e-10)
    q_hist = q_hist / (q_hist.sum() + 1e-10)

    kl = 0.0
    for pi, qi in zip(p_hist, q_hist):
        if pi > 0 and qi > 0:
            kl += pi * np.log2(pi / qi)
        elif pi > 0 and qi == 0:
            kl += pi * np.log2(pi / 1e-10)

    return kl


def compute_total_correlation(data: np.ndarray, bins: int = 5) -> float:
    n_vars, n_samples = data.shape
    if n_vars < 2 or n_samples < 2:
        return 0.0

    if data.dtype.kind in ('i', 'u'):
        discrete = True
    else:
        discrete = False

    if discrete:
        h_individual = 0.0
        for i in range(n_vars):
            counts = np.bincount(data[i].astype(int))
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h_individual += -np.sum(probs * np.log2(probs))

        joint_counts = np.zeros((bins,) * n_vars)
        for t in range(n_samples):
            idx = tuple(min(int(data[i, t]), bins - 1) for i in range(n_vars))
            joint_counts[idx] += 1
        joint_probs = joint_counts / joint_counts.sum()
        joint_probs = joint_probs[joint_probs > 0]
        h_joint = -np.sum(joint_probs * np.log2(joint_probs))

        return float(h_individual - h_joint)

    h_individual = 0.0
    for i in range(n_vars):
        h_individual += estimate_entropy(data[i], bins)

    h_joint = estimate_entropy(data.flatten(), bins)

    return float(h_individual - h_joint)
