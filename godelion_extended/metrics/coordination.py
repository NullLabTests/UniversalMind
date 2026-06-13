import numpy as np
from typing import Optional


def compute_mutual_information_matrix(action_matrix: np.ndarray) -> np.ndarray:
    n_agents, n_steps = action_matrix.shape
    if n_agents < 2:
        return np.zeros((1, 1))

    mi_matrix = np.zeros((n_agents, n_agents))

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            mi = _mutual_information_discrete(action_matrix[i], action_matrix[j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def _mutual_information_discrete(x: np.ndarray, y: np.ndarray, bins: int = 4) -> float:
    if len(x) < 2:
        return 0.0

    x_bins = np.digitize(x, np.linspace(x.min(), x.max() + 1e-6, bins)) - 1
    y_bins = np.digitize(y, np.linspace(y.min(), y.max() + 1e-6, bins)) - 1

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


def compute_synchronization_index(action_matrix: np.ndarray) -> float:
    n_agents, n_steps = action_matrix.shape
    if n_agents < 2 or n_steps < 2:
        return 0.0

    pairwise_sync = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            agreement = np.mean(action_matrix[i] == action_matrix[j])
            pairwise_sync.append(agreement)

    return float(np.mean(pairwise_sync)) if pairwise_sync else 0.0


def compute_communication_entropy(comm_matrix: np.ndarray) -> np.ndarray:
    n_agents, n_steps, comm_dim = comm_matrix.shape
    entropy_per_agent = np.zeros(n_agents)

    for i in range(n_agents):
        agent_comms = comm_matrix[i]
        entropies = []
        for d in range(comm_dim):
            channel_data = agent_comms[:, d]
            if len(np.unique(channel_data)) > 1:
                h = _estimate_entropy_continuous(channel_data, bins=min(10, n_steps // 2))
                entropies.append(h)
        entropy_per_agent[i] = np.mean(entropies) if entropies else 0.0

    return entropy_per_agent


def _estimate_entropy_continuous(data: np.ndarray, bins: int = 10) -> float:
    if len(data) < 2:
        return 0.0
    hist, _ = np.histogram(data, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_coordination_score(action_matrix: np.ndarray) -> float:
    n_agents, n_steps = action_matrix.shape
    if n_agents < 2 or n_steps < 2:
        return 0.0

    action_entropies = []
    for i in range(n_agents):
        counts = np.bincount(action_matrix[i], minlength=4)[:4]
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log2(probs))
        action_entropies.append(h)

    joint_counts = np.zeros((4, 4))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            for t in range(n_steps):
                joint_counts[action_matrix[i, t], action_matrix[j, t]] += 1

    joint_probs = joint_counts / joint_counts.sum()
    joint_probs = joint_probs[joint_probs > 0]
    h_joint = -np.sum(joint_probs * np.log2(joint_probs))

    mean_h = np.mean(action_entropies)
    if mean_h > 0:
        return float(1.0 - h_joint / (mean_h * 2 + 1e-8))
    return 0.0


def compute_redundancy_ratio(comm_matrix: np.ndarray, action_matrix: np.ndarray) -> float:
    n_agents, n_steps, comm_dim = comm_matrix.shape
    if n_agents < 2 or n_steps < 10 or comm_dim < 1:
        return 0.0

    comm_agreement = 0.0
    total_pairs = 0

    for t in range(n_steps):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                sim = np.dot(comm_matrix[i, t], comm_matrix[j, t])
                norm = np.linalg.norm(comm_matrix[i, t]) * np.linalg.norm(comm_matrix[j, t]) + 1e-8
                comm_agreement += sim / norm
                total_pairs += 1

    avg_comm_sim = comm_agreement / max(total_pairs, 1)

    action_agreement = 0.0
    total_action_pairs = 0
    for t in range(n_steps):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if action_matrix[i, t] == action_matrix[j, t]:
                    action_agreement += 1.0
                total_action_pairs += 1

    action_sim = action_agreement / max(total_action_pairs, 1)

    return float(max(0, avg_comm_sim) / (action_sim + 1e-8))
