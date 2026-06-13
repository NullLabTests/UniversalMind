import numpy as np
from typing import List


def detect_phase_transitions(trajectory: np.ndarray, window: int = 10, threshold: float = 2.0) -> List[int]:
    n_dims, n_steps = trajectory.shape
    if n_steps < window * 2:
        return []

    velocities = np.diff(trajectory, axis=1)
    speed = np.mean(np.abs(velocities), axis=0)

    if len(speed) < window:
        return []

    transitions = []
    for t in range(window, len(speed) - window):
        local_mean = np.mean(speed[t - window : t])
        local_std = np.std(speed[t - window : t]) + 1e-8
        future_mean = np.mean(speed[t : t + window])
        z = abs(future_mean - local_mean) / local_std
        if z > threshold:
            transitions.append(t)

    merged = []
    if transitions:
        merged.append(transitions[0])
        for t in transitions[1:]:
            if t - merged[-1] > window:
                merged.append(t)

    return merged


def compute_cross_prediction_error(action_matrix: np.ndarray, lookback: int = 3) -> float:
    n_agents, n_steps = action_matrix.shape
    if n_agents < 2 or n_steps < lookback + 2:
        return 0.0

    errors = []
    for target_idx in range(n_agents):
        predictors = [i for i in range(n_agents) if i != target_idx]

        for t in range(lookback, n_steps - 1):
            target_action = action_matrix[target_idx, t + 1]

            pred_action = 0
            total_weight = 0.0
            for pred_idx in predictors:
                recent_correlation = np.corrcoef(
                    action_matrix[target_idx, t - lookback : t],
                    action_matrix[pred_idx, t - lookback : t],
                )[0, 1]
                if not np.isnan(recent_correlation):
                    weight = abs(recent_correlation)
                    pred_action += weight * action_matrix[pred_idx, t]
                    total_weight += weight

            if total_weight > 0:
                pred_action = int(round(pred_action / total_weight)) % 4
            else:
                pred_action = target_action

            errors.append(1.0 if pred_action != target_action else 0.0)

    return float(np.mean(errors)) if errors else 0.5


def compute_shared_representation_score(comm_matrix: np.ndarray) -> float:
    n_agents, n_steps, comm_dim = comm_matrix.shape
    if n_agents < 2 or n_steps < 3:
        return 0.0

    pairwise_sims = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            traj_i = comm_matrix[i].flatten()
            traj_j = comm_matrix[j].flatten()
            if np.std(traj_i) > 0 and np.std(traj_j) > 0:
                corr = np.corrcoef(traj_i, traj_j)[0, 1]
                if not np.isnan(corr):
                    pairwise_sims.append(abs(corr))

    return float(np.mean(pairwise_sims)) if pairwise_sims else 0.0


def compute_emergence_index(coord_over_time: np.ndarray, comm_bandwidth: np.ndarray) -> float:
    if len(coord_over_time) < 2 or len(comm_bandwidth) < 2:
        return 0.0

    coord_slope = np.polyfit(np.arange(len(coord_over_time)), coord_over_time, 1)[0]
    comm_slope = np.polyfit(np.arange(len(comm_bandwidth)), comm_bandwidth, 1)[0]

    if abs(comm_slope) < 1e-8:
        return float(coord_slope)

    return float(coord_slope / comm_slope)
