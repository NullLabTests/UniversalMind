import numpy as np
from godelion_extended.metrics.coordination import (
    compute_mutual_information_matrix,
    compute_synchronization_index,
    compute_communication_entropy,
    compute_coordination_score,
    compute_redundancy_ratio,
)
from godelion_extended.metrics.emergence import (
    detect_phase_transitions,
    compute_cross_prediction_error,
    compute_shared_representation_score,
)
from godelion_extended.metrics.information_theory import (
    estimate_entropy,
    estimate_mutual_information,
    compute_kl_divergence,
)


def test_mutual_information_matrix():
    action_matrix = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    ])
    mi = compute_mutual_information_matrix(action_matrix)
    assert mi.shape == (3, 3)
    assert mi[0, 1] > 0  # first two agents are perfectly correlated
    assert np.allclose(mi, mi.T)  # symmetric


def test_synchronization_index():
    action_matrix = np.array([
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
    ])
    sync = compute_synchronization_index(action_matrix)
    assert 0 <= sync <= 1
    assert sync > 0.5  # first two should be in sync


def test_communication_entropy():
    comm_matrix = np.random.randn(3, 20, 2)
    entropy = compute_communication_entropy(comm_matrix)
    assert entropy.shape == (3,)
    assert np.all(entropy >= 0)


def test_coordination_score():
    action_matrix = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
    ])
    score = compute_coordination_score(action_matrix)
    assert 0 <= score <= 1


def test_redundancy_ratio():
    comm_matrix = np.random.randn(3, 10, 2)
    action_matrix = np.random.randint(0, 4, size=(3, 10))
    ratio = compute_redundancy_ratio(comm_matrix, action_matrix)
    assert ratio >= 0


def test_phase_transitions():
    trajectory = np.random.randn(2, 100)
    transitions = detect_phase_transitions(trajectory, window=10, threshold=2.0)
    assert isinstance(transitions, list)


def test_cross_prediction_error():
    action_matrix = np.random.randint(0, 4, size=(3, 30))
    error = compute_cross_prediction_error(action_matrix, lookback=3)
    assert 0 <= error <= 1


def test_shared_representation_score():
    comm_matrix = np.random.randn(3, 20, 2)
    score = compute_shared_representation_score(comm_matrix)
    assert 0 <= score <= 1


def test_estimate_entropy():
    data = np.random.randn(100)
    h = estimate_entropy(data)
    assert h >= 0


def test_estimate_mutual_information():
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.1
    mi = estimate_mutual_information(x, y)
    assert mi >= 0


def test_kl_divergence():
    p = np.random.randn(100)
    q = np.random.randn(100)
    kl = compute_kl_divergence(p, q)
    assert kl >= 0


def test_perfect_correlation_mi():
    action_matrix = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ])
    mi = compute_mutual_information_matrix(action_matrix)
    assert mi[0, 1] >= 0.5  # should have high MI
