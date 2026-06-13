from .coordination import (
    compute_mutual_information_matrix,
    compute_synchronization_index,
    compute_communication_entropy,
    compute_coordination_score,
    compute_redundancy_ratio,
)
from .emergence import (
    detect_phase_transitions,
    compute_cross_prediction_error,
    compute_shared_representation_score,
)
from .information_theory import (
    estimate_entropy,
    estimate_mutual_information,
    compute_kl_divergence,
)

__all__ = [
    "compute_mutual_information_matrix",
    "compute_synchronization_index",
    "compute_communication_entropy",
    "compute_coordination_score",
    "compute_redundancy_ratio",
    "detect_phase_transitions",
    "compute_cross_prediction_error",
    "compute_shared_representation_score",
    "estimate_entropy",
    "estimate_mutual_information",
    "compute_kl_divergence",
]
