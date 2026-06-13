from typing import Dict, List, Optional, Tuple

import numpy as np

from .environment import UniverseDialogueConfig
from .system import MultiAgentSystem
from ..metrics.coordination import (
    compute_mutual_information_matrix,
    compute_synchronization_index,
    compute_communication_entropy,
    compute_coordination_score,
    compute_redundancy_ratio,
)
from ..metrics.emergence import (
    detect_phase_transitions,
    compute_cross_prediction_error,
    compute_shared_representation_score,
)


class UniverseDialogueAnalyzer:
    def __init__(self, config: UniverseDialogueConfig):
        self.config = config

    def analyze_episode(self, system: MultiAgentSystem,
                        comm_history: List[Dict], action_history: List[Dict],
                        reward_history: List[Dict]) -> Dict:
        comm_matrix = system.get_comm_matrix(comm_history)
        action_matrix = system.get_action_matrix(action_history)
        reward_matrix = system.get_reward_matrix(reward_history)

        n_steps = comm_matrix.shape[1]
        n_agents = self.config.n_agents
        comm_dim = self.config.comm_dim

        if n_steps < 3:
            return {"error": "too few steps"}

        metrics = {}

        if n_agents >= 2 and n_steps >= 5:
            mi_matrix = compute_mutual_information_matrix(action_matrix)
            metrics["mutual_information"] = {
                "mean": float(np.mean(mi_matrix)),
                "max": float(np.max(mi_matrix)),
                "matrix": mi_matrix.tolist(),
            }

            sync_idx = compute_synchronization_index(action_matrix)
            metrics["synchronization_index"] = float(sync_idx)

        comm_entropy = compute_communication_entropy(comm_matrix)
        metrics["communication_entropy"] = {
            "mean": float(np.mean(comm_entropy)),
            "per_agent": comm_entropy.tolist(),
        }

        coord_score = compute_coordination_score(action_matrix)
        metrics["coordination_score"] = float(coord_score)

        if n_agents >= 2 and n_steps >= 10:
            redundancy = compute_redundancy_ratio(comm_matrix, action_matrix)
            metrics["redundancy_ratio"] = float(redundancy)

        # Emergence indicators
        if n_steps >= 20:
            phases = detect_phase_transitions(
                comm_matrix.reshape(n_agents, -1),
                window=min(10, n_steps // 5),
            )
            metrics["phase_transitions"] = {
                "n_transitions": len(phases),
                "transition_points": phases,
            }

        if n_agents >= 2 and n_steps >= 10:
            cross_pred = compute_cross_prediction_error(action_matrix)
            metrics["cross_prediction_error"] = float(cross_pred)

        if n_agents >= 2 and n_steps >= 5:
            shared_rep = compute_shared_representation_score(comm_matrix)
            metrics["shared_representation_score"] = float(shared_rep)

        # Agent-level rewards
        agent_rewards = [system.agents[i].total_reward for i in range(n_agents)]
        metrics["agent_rewards"] = {
            "mean": float(np.mean(agent_rewards)),
            "std": float(np.std(agent_rewards)),
            "min": float(np.min(agent_rewards)),
            "max": float(np.max(agent_rewards)),
            "per_agent": agent_rewards,
        }

        metrics["total_system_reward"] = float(np.sum(agent_rewards))
        metrics["n_steps"] = n_steps

        # Reward variance across agents
        metrics["reward_variance"] = float(np.var(agent_rewards))
        metrics["reward_inequality"] = float(
            np.std(agent_rewards) / (np.mean(agent_rewards) + 1e-8)
        )

        return metrics

    def compute_generation_metrics(self, episode_metrics: List[Dict]) -> Dict:
        if not episode_metrics:
            return {}

        gen_metrics = {}

        for key in ["total_system_reward", "coordination_score", "synchronization_index",
                     "redundancy_ratio", "cross_prediction_error", "shared_representation_score",
                     "reward_variance", "reward_inequality"]:
            values = [m.get(key, np.nan) for m in episode_metrics if key in m]
            values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            if values:
                gen_metrics[f"{key}_mean"] = float(np.mean(values))
                gen_metrics[f"{key}_std"] = float(np.std(values))
                gen_metrics[f"{key}_max"] = float(np.max(values))

        mi_values = [m.get("mutual_information", {}).get("mean", np.nan)
                     for m in episode_metrics]
        mi_values = [v for v in mi_values if not (isinstance(v, float) and np.isnan(v))]
        if mi_values:
            gen_metrics["mutual_information_mean"] = float(np.mean(mi_values))

        comm_entropy_values = [m.get("communication_entropy", {}).get("mean", np.nan)
                               for m in episode_metrics]
        comm_entropy_values = [v for v in comm_entropy_values
                               if not (isinstance(v, float) and np.isnan(v))]
        if comm_entropy_values:
            gen_metrics["communication_entropy_mean"] = float(np.mean(comm_entropy_values))

        agent_rews = [m.get("agent_rewards", {}).get("per_agent", [])
                      for m in episode_metrics]
        all_rewards = [r for rew in agent_rews for r in rew]
        if all_rewards:
            gen_metrics["overall_agent_reward_mean"] = float(np.mean(all_rewards))
            gen_metrics["overall_agent_reward_std"] = float(np.std(all_rewards))

        return gen_metrics


def analyze_system(config: UniverseDialogueConfig,
                   system: MultiAgentSystem,
                   n_trials: int = 3) -> Dict:
    analyzer = UniverseDialogueAnalyzer(config)
    trial_metrics = []

    for trial in range(n_trials):
        total_reward, comm_history, action_history, reward_history, metadata = (
            system.run_episode()
        )
        metrics = analyzer.analyze_episode(system, comm_history, action_history, reward_history)
        metrics["trial"] = trial
        trial_metrics.append(metrics)

    gen_metrics = analyzer.compute_generation_metrics(trial_metrics)
    gen_metrics["n_trials"] = n_trials
    gen_metrics["trial_metrics"] = trial_metrics

    return gen_metrics
