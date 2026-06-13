import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ExperimentVisualizer:
    def __init__(self, output_dir: str, dpi: int = 150, fmt: str = "png"):
        self.output_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        self.dpi = dpi
        self.fmt = fmt
        sns.set_theme(style="whitegrid")

    def plot_fitness_over_time(self, best_fitness: List[float], mean_fitness: List[float],
                               filename: str = "fitness_over_time"):
        plt.figure(figsize=(10, 6))
        gens = np.arange(len(best_fitness))
        plt.plot(gens, best_fitness, "b-", label="Best Fitness", linewidth=2)
        plt.plot(gens, mean_fitness, "r--", label="Mean Fitness", linewidth=2)
        plt.fill_between(gens, mean_fitness, best_fitness, alpha=0.1)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Over Generations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_coordination_metrics(self, generation_data: Dict[int, Dict], filename: str = "coordination_metrics"):
        gens = sorted(generation_data.keys())
        if len(gens) < 2:
            return

        plt.figure(figsize=(12, 8))

        metrics_to_plot = [
            ("coordination_score_mean", "Coordination Score", "blue"),
            ("synchronization_index_mean", "Sync Index", "green"),
            ("mutual_information_mean", "Mutual Information", "red"),
        ]

        ax1 = plt.subplot(2, 2, 1)
        for key, label, color in metrics_to_plot:
            values = []
            for g in gens:
                v = generation_data[g].get(key)
                values.append(v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0)
            ax1.plot(gens, values, label=label, color=color, linewidth=2)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Score")
        ax1.set_title("Coordination Metrics")
        ax1.legend()

        ax2 = plt.subplot(2, 2, 2)
        comm_entropy = [
            generation_data[g].get("communication_entropy_mean", 0) or 0
            for g in gens
        ]
        ax2.plot(gens, comm_entropy, "purple", linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Communication Entropy")

        ax3 = plt.subplot(2, 2, 3)
        emergence = [
            generation_data[g].get("shared_representation_score", 0) or 0
            for g in gens
        ]
        ax3.plot(gens, emergence, "orange", linewidth=2)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Score")
        ax3.set_title("Shared Representation Score")

        ax4 = plt.subplot(2, 2, 4)
        cross_pred = [
            generation_data[g].get("cross_prediction_error", 0.5) or 0.5
            for g in gens
        ]
        ax4.plot(gens, cross_pred, "brown", linewidth=2)
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Error")
        ax4.set_title("Cross-Prediction Error")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_reward_distribution(self, reward_matrix: np.ndarray, filename: str = "reward_distribution"):
        plt.figure(figsize=(10, 6))
        n_agents = reward_matrix.shape[0]

        for i in range(n_agents):
            plt.plot(reward_matrix[i], label=f"Agent {i}", alpha=0.7)

        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Agent Rewards Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_communication_heatmap(self, comm_matrix: np.ndarray, filename: str = "communication_heatmap"):
        n_agents, n_steps, comm_dim = comm_matrix.shape
        if n_agents < 2:
            return

        plt.figure(figsize=(10, 8))
        agg_comms = comm_matrix.mean(axis=2)
        sns.heatmap(agg_comms, cmap="viridis", cbar_kws={"label": "Mean comm signal"})
        plt.xlabel("Time Step")
        plt.ylabel("Agent")
        plt.title("Communication Activity Over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_mutual_information_heatmap(self, mi_matrix: np.ndarray, filename: str = "mutual_information"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(mi_matrix, cmap="Reds", annot=True, fmt=".3f",
                    xticklabels=[f"A{i}" for i in range(mi_matrix.shape[0])],
                    yticklabels=[f"A{i}" for i in range(mi_matrix.shape[1])])
        plt.title("Mutual Information Between Agents")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_diversity(self, diversity_history: List[float], filename: str = "diversity"):
        plt.figure(figsize=(10, 6))
        plt.plot(diversity_history, "g-", linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.title("Population Diversity Over Generations")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_redundancy_vs_generations(self, redundancy_history: List[float],
                                        comm_entropy_history: List[float],
                                        filename: str = "redundancy_vs_entropy"):
        plt.figure(figsize=(10, 6))
        gens = np.arange(len(redundancy_history))
        plt.plot(gens, redundancy_history, "b-", label="Redundancy Ratio", linewidth=2)
        plt.plot(gens, comm_entropy_history, "r--", label="Comm Entropy", linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("Communication Redundancy vs Entropy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def plot_phase_transitions(self, transition_counts: List[int], filename: str = "phase_transitions"):
        plt.figure(figsize=(10, 6))
        plt.plot(transition_counts, "m-", linewidth=2, marker="o")
        plt.xlabel("Generation")
        plt.ylabel("Number of Phase Transitions")
        plt.title("Phase Transitions in Collective Behavior")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()

    def generate_summary_plot(self, archive, filename: str = "summary"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        gens = list(range(len(archive.generations)))

        ax = axes[0, 0]
        best = archive.get_fitness_history()
        mean = archive.get_mean_fitness_history()
        ax.plot(gens, best, "b-", label="Best")
        ax.plot(gens, mean, "r--", label="Mean")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(gens, archive.get_diversity_history(), "g-")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Diversity")
        ax.set_title("Population Diversity")

        ax = axes[0, 2]
        coord = archive.get_metrics("coordination_score_mean")
        if coord:
            ax.plot(gens[:len(coord)], coord, "c-")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Coordination")
        ax.set_title("Coordination Score")

        ax = axes[1, 0]
        mi = archive.get_metrics("mutual_information_mean")
        if mi:
            ax.plot(gens[:len(mi)], mi, "m-")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mutual Info")
        ax.set_title("Mutual Information")

        ax = axes[1, 1]
        comm = archive.get_metrics("communication_entropy_mean")
        if comm:
            ax.plot(gens[:len(comm)], comm, "orange")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Entropy")
        ax.set_title("Comm Entropy")

        ax = axes[1, 2]
        sr = archive.get_metrics("shared_representation_score")
        if sr:
            ax.plot(gens[:len(sr)], sr, "purple")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")
        ax.set_title("Shared Representation")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.{self.fmt}"), dpi=self.dpi)
        plt.close()
