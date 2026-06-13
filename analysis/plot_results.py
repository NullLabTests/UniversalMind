#!/usr/bin/env python3
"""
Analysis script: plot results from a completed experiment run.

Usage:
    python -m analysis.plot_results --run-dir ./output/universe_dialogue_20250101_120000
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_metrics(run_dir: str):
    metrics_path = os.path.join(run_dir, "all_metrics.json")
    if not os.path.exists(metrics_path):
        alt_path = os.path.join(run_dir, "metrics_summary.json")
        if os.path.exists(alt_path):
            with open(alt_path) as f:
                return json.load(f)
        print(f"No metrics found in {run_dir}")
        return None

    with open(metrics_path) as f:
        data = json.load(f)
    return data


def plot_fitness(metrics, run_dir: str):
    gens = [m.get("generation", i) for i, m in enumerate(metrics)]
    best = [m.get("best_fitness", m.get("total_system_reward_mean", 0)) for m in metrics]
    mean_vals = [m.get("mean_fitness", m.get("total_system_reward_mean", 0)) for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, best, "b-", label="Best", linewidth=2)
    plt.plot(gens, mean_vals, "r--", label="Mean", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots", "fitness_over_time.png"), dpi=150)
    plt.close()


def plot_coordination(metrics, run_dir: str):
    gens = [m.get("generation", i) for i, m in enumerate(metrics)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (key, title, color) in zip(
        axes.flat,
        [
            ("coordination_score_mean", "Coordination Score", "blue"),
            ("synchronization_index_mean", "Sync Index", "green"),
            ("mutual_information_mean", "Mutual Information", "red"),
            ("shared_representation_score", "Shared Representation", "purple"),
        ],
    ):
        values = [m.get(key, 0) or 0 for m in metrics]
        ax.plot(gens[:len(values)], values, color=color, linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots", "coordination_metrics.png"), dpi=150)
    plt.close()


def plot_communication(metrics, run_dir: str):
    gens = [m.get("generation", i) for i, m in enumerate(metrics)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    comm_entropy = [m.get("communication_entropy_mean", 0) or 0 for m in metrics]
    axes[0].plot(gens[:len(comm_entropy)], comm_entropy, "purple", linewidth=2)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Entropy")
    axes[0].set_title("Communication Entropy")

    redundancy = [m.get("redundancy_ratio", 0) or 0 for m in metrics]
    axes[1].plot(gens[:len(redundancy)], redundancy, "brown", linewidth=2)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Redundancy Ratio")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots", "communication_analysis.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--run-dir", type=str, required=True, help="Experiment run directory")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.exists(run_dir):
        print(f"Directory not found: {run_dir}")
        sys.exit(1)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics = load_metrics(run_dir)
    if metrics is None:
        return

    if isinstance(metrics, dict):
        metrics = [v for v in metrics.values() if isinstance(v, dict)]

    if metrics:
        plot_fitness(metrics, run_dir)
        plot_coordination(metrics, run_dir)
        plot_communication(metrics, run_dir)
        print(f"Plots saved to {plots_dir}")
    else:
        print("No metrics to plot")


if __name__ == "__main__":
    main()
