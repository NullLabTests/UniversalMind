#!/usr/bin/env python3
"""
Generate a comprehensive experiment report from a completed run.

Usage:
    python -m analysis.generate_report --run-dir ./output/universe_dialogue_20250101_120000
    python -m analysis.generate_report --run-dir ./output/universe_dialogue_20250101_120000 --html
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


def load_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "all_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    metrics_path = os.path.join(run_dir, "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            return [v for v in data.values() if isinstance(v, dict)]
    return []


def load_archive(run_dir):
    archive_path = os.path.join(run_dir, "archive.json")
    if os.path.exists(archive_path):
        with open(archive_path) as f:
            return json.load(f)
    return {}


def extract_metric(metrics, key, default=0):
    values = []
    for m in metrics:
        v = m.get(key, m.get(key.replace("_mean", ""), default))
        if isinstance(v, (int, float)):
            values.append(v)
    return values


def generate_report(run_dir, output_path=None):
    metrics = load_metrics(run_dir)
    archive = load_archive(run_dir)

    if not metrics:
        print(f"No metrics found in {run_dir}")
        return

    if output_path is None:
        output_path = os.path.join(run_dir, "report.txt")

    n_gens = len(metrics)
    generations = [m.get("generation", i) for i, m in enumerate(metrics)]

    best_fitness = extract_metric(metrics, "best_fitness", "total_system_reward_mean")
    mean_fitness = extract_metric(metrics, "mean_fitness", "total_system_reward_mean")
    coord = extract_metric(metrics, "coordination_score_mean")
    mi = extract_metric(metrics, "mutual_information_mean")
    comm_entropy = extract_metric(metrics, "communication_entropy_mean")
    sync = extract_metric(metrics, "synchronization_index_mean")
    shared_rep = extract_metric(metrics, "shared_representation_score")
    cross_pred = extract_metric(metrics, "cross_prediction_error")
    diversity = extract_metric(metrics, "diversity")

    lines = []
    lines.append("=" * 72)
    lines.append("  UNIVERSAL MIND — φ-AnomalyCo Experiment Report")
    lines.append("=" * 72)
    lines.append(f"  Run:      {os.path.basename(run_dir)}")
    lines.append(f"  Generated: {datetime.now().isoformat()}")
    lines.append(f"  Generations: {n_gens}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  FITNESS")
    lines.append("-" * 72)
    if best_fitness:
        lines.append(f"  Best:        {max(best_fitness):.4f}  (final: {best_fitness[-1]:.4f})")
    if mean_fitness:
        lines.append(f"  Mean:        {mean_fitness[-1]:.4f}  (start: {mean_fitness[0]:.4f})")
    if len(best_fitness) > 1:
        trend = "↑" if best_fitness[-1] > best_fitness[0] else "↓"
        lines.append(f"  Trend:       {trend}  ({best_fitness[-1] - best_fitness[0]:+.4f})")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  COORDINATION METRICS")
    lines.append("-" * 72)
    if coord:
        lines.append(f"  Coordination:       {coord[-1]:.4f}  (start: {coord[0]:.4f}, "
                     f"trend: {coord[-1] - coord[0]:+.4f})")
    if sync:
        lines.append(f"  Sync Index:         {sync[-1]:.4f}  (start: {sync[0]:.4f}, "
                     f"trend: {sync[-1] - sync[0]:+.4f})")
    if mi:
        lines.append(f"  Mutual Info:        {mi[-1]:.4f}  (start: {mi[0]:.4f}, "
                     f"trend: {mi[-1] - mi[0]:+.4f})")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  EMERGENCE INDICATORS")
    lines.append("-" * 72)
    if shared_rep:
        lines.append(f"  Shared Rep:         {shared_rep[-1]:.4f}  (start: {shared_rep[0]:.4f}, "
                     f"trend: {shared_rep[-1] - shared_rep[0]:+.4f})")
    if cross_pred:
        lines.append(f"  Cross-Pred Error:   {cross_pred[-1]:.4f}  (start: {cross_pred[0]:.4f}, "
                     f"trend: {cross_pred[-1] - cross_pred[0]:+.4f})")
    if comm_entropy:
        lines.append(f"  Comm Entropy:       {comm_entropy[-1]:.4f}  (start: {comm_entropy[0]:.4f}, "
                     f"trend: {comm_entropy[-1] - comm_entropy[0]:+.4f})")
    if diversity:
        lines.append(f"  Diversity:          {diversity[-1]:.4f}  (start: {diversity[0]:.4f}, "
                     f"trend: {diversity[-1] - diversity[0]:+.4f})")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  HYPOTHESIS TEST")
    lines.append("-" * 72)
    lines.append("")
    lines.append("  H1: All coordination is reducible to local interaction + communication")
    lines.append("  H2: Evolution produces emergent global structure (φ-AnomalyCo)")
    lines.append("")

    if len(coord) > 1:
        coord_increasing = coord[-1] > coord[0] + 0.03
        mi_increasing = len(mi) > 1 and mi[-1] > mi[0] + 0.03
        lines.append(f"  Coordination trend:       {'↑ INCREASING' if coord_increasing else '→ STABLE/DECREASING'}")
        lines.append(f"  Mutual info trend:        {'↑ INCREASING' if mi_increasing else '→ STABLE/DECREASING'}")
        lines.append("")
        if coord_increasing or mi_increasing:
            lines.append("  → Evidence consistent with φ-AnomalyCo (H2).")
            lines.append("    Coordination exceeds baseline — structure may be emergent.")
        else:
            lines.append("  → Evidence consistent with local reducibility (H1).")
            lines.append("    No excess coordination detected beyond communication bounds.")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--run-dir", type=str, required=True, help="Experiment run directory")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    if not os.path.exists(args.run_dir):
        print(f"Directory not found: {args.run_dir}")
        sys.exit(1)

    generate_report(args.run_dir, args.output)


if __name__ == "__main__":
    main()
