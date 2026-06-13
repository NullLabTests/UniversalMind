import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_text_report(self, archive, config: Dict, metrics_history: Dict[int, Dict]) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("UNIVERSAL MIND — EXPERIMENT REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Config: {json.dumps(config, indent=2)}")
        lines.append("")

        n_gens = len(archive.generations)
        lines.append(f"Total Generations: {n_gens}")
        if n_gens > 0:
            lines.append(f"Best Fitness: {archive.best_fitness_ever:.4f}")
            lines.append(f"Final Best Fitness: {archive.generations[-1].best_fitness:.4f}")
            lines.append(f"Final Mean Fitness: {archive.generations[-1].mean_fitness:.4f}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("HYPOTHESIS TESTING")
        lines.append("-" * 70)
        lines.append("")

        lines.append("H1: All coordination is reducible to local interaction + communication")
        lines.append("H2: Evolution produces emergent global structure resembling a unified optimization process")
        lines.append("")

        if n_gens >= 2:
            coord_values = archive.get_metrics("coordination_score_mean")
            mi_values = archive.get_metrics("mutual_information_mean")
            sr_values = archive.get_metrics("shared_representation_score")

            if len(coord_values) >= 2:
                coord_trend = coord_values[-1] - coord_values[0]
                lines.append(f"Coordination score trend: {coord_trend:+.4f} (start={coord_values[0]:.4f}, end={coord_values[-1]:.4f})")
            if len(mi_values) >= 2:
                mi_trend = mi_values[-1] - mi_values[0]
                lines.append(f"Mutual information trend: {mi_trend:+.4f} (start={mi_values[0]:.4f}, end={mi_values[-1]:.4f})")
            if len(sr_values) >= 2:
                sr_trend = sr_values[-1] - sr_values[0]
                lines.append(f"Shared representation trend: {sr_trend:+.4f} (start={sr_values[0]:.4f}, end={sr_values[-1]:.4f})")

            lines.append("")
            if coord_values and len(coord_values) >= 2:
                increasing_coord = coord_values[-1] > coord_values[0] + 0.05
                if increasing_coord:
                    lines.append("→ Coordination increased significantly over generations.")
                    lines.append("  This suggests emergent global structure beyond local communication.")
                else:
                    lines.append("→ Coordination remained stable or decreased.")
                    lines.append("  This is consistent with local-interaction reducibility (H1).")

        lines.append("")
        lines.append("-" * 70)
        lines.append("KEY METRICS SUMMARY")
        lines.append("-" * 70)

        key_metrics = [
            "total_system_reward_mean", "coordination_score_mean", "synchronization_index_mean",
            "mutual_information_mean", "communication_entropy_mean", "shared_representation_score",
            "cross_prediction_error", "redundancy_ratio",
        ]
        for metric in key_metrics:
            values = archive.get_metrics(metric)
            if values:
                lines.append(f"  {metric}: start={values[0]:.4f}, end={values[-1]:.4f}, "
                             f"min={min(values):.4f}, max={max(values):.4f}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_report(self, report: str, filename: str = "experiment_report.txt"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(report)

    def save_metrics_summary(self, metrics_history: Dict[int, Dict], filename: str = "metrics_summary.json"):
        path = os.path.join(self.output_dir, filename)
        serializable = {}
        for gen, metrics in metrics_history.items():
            serializable[str(gen)] = _make_serializable(metrics)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def generate_summary_stats(self, archive) -> Dict:
        if not archive.generations:
            return {}

        stats = {
            "n_generations": len(archive.generations),
            "best_fitness_ever": archive.best_fitness_ever,
            "final_best_fitness": archive.generations[-1].best_fitness,
            "final_mean_fitness": archive.generations[-1].mean_fitness,
        }

        for metric in ["coordination_score_mean", "mutual_information_mean",
                        "communication_entropy_mean", "shared_representation_score"]:
            values = archive.get_metrics(metric)
            if values:
                stats[f"{metric}_start"] = values[0]
                stats[f"{metric}_end"] = values[-1]
                stats[f"{metric}_trend"] = values[-1] - values[0]

        return stats


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj
