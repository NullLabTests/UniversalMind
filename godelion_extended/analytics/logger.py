import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class ExperimentLogger:
    def __init__(self, output_dir: str = "./output", experiment_name: str = "experiment",
                 level: str = "INFO"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.run_dir = os.path.join(output_dir, f"{experiment_name}_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_csv: Optional[str] = None
        self._csv_writer = None
        self._csv_file = None
        self._metrics_data: List[Dict] = []

        self._setup_logging(level)
        self._setup_csv()

    def _setup_logging(self, level: str):
        log_file = os.path.join(self.run_dir, "experiment.log")
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.info(f"Experiment '{self.experiment_name}' started (run_id: {self.run_id})")
        self.logger.info(f"Output directory: {self.run_dir}")

    def _setup_csv(self):
        self.metrics_csv = os.path.join(self.run_dir, "metrics.csv")

    def log_metrics(self, generation: int, metrics: Dict):
        record = {"generation": generation, **metrics}
        self._metrics_data.append(record)

        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                writer.writeheader()
                writer.writerow(record)
        else:
            with open(self.metrics_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                writer.writerow(record)

        total_reward = metrics.get('total_system_reward_mean', metrics.get('mean_fitness', 0.0))
        coord = metrics.get('coordination_score_mean', 0.0)
        self.logger.info(f"Gen {generation}: total_reward={total_reward:.2f}, "
                         f"coord={coord:.4f}")

    def log_message(self, message: str, level: str = "INFO"):
        getattr(self.logger, level.lower(), self.logger.info)(message)

    def save_json(self, data: Any, filename: str):
        path = os.path.join(self.run_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        self.logger.debug(f"Saved {filename}")

    def save_config(self, config_dict: Dict):
        self.save_json(config_dict, "config.json")

    def get_run_dir(self) -> str:
        return self.run_dir

    def get_metrics_data(self) -> List[Dict]:
        return self._metrics_data

    def close(self):
        self.save_json(self._metrics_data, "all_metrics.json")
        self.logger.info(f"Experiment '{self.experiment_name}' completed.")
        logging.shutdown()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)
