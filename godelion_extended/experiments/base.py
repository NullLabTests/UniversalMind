from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..analytics.logger import ExperimentLogger
from ..analytics.visualization import ExperimentVisualizer
from ..analytics.reporting import ReportGenerator
from ..evolution.archive import Archive


class BaseExperiment(ABC):
    def __init__(self, config: Dict, output_dir: str = "./output"):
        self.config = config
        self.output_dir = output_dir
        self.seed = config.get("experiment", {}).get("seed", 42)

        if config.get("reproducibility", {}).get("deterministic", True):
            np.random.seed(self.seed)

        experiment_name = config.get("experiment", {}).get("name", "experiment")
        self.logger = ExperimentLogger(output_dir, experiment_name)
        self.visualizer = ExperimentVisualizer(
            self.logger.get_run_dir(),
            dpi=config.get("visualization", {}).get("dpi", 150),
            fmt=config.get("visualization", {}).get("plot_format", "png"),
        )
        self.reporter = ReportGenerator(self.logger.get_run_dir())
        self.archive = Archive(self.logger.get_run_dir())

        self.logger.save_config(config)
        self.generation_data: Dict[int, Dict] = {}

    @abstractmethod
    def setup(self):
        ...

    @abstractmethod
    def run_generation(self, generation: int) -> Dict:
        ...

    @abstractmethod
    def evaluate(self) -> Dict:
        ...

    def run(self):
        self.logger.log_message(f"Starting experiment: {self.__class__.__name__}")
        self.setup()

        n_generations = self._get_n_generations()

        for gen in range(n_generations):
            self.logger.log_message(f"--- Generation {gen} ---")
            metrics = self.run_generation(gen)
            self.generation_data[gen] = metrics

            plot_interval = self.config.get("visualization", {}).get("plot_interval", 5)
            if gen % plot_interval == 0 or gen == n_generations - 1:
                self._generate_plots()

            self.logger.log_metrics(gen, metrics)

        final_metrics = self.evaluate()
        self._finalize()

        return self.archive

    def _get_n_generations(self) -> int:
        return self.config.get(
            list(self.config.keys())[1], {}
        ).get("n_generations", 50)

    def _generate_plots(self):
        if not self.config.get("visualization", {}).get("enabled", True):
            return

        best = self.archive.get_fitness_history()
        mean = self.archive.get_mean_fitness_history()
        if best:
            self.visualizer.plot_fitness_over_time(best, mean)

        if self.generation_data:
            self.visualizer.plot_coordination_metrics(self.generation_data)

        diversity = self.archive.get_diversity_history()
        if diversity:
            self.visualizer.plot_diversity(diversity)

        redundancy = self.archive.get_metrics("redundancy_ratio")
        comm_entropy = self.archive.get_metrics("communication_entropy_mean")
        if redundancy and comm_entropy:
            self.visualizer.plot_redundancy_vs_generations(redundancy, comm_entropy)

        trans = self.archive.get_metrics("phase_transitions_n")
        if trans:
            self.visualizer.plot_phase_transitions(trans)

        self.visualizer.generate_summary_plot(self.archive)

    def _finalize(self):
        self.archive.save()
        report = self.reporter.generate_text_report(self.archive, self.config, self.generation_data)
        self.reporter.save_report(report)
        self.reporter.save_metrics_summary(self.generation_data)

        if self.config.get("visualization", {}).get("enabled", True):
            self._generate_plots()

        summary = self.reporter.generate_summary_stats(self.archive)
        self.logger.save_json(summary, "summary_stats.json")

        self.logger.log_message(f"Experiment complete. Results in: {self.logger.get_run_dir()}")
        self.logger.close()
