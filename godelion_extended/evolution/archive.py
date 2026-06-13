import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GenerationRecord:
    generation: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    agent_fitness: List[float] = field(default_factory=list)
    best_fitness: float = -float("inf")
    mean_fitness: float = -float("inf")
    diversity: float = 0.0
    mutation_rate: float = 0.1
    n_agents: int = 0
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class Archive:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.generations: List[GenerationRecord] = []
        self.best_agent_genome: Optional[Dict] = None
        self.best_fitness_ever: float = -float("inf")
        self._agent_genomes: Dict[int, List[Dict]] = {}

        os.makedirs(output_dir, exist_ok=True)

    def add_generation(self, record: GenerationRecord):
        self.generations.append(record)
        if record.best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = record.best_fitness

    def save_agent_genome(self, generation: int, genome: Dict):
        if generation not in self._agent_genomes:
            self._agent_genomes[generation] = []
        self._agent_genomes[generation].append(genome)

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.output_dir, "archive.json")

        data = {
            "best_fitness_ever": self.best_fitness_ever,
            "n_generations": len(self.generations),
            "generations": [g.to_dict() for g in self.generations],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)

        self.generations = [GenerationRecord(**g) for g in data["generations"]]
        self.best_fitness_ever = data["best_fitness_ever"]
        return self

    def get_fitness_history(self) -> List[float]:
        return [g.best_fitness for g in self.generations]

    def get_mean_fitness_history(self) -> List[float]:
        return [g.mean_fitness for g in self.generations]

    def get_diversity_history(self) -> List[float]:
        return [g.diversity for g in self.generations]

    def get_metrics(self, key: str) -> List[float]:
        return [g.metrics.get(key, 0.0) for g in self.generations]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)
