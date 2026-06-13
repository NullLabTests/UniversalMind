from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvolutionStrategy:
    mutation_rate: float = 0.1
    mutation_strength: float = 0.05
    selection_method: str = "tournament"
    tournament_size: int = 3
    elite_ratio: float = 0.1
    crossover_rate: float = 0.3
    temperature: float = 1.0


class MetaLearner:
    def __init__(self, n_strategies: int = 5, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.strategies = [
            EvolutionStrategy(
                mutation_rate=self.rng.uniform(0.01, 0.5),
                mutation_strength=self.rng.uniform(0.005, 0.3),
                selection_method=self.rng.choice(["tournament", "proportional", "elite", "rank"]),
                tournament_size=self.rng.randint(2, 6),
                elite_ratio=self.rng.uniform(0.05, 0.3),
                crossover_rate=self.rng.uniform(0.0, 0.6),
                temperature=self.rng.uniform(0.5, 2.0),
            )
            for _ in range(n_strategies)
        ]
        self.strategy_fitness = [0.0] * n_strategies
        self.history: List[Dict] = []

    def select_strategy(self) -> EvolutionStrategy:
        probs = np.array([max(0.0, f) for f in self.strategy_fitness], dtype=float)
        if probs.sum() == 0:
            probs = np.ones(len(self.strategies)) / len(self.strategies)
        else:
            probs = probs / probs.sum()
        idx = self.rng.choice(len(self.strategies), p=probs)
        return self.strategies[idx]

    def update_strategy_fitness(self, strategy_idx: int, fitness_gain: float):
        self.strategy_fitness[strategy_idx] = 0.9 * self.strategy_fitness[strategy_idx] + 0.1 * fitness_gain

    def mutate_strategies(self):
        for i, strategy in enumerate(self.strategies):
            if self.rng.random() < 0.2:
                strategy.mutation_rate *= self.rng.uniform(0.5, 2.0)
                strategy.mutation_rate = np.clip(strategy.mutation_rate, 0.001, 0.5)
            if self.rng.random() < 0.2:
                strategy.mutation_strength *= self.rng.uniform(0.5, 2.0)
                strategy.mutation_strength = np.clip(strategy.mutation_strength, 0.001, 0.5)
            if self.rng.random() < 0.1:
                strategy.selection_method = self.rng.choice(["tournament", "proportional", "elite", "rank"])
            if self.rng.random() < 0.15:
                strategy.crossover_rate = np.clip(
                    strategy.crossover_rate + self.rng.uniform(-0.1, 0.1), 0.0, 0.8
                )

    def log_generation(self, gen: int, strategy_idx: int, fitness_gain: float, strategy: EvolutionStrategy):
        self.history.append({
            "generation": gen,
            "strategy_idx": strategy_idx,
            "fitness_gain": fitness_gain,
            "mutation_rate": strategy.mutation_rate,
            "mutation_strength": strategy.mutation_strength,
            "selection_method": strategy.selection_method,
            "crossover_rate": strategy.crossover_rate,
        })

    def best_strategy(self) -> EvolutionStrategy:
        best_idx = int(np.argmax(self.strategy_fitness))
        return self.strategies[best_idx]
