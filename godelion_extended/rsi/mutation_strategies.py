from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..evolution.mutation import mutate_agent, crossover_agents
from ..universe_dialogue.agent import DialogueAgent


class MutationStrategy(ABC):
    @abstractmethod
    def apply(self, agent: DialogueAgent, generation: int, fitness_trend: float) -> DialogueAgent:
        ...


class ConstantStrategy(MutationStrategy):
    def __init__(self, rate: float = 0.1, strength: float = 0.05):
        self.rate = rate
        self.strength = strength

    def apply(self, agent: DialogueAgent, generation: int, fitness_trend: float) -> DialogueAgent:
        return mutate_agent(agent, self.rate, self.strength)


class AdaptiveStrategy(MutationStrategy):
    def __init__(self, initial_rate: float = 0.1, initial_strength: float = 0.05,
                 min_rate: float = 0.01, max_rate: float = 0.5,
                 min_strength: float = 0.005, max_strength: float = 0.3):
        self.rate = initial_rate
        self.strength = initial_strength
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.adaptation_rate = 0.1

    def apply(self, agent: DialogueAgent, generation: int, fitness_trend: float) -> DialogueAgent:
        if fitness_trend > 0:
            self.rate *= (1 - self.adaptation_rate * min(fitness_trend, 1.0))
            self.strength *= (1 - self.adaptation_rate * min(fitness_trend, 1.0))
        else:
            self.rate *= (1 + self.adaptation_rate * min(abs(fitness_trend), 1.0))
            self.strength *= (1 + self.adaptation_rate * min(abs(fitness_trend), 1.0))

        self.rate = np.clip(self.rate, self.min_rate, self.max_rate)
        self.strength = np.clip(self.strength, self.min_strength, self.max_strength)

        return mutate_agent(agent, self.rate, self.strength)


class RandomStrategy(MutationStrategy):
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def apply(self, agent: DialogueAgent, generation: int, fitness_trend: float) -> DialogueAgent:
        rate = self.rng.uniform(0.01, 0.5)
        strength = self.rng.uniform(0.005, 0.3)
        return mutate_agent(agent, rate, strength)
