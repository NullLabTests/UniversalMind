from typing import Optional, Tuple

import numpy as np

from ..universe_dialogue.agent import DialogueAgent


def mutate_agent(agent: DialogueAgent, rate: float = 0.1, strength: float = 0.05,
                 rng: Optional[np.random.RandomState] = None) -> DialogueAgent:
    mutant = agent.clone()
    mutant.mutate(rate, strength)
    return mutant


def crossover_agents(parent_a: DialogueAgent, parent_b: DialogueAgent,
                     rng: Optional[np.random.RandomState] = None) -> DialogueAgent:
    if rng is None:
        rng = np.random.RandomState()

    child = parent_a.clone()
    child.policy = parent_a.policy.crossover(parent_b.policy, rng)

    if child.comm_channel is not None and parent_b.comm_channel is not None:
        if rng.random() < 0.5:
            child.comm_channel = parent_b.comm_channel.clone()

    if rng.random() < 0.1:
        child.mutate(rate=0.05, strength=0.02)

    return child


def adapt_mutation_rate(current_rate: float, fitness_trend: float,
                        min_rate: float = 0.01, max_rate: float = 0.5,
                        adaptation_strength: float = 0.1) -> float:
    if fitness_trend > 0:
        new_rate = current_rate * (1 - adaptation_strength * fitness_trend)
    else:
        new_rate = current_rate * (1 + adaptation_strength * abs(fitness_trend))

    return float(np.clip(new_rate, min_rate, max_rate))


def adapt_mutation_strength(current_strength: float, fitness_trend: float,
                            min_strength: float = 0.005, max_strength: float = 0.5,
                            adaptation_strength: float = 0.1) -> float:
    if fitness_trend > 0:
        new_strength = current_strength * (1 - adaptation_strength * fitness_trend)
    else:
        new_strength = current_strength * (1 + adaptation_strength * abs(fitness_trend))

    return float(np.clip(new_strength, min_strength, max_strength))
