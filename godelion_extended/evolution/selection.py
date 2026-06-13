from typing import Callable, List, Optional, Tuple

import numpy as np


def select_parents(
    fitness_scores: List[float],
    n_parents: int,
    method: str = "tournament",
    tournament_size: int = 3,
    rng: Optional[np.random.RandomState] = None,
    temperature: float = 1.0,
) -> List[int]:
    if rng is None:
        rng = np.random.RandomState()

    n_agents = len(fitness_scores)
    if n_agents == 0:
        return []

    n_parents = min(n_parents, n_agents)

    if method == "tournament":
        parents = []
        used = set()
        for _ in range(n_parents):
            candidates = [i for i in range(n_agents) if i not in used]
            if len(candidates) < tournament_size:
                candidates = list(range(n_agents))
            tournament = rng.choice(candidates, size=min(tournament_size, len(candidates)), replace=False)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            parents.append(winner)
            used.add(winner)
        return parents

    elif method == "proportional":
        min_fit = min(fitness_scores)
        adjusted = [f - min_fit + 1e-8 for f in fitness_scores]
        probs = np.array(adjusted) / sum(adjusted)
        probs = probs ** (1.0 / temperature)
        probs /= probs.sum()
        selected = rng.choice(n_agents, size=n_parents, replace=False, p=probs)
        return selected.tolist()

    elif method == "elite":
        sorted_idx = np.argsort(fitness_scores)[::-1]
        return sorted_idx[:n_parents].tolist()

    elif method == "rank":
        sorted_idx = np.argsort(fitness_scores)
        ranks = np.arange(1, n_agents + 1, dtype=float)
        probs = ranks / ranks.sum()
        selected = rng.choice(n_agents, size=n_parents, replace=False, p=probs[::-1])
        return selected.tolist()

    else:
        return rng.choice(n_agents, size=n_parents, replace=False).tolist()


def select_survivors(
    fitness_scores: List[float],
    n_survivors: int,
    method: str = "truncation",
    elite_ratio: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    if rng is None:
        rng = np.random.RandomState()

    n_agents = len(fitness_scores)
    n_survivors = min(n_survivors, n_agents)
    sorted_idx = np.argsort(fitness_scores)[::-1]

    if method == "truncation":
        return sorted_idx[:n_survivors].tolist()

    elif method == "elite_plus_random":
        n_elite = max(1, int(n_survivors * elite_ratio))
        elite = sorted_idx[:n_elite].tolist()
        remaining = [i for i in range(n_agents) if i not in elite]
        n_random = n_survivors - n_elite
        if n_random > 0 and remaining:
            random_choices = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False).tolist()
            return elite + random_choices
        return elite

    else:
        return sorted_idx[:n_survivors].tolist()
