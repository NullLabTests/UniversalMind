from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GridWorldConfig:
    size: int = 10
    n_agents: int = 5
    shared_goal_prob: float = 0.3
    reward_type: str = "distance"  # distance | sparse | potential
    task_type: str = "individual"  # individual | shared | mixed
    max_steps: int = 100
    seed: Optional[int] = None


class GridWorld:
    def __init__(self, config: GridWorldConfig):
        self.config = config
        self.size = config.size
        self.n_agents = config.n_agents
        self.shared_goal_prob = config.shared_goal_prob
        self.reward_type = config.reward_type
        self.task_type = config.task_type
        self.max_steps = config.max_steps
        self.rng = np.random.RandomState(config.seed)

        self.positions: Dict[int, np.ndarray] = {}
        self.targets: Dict[int, np.ndarray] = {}
        self.shared_target: Optional[np.ndarray] = None
        self.step_count = 0

    def reset(self) -> Dict[int, np.ndarray]:
        self.step_count = 0
        self.positions = {}
        self.targets = {}

        self.shared_target = None
        if self.task_type in ("shared", "mixed"):
            self.shared_target = self.rng.randint(0, self.size, size=2).astype(float)

        for i in range(self.n_agents):
            self.positions[i] = self.rng.randint(0, self.size, size=2).astype(float)

            if self.task_type == "shared":
                self.targets[i] = self.shared_target.copy()
            elif self.task_type == "mixed" and self.rng.random() < self.shared_goal_prob:
                self.targets[i] = self.shared_target.copy()
            else:
                self.targets[i] = self.rng.randint(0, self.size, size=2).astype(float)

        return self._observe()

    def _observe(self) -> Dict[int, np.ndarray]:
        obs = {}
        for i in range(self.n_agents):
            obs[i] = np.concatenate([self.positions[i], self.targets[i]])
        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool]:
        self.step_count += 1
        rewards: Dict[int, float] = {}

        for i, act in actions.items():
            if act == 0:
                self.positions[i][1] += 1
            elif act == 1:
                self.positions[i][1] -= 1
            elif act == 2:
                self.positions[i][0] -= 1
            elif act == 3:
                self.positions[i][0] += 1

            self.positions[i] = np.clip(self.positions[i], 0, self.size - 1)

            if self.reward_type == "distance":
                dist = np.linalg.norm(self.positions[i] - self.targets[i])
                rewards[i] = -dist / (self.size * np.sqrt(2))
            elif self.reward_type == "shared":
                dists = [np.linalg.norm(self.positions[j] - self.targets[j]) for j in range(self.n_agents)]
                mean_dist = sum(dists) / len(dists)
                for j in range(self.n_agents):
                    rewards[j] = -mean_dist / (self.size * np.sqrt(2))
                break
            elif self.reward_type == "sparse":
                dist = np.linalg.norm(self.positions[i] - self.targets[i])
                rewards[i] = 1.0 if dist < 1.0 else -0.1
            elif self.reward_type == "potential":
                prev_dist = np.linalg.norm(
                    self.positions[i] - actions.get(f"_prev_{i}", self.positions[i])
                )
                new_dist = np.linalg.norm(self.positions[i] - self.targets[i])
                rewards[i] = (prev_dist - new_dist) / self.size

        done = self.step_count >= self.max_steps
        return self._observe(), rewards, done

    def get_global_state(self) -> np.ndarray:
        pos_list = [self.positions[i] for i in range(self.n_agents)]
        tgt_list = [self.targets[i] for i in range(self.n_agents)]
        return np.concatenate([np.concatenate(pos_list), np.concatenate(tgt_list)])

    def get_positions(self) -> Dict[int, np.ndarray]:
        return self.positions

    def get_targets(self) -> Dict[int, np.ndarray]:
        return self.targets
