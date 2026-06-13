from typing import Dict, List, Optional, Tuple

import numpy as np

from .policy import PolicyConfig


class SimplePolicy:
    """Minimal evolvable policy with 3 interpretable parameters.

    Parameters (unconstrained floats, sigmoid-normalized at inference):
      [0] goal_bias:     how strongly to move toward goal (0..1)
      [1] comm_weight:   how much communication affects action (0..1-gb)
      [2] exploration:   random-action probability   (0..1-gb-cw)

    Action: blend of goal-direction, comm-direction, and random exploration,
    mapped to discrete {up, down, left, right}.
    """

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed if config.seed is not None else 0)
        self.params = self.rng.randn(3) * 0.5

    def forward(self, x: np.ndarray) -> np.ndarray:
        gb, cw, ex = self._normalized()

        pos = x[0:2]
        target = x[2:4]

        # Goal direction
        goal_vec = target - pos
        gn = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (gn + 1e-8) if gn > 0 else np.zeros(2)

        # Comm direction (from later part of input)
        comm_start = 4
        comm_vec = np.zeros(2)
        if self.config.input_dim > comm_start + 1:
            rest = x[comm_start:]
            comm_part = rest[:2]
            cn = np.linalg.norm(comm_part)
            if cn > 1e-8:
                comm_vec = comm_part / cn

        # Blend with exploration
        rand_dir = self.rng.uniform(-1, 1, 2)
        rn = np.linalg.norm(rand_dir)
        rand_dir = rand_dir / (rn + 1e-8) if rn > 0 else np.zeros(2)

        blend = goal_dir * gb + comm_vec * cw + rand_dir * (1.0 - gb - cw)
        bn = np.linalg.norm(blend)
        if bn > 0:
            blend = blend / bn

        logits = np.zeros(4)
        if blend[1] > 0.3:
            logits[0] = 1.0
        elif blend[1] < -0.3:
            logits[1] = 1.0
        if blend[0] < -0.3:
            logits[2] = 1.0
        elif blend[0] > 0.3:
            logits[3] = 1.0

        if ex > 0:
            logits += self.rng.randn(4) * ex

        return logits

    def mutate(self, rate: float = 0.1, strength: float = 0.05, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = self.rng
        for i in range(len(self.params)):
            if rng.random() < rate:
                self.params[i] += rng.randn() * strength

    def crossover(self, other: "SimplePolicy", rng: Optional[np.random.RandomState] = None) -> "SimplePolicy":
        if rng is None:
            rng = np.random.RandomState()
        child = SimplePolicy(self.config)
        for i in range(len(self.params)):
            child.params[i] = self.params[i] if rng.random() < 0.5 else other.params[i]
        return child

    def clone(self) -> "SimplePolicy":
        child = SimplePolicy(self.config)
        child.params = self.params.copy()
        return child

    def get_weights(self) -> Dict:
        gb, cw, ex = self._normalized()
        return {"goal_bias": float(gb), "comm_weight": float(cw), "exploration": float(ex)}

    def _normalized(self) -> Tuple[float, float, float]:
        gb = float(1.0 / (1.0 + np.exp(-self.params[0])))
        cw = float(1.0 / (1.0 + np.exp(-self.params[1]))) * (1.0 - gb)
        ex = float(1.0 / (1.0 + np.exp(-self.params[2]))) * (1.0 - gb - cw)
        return gb, cw, ex
