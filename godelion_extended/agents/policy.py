from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PolicyConfig:
    input_dim: int = 4
    hidden_dim: int = 16
    output_dim: int = 4
    policy_type: str = "mlp"  # mlp | linear
    use_bias: bool = True
    seed: Optional[int] = None


class MLPPolicy:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        scale = np.sqrt(2.0 / config.input_dim)
        self.W1 = self.rng.randn(config.input_dim, config.hidden_dim) * scale
        self.b1 = np.zeros(config.hidden_dim) if config.use_bias else None

        scale2 = np.sqrt(2.0 / config.hidden_dim)
        self.W2 = self.rng.randn(config.hidden_dim, config.hidden_dim) * scale2
        self.b2 = np.zeros(config.hidden_dim) if config.use_bias else None

        scale3 = np.sqrt(2.0 / config.hidden_dim)
        self.W3 = self.rng.randn(config.hidden_dim, config.output_dim) * scale3
        self.b3 = np.zeros(config.output_dim) if config.use_bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + (self.b1 if self.b1 is not None else 0))
        h = np.tanh(h @ self.W2 + (self.b2 if self.b2 is not None else 0))
        logits = h @ self.W3 + (self.b3 if self.b3 is not None else 0)
        return logits

    def get_weights(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, weights: list):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights

    def clone(self) -> "MLPPolicy":
        clone = MLPPolicy(self.config)
        clone.W1 = self.W1.copy()
        clone.b1 = self.b1.copy() if self.b1 is not None else None
        clone.W2 = self.W2.copy()
        clone.b2 = self.b2.copy() if self.b2 is not None else None
        clone.W3 = self.W3.copy()
        clone.b3 = self.b3.copy() if self.b3 is not None else None
        return clone

    def mutate(self, rate: float = 0.1, strength: float = 0.05, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        for w in [self.W1, self.W2, self.W3]:
            mask = rng.random(w.shape) < rate
            w += mask * rng.randn(*w.shape) * strength
        for b in [self.b1, self.b2, self.b3]:
            if b is not None:
                mask = rng.random(b.shape) < rate
                b += mask * rng.randn(*b.shape) * strength

    def crossover(self, other: "MLPPolicy", rng: Optional[np.random.RandomState] = None) -> "MLPPolicy":
        if rng is None:
            rng = np.random.RandomState()
        child = self.clone()
        for w_self, w_other in zip([child.W1, child.W2, child.W3], [other.W1, other.W2, other.W3]):
            mask = rng.random(w_self.shape) < 0.5
            w_self[mask] = w_other[mask]
        for b_self, b_other in zip([child.b1, child.b2, child.b3], [other.b1, other.b2, other.b3]):
            if b_self is not None and b_other is not None:
                mask = rng.random(b_self.shape) < 0.5
                b_self[mask] = b_other[mask]
        return child

    def __repr__(self):
        return f"MLPPolicy({self.config.input_dim}->{self.config.hidden_dim}->{self.config.output_dim})"


class LinearPolicy:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        scale = np.sqrt(2.0 / config.input_dim)
        self.W = self.rng.randn(config.input_dim, config.output_dim) * scale
        self.b = np.zeros(config.output_dim) if config.use_bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + (self.b if self.b is not None else 0)

    def get_weights(self) -> list:
        return [self.W, self.b]

    def set_weights(self, weights: list):
        self.W, self.b = weights

    def clone(self) -> "LinearPolicy":
        clone = LinearPolicy(self.config)
        clone.W = self.W.copy()
        clone.b = self.b.copy() if self.b is not None else None
        return clone

    def mutate(self, rate: float = 0.1, strength: float = 0.05, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        mask = rng.random(self.W.shape) < rate
        self.W += mask * rng.randn(*self.W.shape) * strength
        if self.b is not None:
            mask = rng.random(self.b.shape) < rate
            self.b += mask * rng.randn(*self.b.shape) * strength

    def crossover(self, other: "LinearPolicy", rng: Optional[np.random.RandomState] = None) -> "LinearPolicy":
        if rng is None:
            rng = np.random.RandomState()
        child = self.clone()
        mask = rng.random(child.W.shape) < 0.5
        child.W[mask] = other.W[mask]
        if child.b is not None and other.b is not None:
            mask = rng.random(child.b.shape) < 0.5
            child.b[mask] = other.b[mask]
        return child

    def __repr__(self):
        return f"LinearPolicy({self.config.input_dim}->{self.config.output_dim})"
