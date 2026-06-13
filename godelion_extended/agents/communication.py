from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class CommunicationChannel(ABC):
    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        self.dim = dim
        self.rng = rng or np.random.RandomState()

    @abstractmethod
    def encode(self, agent_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, message: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        ...

    @abstractmethod
    def mutate(self, rate: float = 0.1, strength: float = 0.05):
        ...

    @abstractmethod
    def clone(self) -> "CommunicationChannel":
        ...


class VectorChannel(CommunicationChannel):
    def __init__(self, dim: int = 4, state_dim: int = 4, rng: Optional[np.random.RandomState] = None):
        super().__init__(dim, rng)
        self.state_dim = state_dim
        scale = np.sqrt(2.0 / state_dim)
        self.W_enc = self.rng.randn(state_dim, dim) * scale
        self.W_dec = self.rng.randn(dim, state_dim) * scale
        self._seed_unused = None  # placeholder for compatibility

    def encode(self, agent_state: np.ndarray) -> np.ndarray:
        return np.tanh(agent_state[:self.state_dim] @ self.W_enc)

    def decode(self, message: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        decoded = message @ self.W_dec
        if context is not None:
            decoded = 0.7 * decoded + 0.3 * context[:self.state_dim]
        return np.tanh(decoded)

    def mutate(self, rate: float = 0.1, strength: float = 0.05):
        for w in [self.W_enc, self.W_dec]:
            mask = self.rng.random(w.shape) < rate
            w += mask * self.rng.randn(*w.shape) * strength

    def clone(self) -> "VectorChannel":
        clone = VectorChannel(self.dim, self.state_dim, self.rng)
        clone.W_enc = self.W_enc.copy()
        clone.W_dec = self.W_dec.copy()
        return clone


class DiscreteChannel(CommunicationChannel):
    def __init__(self, vocab_size: int = 16, message_length: int = 4, state_dim: int = 4,
                 rng: Optional[np.random.RandomState] = None):
        super().__init__(message_length, rng)
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.state_dim = state_dim

        scale = np.sqrt(2.0 / state_dim)
        self.W_enc = self.rng.randn(state_dim, message_length * vocab_size) * scale
        scale2 = np.sqrt(2.0 / (message_length * vocab_size))
        self.W_dec = self.rng.randn(message_length * vocab_size, state_dim) * scale2

    def encode(self, agent_state: np.ndarray) -> np.ndarray:
        logits = agent_state[:self.state_dim] @ self.W_enc
        logits = logits.reshape(self.message_length, self.vocab_size)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        symbols = np.array([self.rng.choice(self.vocab_size, p=probs[i]) for i in range(self.message_length)])
        one_hot = np.zeros((self.message_length, self.vocab_size))
        one_hot[np.arange(self.message_length), symbols] = 1
        return one_hot.flatten()

    def decode(self, message: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        decoded = message @ self.W_dec
        if context is not None:
            decoded = 0.7 * decoded + 0.3 * context[:self.state_dim]
        return np.tanh(decoded)

    def mutate(self, rate: float = 0.1, strength: float = 0.05):
        for w in [self.W_enc, self.W_dec]:
            mask = self.rng.random(w.shape) < rate
            w += mask * self.rng.randn(*w.shape) * strength

    def clone(self) -> "DiscreteChannel":
        clone = DiscreteChannel(self.vocab_size, self.message_length, self.state_dim, self.rng)
        clone.W_enc = self.W_enc.copy()
        clone.W_dec = self.W_dec.copy()
        return clone


class AttentionChannel(CommunicationChannel):
    def __init__(self, dim: int = 4, n_heads: int = 2, state_dim: int = 4,
                 rng: Optional[np.random.RandomState] = None):
        super().__init__(dim, rng)
        self.n_heads = n_heads
        self.state_dim = state_dim
        head_dim = dim // n_heads
        self.head_dim = max(1, head_dim)

        scale = np.sqrt(2.0 / state_dim)
        self.W_q = self.rng.randn(state_dim, dim) * scale
        self.W_k = self.rng.randn(state_dim, dim) * scale
        self.W_v = self.rng.randn(state_dim, dim) * scale
        self.W_o = self.rng.randn(dim, state_dim) * scale

    def encode(self, agent_state: np.ndarray) -> np.ndarray:
        x = agent_state[:self.state_dim]
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        q_heads = q.reshape(self.n_heads, self.head_dim)
        k_heads = k.reshape(self.n_heads, self.head_dim)
        v_heads = v.reshape(self.n_heads, self.head_dim)

        scores = q_heads @ k_heads.T / np.sqrt(self.head_dim)
        attn = np.exp(scores - scores.max())
        attn /= attn.sum(axis=1, keepdims=True)

        out_heads = attn @ v_heads
        out = out_heads.reshape(-1)
        return np.tanh(out)

    def decode(self, message: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        decoded = message @ self.W_o
        if context is not None:
            decoded = 0.7 * decoded + 0.3 * context[:self.state_dim]
        return np.tanh(decoded)

    def mutate(self, rate: float = 0.1, strength: float = 0.05):
        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            mask = self.rng.random(w.shape) < rate
            w += mask * self.rng.randn(*w.shape) * strength

    def clone(self) -> "AttentionChannel":
        clone = AttentionChannel(self.dim, self.n_heads, self.state_dim, self.rng)
        clone.W_q = self.W_q.copy()
        clone.W_k = self.W_k.copy()
        clone.W_v = self.W_v.copy()
        clone.W_o = self.W_o.copy()
        return clone


def create_channel(channel_type: str, config: dict, rng: np.random.RandomState) -> CommunicationChannel:
    if channel_type == "vector":
        return VectorChannel(
            dim=config.get("comm_dim", 4),
            state_dim=config.get("state_dim", 4),
            rng=rng,
        )
    elif channel_type == "discrete":
        return DiscreteChannel(
            vocab_size=config.get("vocab_size", 16),
            message_length=config.get("message_length", 4),
            state_dim=config.get("state_dim", 4),
            rng=rng,
        )
    elif channel_type == "attention":
        return AttentionChannel(
            dim=config.get("comm_dim", 4),
            n_heads=config.get("attention_heads", 2),
            state_dim=config.get("state_dim", 4),
            rng=rng,
        )
    raise ValueError(f"Unknown channel type: {channel_type}")
