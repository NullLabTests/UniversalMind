import numpy as np
from typing import Optional


class AgentMemory:
    def __init__(self, memory_size: int = 8, state_dim: int = 4):
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.buffer = np.zeros((memory_size, state_dim))
        self.position = 0
        self.full = False

    def push(self, state: np.ndarray):
        self.buffer[self.position] = state[:self.state_dim]
        self.position = (self.position + 1) % self.memory_size
        if self.position == 0:
            self.full = True

    def get_state(self) -> np.ndarray:
        if self.full:
            return self.buffer.mean(axis=0)
        elif self.position > 0:
            return self.buffer[:self.position].mean(axis=0)
        return np.zeros(self.state_dim)

    def get_recent(self, n: int = 3) -> np.ndarray:
        if self.full:
            idx = np.arange(self.position - n, self.position) % self.memory_size
            return self.buffer[idx].flatten()
        elif self.position >= n:
            return self.buffer[self.position - n : self.position].flatten()
        return self.buffer[:self.position].flatten()

    def reset(self):
        self.buffer.fill(0)
        self.position = 0
        self.full = False

    def clone(self) -> "AgentMemory":
        clone = AgentMemory(self.memory_size, self.state_dim)
        clone.buffer = self.buffer.copy()
        clone.position = self.position
        clone.full = self.full
        return clone
