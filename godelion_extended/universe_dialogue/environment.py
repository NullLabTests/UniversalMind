from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class UniverseDialogueConfig:
    n_agents: int = 6
    env_size: int = 10
    n_steps: int = 100
    n_generations: int = 50
    n_trials: int = 3

    # Agent
    policy_type: str = "mlp"  # mlp | simple
    policy_hidden_dim: int = 16
    comm_dim: int = 4
    memory_size: int = 8
    use_memory: bool = True
    learn_communication: bool = True

    # Communication
    channel_type: str = "vector"
    vocab_size: int = 16
    message_length: int = 4
    attention_heads: int = 2

    # Task
    task_type: str = "individual"
    shared_goal_prob: float = 0.3
    reward_type: str = "distance"

    # Misc
    seed: Optional[int] = None
