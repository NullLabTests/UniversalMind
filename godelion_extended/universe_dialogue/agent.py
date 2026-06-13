from typing import Dict, List, Optional, Tuple

import numpy as np

from ..agents.policy import MLPPolicy, LinearPolicy, PolicyConfig
from ..agents.simple_policy import SimplePolicy
from ..agents.memory import AgentMemory
from ..agents.communication import create_channel, CommunicationChannel


class DialogueAgent:
    def __init__(self, agent_id: int, config: "UniverseDialogueConfig",
                 rng: Optional[np.random.RandomState] = None):
        self.id = agent_id
        self.config = config
        self.rng = rng or np.random.RandomState()

        obs_dim = 4
        self._memory_dim = obs_dim if config.use_memory else 0
        self._comm_dim = config.comm_dim if config.learn_communication else 0
        if config.learn_communication and config.channel_type == "discrete":
            self._comm_dim = config.message_length * config.vocab_size
        policy_input_dim = obs_dim + self._memory_dim + self._comm_dim

        policy_cfg = PolicyConfig(
            input_dim=policy_input_dim,
            hidden_dim=config.policy_hidden_dim,
            output_dim=4,
            policy_type=config.policy_type,
            seed=self.rng.randint(0, 2**31) if config.seed is None else config.seed + agent_id,
        )
        if config.policy_type == "simple":
            self.policy = SimplePolicy(policy_cfg)
        else:
            self.policy = MLPPolicy(policy_cfg)

        self.comm_channel: Optional[CommunicationChannel] = None
        if config.learn_communication:
            channel_config = {
                "comm_dim": config.comm_dim,
                "state_dim": 4,
                "vocab_size": config.vocab_size,
                "message_length": config.message_length,
                "attention_heads": config.attention_heads,
            }
            self.comm_channel = create_channel(
                config.channel_type, channel_config,
                rng=np.random.RandomState(self.rng.randint(0, 2**31)),
            )

        self.memory: Optional[AgentMemory] = None
        if config.use_memory:
            self.memory = AgentMemory(memory_size=config.memory_size, state_dim=obs_dim)

        self.fitness: float = 0.0
        self.total_reward: float = 0.0
        self.comm_state: np.ndarray = np.zeros(config.comm_dim)

    def act(self, obs: np.ndarray, comm_input: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
        x = np.tanh(obs)

        if self.memory is not None:
            self.memory.push(x)
            mem = self.memory.get_state()
            x = np.concatenate([x, mem])

        if self.comm_channel is not None:
            if comm_input is not None:
                ci = np.asarray(comm_input).flatten()
                if len(ci) < self._comm_dim:
                    ci = np.pad(ci, (0, self._comm_dim - len(ci)))
                elif len(ci) > self._comm_dim:
                    ci = ci[:self._comm_dim]
                x = np.concatenate([x, ci])
            else:
                x = np.concatenate([x, np.zeros(self._comm_dim)])

        logits = self.policy.forward(x)
        action = int(np.argmax(logits + self.rng.randn(4) * 0.1))

        if self.comm_channel is not None:
            self.comm_state = self.comm_channel.encode(obs)

        return action, self.comm_state

    def reset(self):
        self.total_reward = 0.0
        self.fitness = 0.0
        self.comm_state = np.zeros(self._comm_dim)
        if self.memory is not None:
            self.memory.reset()

    def clone(self) -> "DialogueAgent":
        clone = DialogueAgent.__new__(DialogueAgent)
        clone.id = self.id
        clone.config = self.config
        clone.rng = self.rng
        clone.policy = self.policy.clone()
        clone.comm_channel = self.comm_channel.clone() if self.comm_channel is not None else None
        clone.memory = self.memory.clone() if self.memory is not None else None
        clone.fitness = self.fitness
        clone.total_reward = self.total_reward
        clone.comm_state = self.comm_state.copy()
        clone._comm_dim = self._comm_dim
        clone._memory_dim = self._memory_dim
        return clone

    def mutate(self, rate: float = 0.1, strength: float = 0.05):
        self.policy.mutate(rate, strength, self.rng)
        if self.comm_channel is not None:
            self.comm_channel.mutate(rate, strength)

    def get_genome(self) -> Dict:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "policy": self.policy.get_weights(),
            "comm_weights": (
                [self.comm_channel.W_enc.copy(), self.comm_channel.W_dec.copy()]
                if self.comm_channel is not None else None
            ),
        }

    def __repr__(self):
        return f"DialogueAgent({self.id}, fitness={self.fitness:.4f})"
