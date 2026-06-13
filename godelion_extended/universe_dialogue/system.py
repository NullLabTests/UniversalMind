from typing import Dict, List, Optional, Tuple

import numpy as np

from .environment import UniverseDialogueConfig
from .agent import DialogueAgent
from ..environments.grid_world import GridWorld, GridWorldConfig


class MultiAgentSystem:
    def __init__(self, config: UniverseDialogueConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        self.env = GridWorld(GridWorldConfig(
            size=config.env_size,
            n_agents=config.n_agents,
            shared_goal_prob=config.shared_goal_prob,
            reward_type=config.reward_type,
            task_type=config.task_type,
            max_steps=config.n_steps,
            seed=(config.seed + 1 if config.seed is not None else None),
        ))

        self.agents: List[DialogueAgent] = [
            DialogueAgent(
                i, config,
                rng=np.random.RandomState(
                    (config.seed + 100 + i) if config.seed is not None else None
                ),
            )
            for i in range(config.n_agents)
        ]

    def run_episode(self) -> Tuple[float, List[Dict], Dict]:
        obs = self.env.reset()
        for agent in self.agents:
            agent.reset()

        total_reward = 0.0
        comm_history: List[Dict[int, np.ndarray]] = []
        action_history: List[Dict[int, int]] = []
        reward_history: List[Dict[int, float]] = []

        for step in range(self.config.n_steps):
            actions: Dict[int, int] = {}
            comms: Dict[int, np.ndarray] = {}

            for i, agent in enumerate(self.agents):
                comm_input = None
                if agent.comm_channel is not None and step > 0:
                    other_comms = [
                        self.agents[j].comm_state
                        for j in range(self.config.n_agents)
                        if j != i
                    ]
                    if other_comms:
                        avg_comm = np.mean(other_comms, axis=0)
                        comm_input = agent.comm_channel.decode(avg_comm, obs[i])

                action, comm = agent.act(obs[i], comm_input)
                actions[i] = action
                comms[i] = comm

            comm_history.append({k: v.copy() for k, v in comms.items()})
            action_history.append(dict(actions))

            obs, rewards, done = self.env.step(actions)
            reward_history.append(dict(rewards))

            for i, agent in enumerate(self.agents):
                agent.total_reward += rewards[i]

            total_reward += sum(rewards.values())

            if done:
                break

        for agent in self.agents:
            agent.fitness = agent.total_reward

        metadata = {
            "total_reward": total_reward,
            "n_steps": len(comm_history),
            "final_positions": {i: self.env.positions[i].copy() for i in range(self.config.n_agents)},
            "final_targets": {i: self.env.targets[i].copy() for i in range(self.config.n_agents)},
        }

        return total_reward, comm_history, action_history, reward_history, metadata

    def get_comm_matrix(self, comm_history: List[Dict]) -> np.ndarray:
        n_agents = self.config.n_agents
        n_steps = len(comm_history)
        comm_dim = self.config.comm_dim
        matrix = np.zeros((n_agents, n_steps, comm_dim))
        for t, comm_dict in enumerate(comm_history):
            for i in range(n_agents):
                matrix[i, t] = comm_dict[i][:comm_dim]
        return matrix

    def get_action_matrix(self, action_history: List[Dict]) -> np.ndarray:
        n_agents = self.config.n_agents
        n_steps = len(action_history)
        matrix = np.zeros((n_agents, n_steps), dtype=int)
        for t, action_dict in enumerate(action_history):
            for i in range(n_agents):
                matrix[i, t] = action_dict[i]
        return matrix

    def get_reward_matrix(self, reward_history: List[Dict]) -> np.ndarray:
        n_agents = self.config.n_agents
        n_steps = len(reward_history)
        matrix = np.zeros((n_agents, n_steps))
        for t, reward_dict in enumerate(reward_history):
            for i in range(n_agents):
                matrix[i, t] = reward_dict[i]
        return matrix
