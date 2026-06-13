"""Minimal coordination experiment: 2 blind agents on a 2x2 grid.

Each agent sees only its position (not the goal). The shared goal is at a
random corner. Reward is shared (-mean distance).  The ONLY way to navigate to
the goal is through the communication channel.  This forces evolution to
discover coordinated communication or fail.
"""

from typing import Dict, List, Optional

import numpy as np

from .base import BaseExperiment
from ..universe_dialogue import (
    UniverseDialogueConfig,
    MultiAgentSystem,
    UniverseDialogueAnalyzer,
)
from ..universe_dialogue.environment import UniverseDialogueConfig as UDConfig
from ..environments.grid_world import GridWorld, GridWorldConfig
from ..evolution.selection import select_parents
from ..evolution.mutation import mutate_agent, crossover_agents
from ..evolution.archive import GenerationRecord
from ..agents.simple_policy import SimplePolicy
from ..agents.communication import VectorChannel


class MinimalCoordinationExperiment(BaseExperiment):
    def __init__(self, config: Dict, output_dir: str = "./output"):
        super().__init__(config, output_dir)
        self.n_systems = 20
        self.n_generations = 300
        self.elite_ratio = 0.15
        self.crossover_rate = 0.3
        self.mutation_rate = 0.4
        self.mutation_strength = 0.3

        self.ud_kwargs = dict(
            n_agents=2,
            env_size=2,
            n_steps=20,
            n_trials=5,
            policy_type="simple",
            policy_hidden_dim=4,
            comm_dim=2,
            memory_size=2,
            use_memory=False,
            learn_communication=True,
            channel_type="vector",
            vocab_size=4,
            message_length=2,
            task_type="shared",
            reward_type="shared",
            partitioned=False,
            blind_ratio=0.0,
        )

    def setup(self):
        self.analyzer = UniverseDialogueAnalyzer(
            UniverseDialogueConfig(**self.ud_kwargs, n_generations=1, seed=self.seed + 200)
        )
        self.systems: List[MinimalSystem] = []
        base_seed = self.seed if self.seed is not None else 42
        for i in range(self.n_systems):
            self.systems.append(MinimalSystem(base_seed + 1000 + i))
        self.logger.log_message(
            f"MinimalCoordination: {self.n_systems} systems, "
            f"2 blind agents, 2x2 grid, {self.n_generations} generations"
        )

    def run_generation(self, generation: int) -> Dict:
        current_rate = self.mutation_rate * (0.995 ** generation)
        current_strength = self.mutation_strength * (0.995 ** generation)

        all_fitness: List[float] = []
        all_metrics: List[Dict] = []

        for system in self.systems:
            trial_results = []
            trial_total_rewards = []
            for _ in range(self.ud_kwargs["n_trials"]):
                total_reward, comm_history, action_history, reward_history, metadata = (
                    system.run_episode()
                )
                metrics = self.analyzer.analyze_episode(
                    system, comm_history, action_history, reward_history
                )
                trial_results.append(metrics)
                trial_total_rewards.append(total_reward)

            base_fitness = float(np.mean(trial_total_rewards))
            coord = float(np.mean([m.get("coordination_score_mean", 0) for m in trial_results]))

            sorted_idx = np.argsort(trial_total_rewards)
            median_reward = float(np.median(trial_total_rewards))
            worst_reward = float(np.min(trial_total_rewards))
            coord_bonus = coord * abs(base_fitness) * 0.5

            system_fitness = median_reward + coord_bonus

            all_fitness.append(system_fitness)
            combined = self.analyzer.compute_generation_metrics(trial_results)
            combined["system_fitness"] = system_fitness
            combined["base_reward"] = base_fitness
            combined["coord"] = coord
            combined["worst_reward"] = worst_reward
            all_metrics.append(combined)

        n_survivors = max(1, int(self.n_systems * self.elite_ratio))
        sorted_idx = np.argsort(all_fitness)[::-1]
        survivor_indices = sorted_idx[:n_survivors].tolist()

        n_children = self.n_systems - n_survivors
        new_systems: List[MinimalSystem] = []
        if n_children > 0:
            parent_indices = select_parents(
                [all_fitness[i] for i in survivor_indices],
                n_children, method="tournament", tournament_size=3,
            )
            parent_indices = [survivor_indices[i] for i in parent_indices]
            for idx in parent_indices:
                other = [i for i in parent_indices if i != idx]
                if self.crossover_rate > 0 and len(other) > 0 and np.random.random() < self.crossover_rate:
                    other_idx = int(np.random.choice(other))
                    child = self._crossover(self.systems[idx], self.systems[other_idx])
                else:
                    child = self._mutate(self.systems[idx], current_rate, current_strength)
                new_systems.append(child)

        self.systems = [self.systems[i] for i in survivor_indices] + new_systems

        gen_metrics: Dict[str, float] = {}
        for key in [
            "coordination_score_mean", "synchronization_index_mean",
            "mutual_information_mean", "communication_entropy_mean",
            "shared_representation_score",
        ]:
            vals = [m.get(key, 0) for m in all_metrics if key in m]
            gen_metrics[key] = float(np.mean(vals)) if vals else 0.0

        gen_metrics["mean_fitness"] = float(np.mean(all_fitness))
        gen_metrics["best_fitness"] = float(max(all_fitness))
        gen_metrics["fitness_std"] = float(np.std(all_fitness))
        gen_metrics["diversity"] = float(np.std(all_fitness) / (abs(np.mean(all_fitness)) + 1e-8))
        gen_metrics["coord_max"] = float(np.max([m.get("coord", 0) for m in all_metrics]))
        gen_metrics["mutation_rate"] = current_rate

        record = GenerationRecord(
            generation=generation, metrics=gen_metrics,
            agent_fitness=all_fitness, best_fitness=float(max(all_fitness)),
            mean_fitness=float(np.mean(all_fitness)),
            diversity=gen_metrics["diversity"], mutation_rate=current_rate,
            n_agents=self.n_systems,
        )
        self.archive.add_generation(record)
        return gen_metrics

    def _mutate(self, system: "MinimalSystem", rate: float, strength: float) -> "MinimalSystem":
        child = MinimalSystem.__new__(MinimalSystem)
        child.rng = np.random.RandomState()
        child.env = GridWorld(GridWorldConfig(
            size=2, n_agents=2, shared_goal_prob=1.0,
            reward_type="shared", task_type="shared",
            max_steps=20, seed=int(np.random.randint(0, 2**31)),
        ))
        comm_config = {"comm_dim": 2, "state_dim": 4, "vocab_size": 4, "message_length": 2}
        child.agents = []
        for agent in system.agents:
            a = BlindAgent(agent.id, agent.rng)
            # Mutate SimplePolicy params
            a.policy.params = agent.policy.params.copy()
            a.policy.mutate(rate, strength)
            a._comm_dim = agent._comm_dim
            a.memory = None
            child.agents.append(a)
        return child

    def _crossover(self, a: "MinimalSystem", b: "MinimalSystem") -> "MinimalSystem":
        child = MinimalSystem.__new__(MinimalSystem)
        child.rng = np.random.RandomState()
        child.env = GridWorld(GridWorldConfig(
            size=2, n_agents=2, shared_goal_prob=1.0,
            reward_type="shared", task_type="shared",
            max_steps=20, seed=int(np.random.randint(0, 2**31)),
        ))
        child.agents = []
        for i in range(len(a.agents)):
            parent_b = b.agents[i % len(b.agents)]
            child_agent = BlindAgent(a.agents[i].id, a.agents[i].rng)
            child_agent.policy = a.agents[i].policy.crossover(parent_b.policy)
            child_agent._comm_dim = a.agents[i]._comm_dim
            child_agent.memory = None
            child.agents.append(child_agent)
        if np.random.random() < 0.1:
            for agent in child.agents:
                agent.policy.mutate(0.05, 0.02)
        return child

    def evaluate(self) -> Dict:
        return {}

    def _get_n_generations(self) -> int:
        return self.n_generations


class BlindAgent:
    """Minimal agent for the blind coordination game."""

    def __init__(self, agent_id: int, rng: Optional[np.random.RandomState] = None):
        self.id = agent_id
        self.rng = rng or np.random.RandomState()
        self.fitness: float = 0.0
        self.total_reward: float = 0.0
        self._comm_dim = 2

        cfg = UniverseDialogueConfig.__new__(UniverseDialogueConfig)
        cfg.policy_type = "simple"
        cfg.policy_hidden_dim = 4
        cfg.comm_dim = 2
        cfg.memory_size = 2
        cfg.use_memory = False
        cfg.learn_communication = True
        cfg.channel_type = "vector"
        cfg.vocab_size = 4
        cfg.message_length = 2
        cfg.seed = self.rng.randint(0, 2**31)

        from ..agents.policy import PolicyConfig
        import copy
        self.config = cfg
        self.policy_cfg = PolicyConfig(
            input_dim=4 + 0 + 2,  # obs(4) + mem(0) + comm(2)
            hidden_dim=4,
            output_dim=4,
            policy_type="simple",
            seed=self.rng.randint(0, 2**31),
        )
        self.policy = SimplePolicy(self.policy_cfg)
        self.comm_channel = VectorChannel(dim=2, state_dim=4, rng=self.rng)
        self.memory = None
        self.comm_state = np.zeros(2)

    def act(self, obs: np.ndarray, comm_input: Optional[np.ndarray] = None) -> tuple:
        x = np.tanh(obs)

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

    def clone(self) -> "BlindAgent":
        c = BlindAgent(self.id, self.rng)
        c.policy = self.policy.clone()
        c.comm_channel = self.comm_channel.clone()
        c._comm_dim = self._comm_dim
        return c


class MinimalSystem:
    """A system of 2 blind agents on a 2x2 grid."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.env = GridWorld(GridWorldConfig(
            size=2, n_agents=2, shared_goal_prob=1.0,
            reward_type="shared", task_type="shared",
            max_steps=20, seed=seed + 1,
        ))
        self.agents = [
            BlindAgent(0, np.random.RandomState(seed + 100)),
            BlindAgent(1, np.random.RandomState(seed + 200)),
        ]

    def run_episode(self) -> tuple:
        obs = self.env.reset()

        # Blind ALL agents: zero out the goal in their observation
        for i in range(self.env.n_agents):
            obs[i][2:] = 0.0

        for agent in self.agents:
            agent.reset()

        total_reward = 0.0
        comm_history: list = []
        action_history: list = []
        reward_history: list = []

        for step in range(20):
            actions: Dict[int, int] = {}
            comms: Dict[int, np.ndarray] = {}

            for i, agent in enumerate(self.agents):
                comm_input = None
                if agent.comm_channel is not None and step > 0:
                    other_comms = [
                        self.agents[j].comm_state for j in range(len(self.agents)) if j != i
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

            # Blind again each step
            for j in range(self.env.n_agents):
                obs[j][2:] = 0.0

            reward_history.append(dict(rewards))

            for j, agent in enumerate(self.agents):
                agent.total_reward += rewards[j]

            total_reward += sum(rewards.values())
            if done:
                break

        for agent in self.agents:
            agent.fitness = agent.total_reward

        metadata = {
            "total_reward": total_reward,
            "n_steps": len(comm_history),
            "final_positions": {i: self.env.positions[i].copy() for i in range(self.env.n_agents)},
        }
        self._comm_history = comm_history
        self._action_history = action_history
        self._reward_history = reward_history
        return total_reward, comm_history, action_history, reward_history, metadata

    def get_comm_matrix(self, comm_history: List) -> np.ndarray:
        n_agents = len(self.agents)
        n_steps = len(comm_history)
        comm_dim = 2
        matrix = np.zeros((n_agents, n_steps, comm_dim))
        for t, comm_dict in enumerate(comm_history):
            for i in range(n_agents):
                matrix[i, t] = comm_dict[i][:comm_dim]
        return matrix

    def get_action_matrix(self, action_history: List) -> np.ndarray:
        n_agents = len(self.agents)
        n_steps = len(action_history)
        matrix = np.zeros((n_agents, n_steps), dtype=int)
        for t, action_dict in enumerate(action_history):
            for i in range(n_agents):
                matrix[i, t] = action_dict[i]
        return matrix

    def get_reward_matrix(self, reward_history: List) -> np.ndarray:
        n_agents = len(self.agents)
        n_steps = len(reward_history)
        matrix = np.zeros((n_agents, n_steps))
        for t, reward_dict in enumerate(reward_history):
            for i in range(n_agents):
                matrix[i, t] = reward_dict[i]
        return matrix
