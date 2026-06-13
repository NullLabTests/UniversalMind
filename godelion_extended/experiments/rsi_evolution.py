from typing import Any, Dict, List, Optional

import numpy as np
import numpy.linalg  # noqa: used for dispersion computation

from .base import BaseExperiment
from ..universe_dialogue import (
    UniverseDialogueConfig,
    MultiAgentSystem,
    UniverseDialogueAnalyzer,
    analyze_system,
)
from ..evolution.selection import select_parents, select_survivors
from ..evolution.mutation import mutate_agent, crossover_agents
from ..evolution.archive import GenerationRecord


class RSIEvolutionExperiment(BaseExperiment):
    """System-level evolution: population of full multi-agent systems.

    Each 'individual' in the evolutionary population is an entire
    MultiAgentSystem.  Fitness = total system reward + coordination bonus,
    so selection directly rewards emergent coordination.
    """

    def __init__(self, config: Dict, output_dir: str = "./output"):
        super().__init__(config, output_dir)

        evo_config = config.get("rsi_evolution", {})
        ud_config = config.get("universe_dialogue", {})

        self.n_systems = evo_config.get("n_systems", 20)
        self.n_generations = evo_config.get("n_generations", 50)
        self.mutation_rate = evo_config.get("mutation_rate", 0.1)
        self.mutation_strength = evo_config.get("mutation_strength", 0.05)
        self.selection_method = evo_config.get("selection", "truncation")
        self.tournament_size = evo_config.get("tournament_size", 3)
        self.elite_ratio = evo_config.get("elite_ratio", 0.2)
        self.crossover_rate = evo_config.get("crossover_rate", 0.3)
        self.coord_bonus = evo_config.get("coordination_bonus", 1.0)
        self.mutation_decay = evo_config.get("mutation_decay", 0.99)

        self.ud_kwargs = dict(
            n_agents=ud_config.get("n_agents", 6),
            env_size=ud_config.get("env_size", 10),
            n_steps=ud_config.get("n_steps", 100),
            n_trials=ud_config.get("n_trials", 3),
            policy_type=ud_config.get("agent", {}).get("policy_type", "mlp"),
            policy_hidden_dim=ud_config.get("agent", {}).get("policy_hidden_dim", 16),
            comm_dim=ud_config.get("agent", {}).get("comm_dim", 4),
            memory_size=ud_config.get("agent", {}).get("memory_size", 8),
            use_memory=ud_config.get("agent", {}).get("use_memory", True),
            learn_communication=ud_config.get("agent", {}).get("learn_communication", True),
            channel_type=ud_config.get("communication", {}).get("channel_type", "vector"),
            vocab_size=ud_config.get("communication", {}).get("vocab_size", 16),
            message_length=ud_config.get("communication", {}).get("message_length", 4),
            attention_heads=ud_config.get("communication", {}).get("attention_heads", 2),
            task_type=ud_config.get("task", {}).get("type", "mixed"),
            shared_goal_prob=ud_config.get("task", {}).get("shared_goal_prob", 0.5),
            reward_type=ud_config.get("task", {}).get("reward_type", "distance"),
            partitioned=ud_config.get("task", {}).get("partitioned", False),
            blind_ratio=ud_config.get("task", {}).get("blind_ratio", 0.5),
        )

    def setup(self):
        self.analyzer = UniverseDialogueAnalyzer(
            UniverseDialogueConfig(**self.ud_kwargs, n_generations=1, seed=self.seed + 200)
        )
        self.systems: List[MultiAgentSystem] = []

        base_seed = self.seed if self.seed is not None else 42
        for i in range(self.n_systems):
            cfg = UniverseDialogueConfig(
                **self.ud_kwargs,
                n_generations=1,
                seed=base_seed + 1000 + i,
            )
            self.systems.append(MultiAgentSystem(cfg))

        self.logger.log_message(
            f"RSI Evolution: {self.n_systems} systems, "
            f"{self.ud_kwargs['n_agents']} agents each, "
            f"{self.n_generations} generations"
        )

    def run_generation(self, generation: int) -> Dict:
        current_rate = self._decay(self.mutation_rate, generation)
        current_strength = self._decay(self.mutation_strength, generation)

        all_fitness: List[float] = []
        all_metrics: List[Dict] = []

        for system in self.systems:
            trial_results = []
            trial_total_rewards = []
            trial_dispersions = []
            for trial in range(self.ud_kwargs["n_trials"]):
                total_reward, comm_history, action_history, reward_history, metadata = (
                    system.run_episode()
                )
                metrics = self.analyzer.analyze_episode(
                    system, comm_history, action_history, reward_history
                )
                trial_results.append(metrics)
                trial_total_rewards.append(total_reward)

                # Dispersion: mean pairwise distance at final step
                positions = list(metadata.get("final_positions", {}).values())
                if len(positions) >= 2:
                    pairwise = 0.0
                    count = 0
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            pairwise += float(np.linalg.norm(positions[i] - positions[j]))
                            count += 1
                    trial_dispersions.append(pairwise / count if count > 0 else 0.0)
                else:
                    trial_dispersions.append(0.0)

            base_fitness = float(np.mean(trial_total_rewards))
            dispersion = float(np.mean(trial_dispersions)) if trial_dispersions else 0.0
            coord = float(np.mean([m.get("coordination_score_mean", 0) for m in trial_results]))

            # Fitness = shared reward - strong dispersion penalty (reward closeness)
            norm = max(float(abs(base_fitness)), 1.0)
            dispersion_penalty = 0.3 * dispersion * norm
            coord_bonus_value = self.coord_bonus * coord * norm
            system_fitness = base_fitness + coord_bonus_value - dispersion_penalty

            all_fitness.append(system_fitness)
            combined = self.analyzer.compute_generation_metrics(trial_results)
            combined["system_fitness"] = system_fitness
            combined["base_reward"] = base_fitness
            combined["dispersion"] = dispersion
            combined["dispersion_penalty"] = dispersion_penalty
            combined["coord_bonus_value"] = coord_bonus_value
            all_metrics.append(combined)

        # --- Selection: keep elite survivors, fill rest with offspring ---
        n_survivors = max(1, int(self.n_systems * self.elite_ratio))
        sorted_idx = np.argsort(all_fitness)[::-1]
        survivor_indices = sorted_idx[:n_survivors].tolist()

        n_children = self.n_systems - n_survivors
        new_systems: List[MultiAgentSystem] = []
        if n_children > 0:
            parent_indices = select_parents(
                [all_fitness[i] for i in survivor_indices],
                n_children,
                method="tournament",
                tournament_size=min(self.tournament_size, n_survivors),
            )
            parent_indices = [survivor_indices[i] for i in parent_indices]

            for idx in parent_indices:
                other = [i for i in parent_indices if i != idx]
                if self.crossover_rate > 0 and len(other) > 0 and np.random.random() < self.crossover_rate:
                    other_idx = int(np.random.choice(other))
                    child = self._crossover_systems(self.systems[idx], self.systems[other_idx])
                else:
                    child = self._mutate_system(self.systems[idx], current_rate, current_strength)
                new_systems.append(child)

        self.systems = [self.systems[i] for i in survivor_indices] + new_systems

        # --- Aggregate ---
        gen_metrics: Dict[str, float] = {}
        for key in [
            "coordination_score_mean", "synchronization_index_mean",
            "mutual_information_mean", "communication_entropy_mean",
            "shared_representation_score", "cross_prediction_error",
        ]:
            vals = [m.get(key, 0) for m in all_metrics if key in m]
            gen_metrics[key] = float(np.mean(vals)) if vals else 0.0

        gen_metrics["mean_fitness"] = float(np.mean(all_fitness))
        gen_metrics["best_fitness"] = float(max(all_fitness))
        gen_metrics["fitness_std"] = float(np.std(all_fitness))
        gen_metrics["diversity"] = float(
            np.std(all_fitness) / (abs(float(np.mean(all_fitness))) + 1e-8)
        )
        gen_metrics["mutation_rate"] = current_rate

        record = GenerationRecord(
            generation=generation,
            metrics=gen_metrics,
            agent_fitness=all_fitness,
            best_fitness=float(max(all_fitness)),
            mean_fitness=float(np.mean(all_fitness)),
            diversity=gen_metrics["diversity"],
            mutation_rate=current_rate,
            n_agents=self.n_systems,
        )
        self.archive.add_generation(record)
        return gen_metrics

    def _mutate_system(self, system: MultiAgentSystem, rate: float, strength: float) -> MultiAgentSystem:
        config = system.config
        child = MultiAgentSystem.__new__(MultiAgentSystem)
        child.config = config
        child.rng = np.random.RandomState()
        from ..environments.grid_world import GridWorld, GridWorldConfig
        child.env = GridWorld(GridWorldConfig(
            size=config.env_size, n_agents=config.n_agents,
            shared_goal_prob=config.shared_goal_prob,
            reward_type=config.reward_type, task_type=config.task_type,
            partitioned=config.partitioned, blind_ratio=config.blind_ratio,
            max_steps=config.n_steps,
            seed=int(config.seed) ^ 42 if config.seed is not None else int(np.random.randint(0, 2**31)),
        ))
        child.agents = [mutate_agent(a, rate, strength) for a in system.agents]
        return child

    def _crossover_systems(self, a: MultiAgentSystem, b: MultiAgentSystem) -> MultiAgentSystem:
        config = a.config
        child = MultiAgentSystem.__new__(MultiAgentSystem)
        child.config = config
        child.rng = np.random.RandomState()
        from ..environments.grid_world import GridWorld, GridWorldConfig
        child.env = GridWorld(GridWorldConfig(
            size=config.env_size, n_agents=config.n_agents,
            shared_goal_prob=config.shared_goal_prob,
            reward_type=config.reward_type, task_type=config.task_type,
            partitioned=config.partitioned, blind_ratio=config.blind_ratio,
            max_steps=config.n_steps,
            seed=int(config.seed) ^ 24 if config.seed is not None else int(np.random.randint(0, 2**31)),
        ))
        child.agents = []
        for i in range(len(a.agents)):
            parent_b = b.agents[i % len(b.agents)]
            child.agents.append(crossover_agents(a.agents[i], parent_b))
        return child

    @staticmethod
    def _decay(initial: float, gen: int) -> float:
        return float(np.clip(initial * (0.99 ** gen), 0.005, 0.5))

    def evaluate(self) -> Dict:
        return {}

    def _get_n_generations(self) -> int:
        return self.n_generations
