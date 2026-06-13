from typing import Any, Dict, List, Optional

import numpy as np

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
from ..rsi.meta_learner import MetaLearner
from ..rsi.mutation_strategies import AdaptiveStrategy


class RSIEvolutionExperiment(BaseExperiment):
    def __init__(self, config: Dict, output_dir: str = "./output"):
        super().__init__(config, output_dir)

        evo_config = config.get("rsi_evolution", {})
        ud_config = config.get("universe_dialogue", {})

        self.n_agents = evo_config.get("n_agents", 20)
        self.n_generations = evo_config.get("n_generations", 50)
        self.mutation_rate = evo_config.get("mutation_rate", 0.1)
        self.mutation_strength = evo_config.get("mutation_strength", 0.05)
        self.selection_method = evo_config.get("selection", "tournament")
        self.tournament_size = evo_config.get("tournament_size", 3)
        self.elite_ratio = evo_config.get("elite_ratio", 0.1)
        self.crossover_rate = evo_config.get("crossover_rate", 0.3)

        self.use_meta = evo_config.get("meta_evolution", {}).get("enabled", False)

        self.ud_config = UniverseDialogueConfig(
            n_agents=ud_config.get("n_agents", 6),
            env_size=ud_config.get("env_size", 10),
            n_steps=ud_config.get("n_steps", 100),
            n_generations=1,
            n_trials=ud_config.get("n_trials", 3),
            seed=self.seed + 200,
        )

        self.mutation_strategy = AdaptiveStrategy(
            initial_rate=self.mutation_rate,
            initial_strength=self.mutation_strength,
        )
        self.meta_learner: Optional[MetaLearner] = None
        self.systems: List[MultiAgentSystem] = []

    def setup(self):
        self.analyzer = UniverseDialogueAnalyzer(self.ud_config)
        if self.use_meta:
            self.meta_learner = MetaLearner(n_strategies=3, seed=self.seed + 100)

        self.systems = []
        for i in range(self.n_agents):
            gen_config = UniverseDialogueConfig(
                n_agents=self.ud_config.n_agents,
                env_size=self.ud_config.env_size,
                n_steps=self.ud_config.n_steps,
                n_generations=1,
                n_trials=self.ud_config.n_trials,
                seed=(self.seed + 1000 + i) if self.seed is not None else None,
            )
            system = MultiAgentSystem(gen_config)
            self.systems.append(system)

        self.logger.log_message(f"RSI Evolution experiment initialized with {self.n_agents} agents")

    def run_generation(self, generation: int) -> Dict:
        all_fitness = []
        all_metrics = []

        current_rate = self.mutation_rate
        current_strength = self.mutation_strength

        for i, system in enumerate(self.systems):
            trial_results = []
            agent_fitness = []
            for trial in range(self.ud_config.n_trials):
                total_reward, comm_history, action_history, reward_history, metadata = (
                    system.run_episode()
                )
                metrics = self.analyzer.analyze_episode(
                    system, comm_history, action_history, reward_history
                )
                trial_results.append(metrics)
                trial_fitness = [system.agents[j].total_reward for j in range(self.ud_config.n_agents)]
                agent_fitness.append(np.mean(trial_fitness))

            avg_fitness = float(np.mean(agent_fitness))
            all_fitness.append(avg_fitness)

            combined = self.analyzer.compute_generation_metrics(trial_results)
            combined["mean_agent_fitness"] = avg_fitness
            all_metrics.append(combined)

        n_survivors = max(1, int(self.n_agents * self.elite_ratio))
        survivor_indices = select_survivors(
            all_fitness, n_survivors,
            method="truncation",
            elite_ratio=self.elite_ratio,
        )

        fitness_trend = 0.0
        if generation > 0 and self.archive.generations:
            prev_mean = self.archive.generations[-1].mean_fitness
            curr_mean = float(np.mean(all_fitness))
            fitness_trend = curr_mean - prev_mean

        n_parents = self.n_agents - n_survivors
        if n_parents > 0 and len(survivor_indices) > 0:
            parent_indices = select_parents(
                [all_fitness[i] for i in survivor_indices],
                n_parents,
                method=self.selection_method,
                tournament_size=self.tournament_size,
            )
            parent_indices = [survivor_indices[i] for i in parent_indices]

            new_systems = []
            for idx in parent_indices:
                if self.crossover_rate > 0 and np.random.random() < self.crossover_rate:
                    other_idx = np.random.choice([i for i in parent_indices if i != idx])
                    child_system = self._crossover_systems(self.systems[idx], self.systems[other_idx])
                else:
                    parent = self.systems[idx]
                    child_system = self._mutate_system(parent, current_rate, current_strength)
                new_systems.append(child_system)

            replace_indices = [i for i in range(self.n_agents) if i not in survivor_indices]
            for j, idx in enumerate(replace_indices[:len(new_systems)]):
                self.systems[idx] = new_systems[j]

        diversity = float(np.std(all_fitness) / (np.abs(np.mean(all_fitness)) + 1e-8))

        # Aggregate metrics
        gen_metrics = {}
        for key in ["coordination_score_mean", "synchronization_index_mean",
                     "mutual_information_mean", "communication_entropy_mean",
                     "shared_representation_score", "cross_prediction_error"]:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                gen_metrics[key] = float(np.mean(values))
            else:
                gen_metrics[key] = 0.0

        gen_metrics["mean_fitness"] = float(np.mean(all_fitness))
        gen_metrics["best_fitness"] = float(max(all_fitness))
        gen_metrics["fitness_std"] = float(np.std(all_fitness))
        gen_metrics["diversity"] = diversity

        record = GenerationRecord(
            generation=generation,
            metrics=gen_metrics,
            agent_fitness=all_fitness,
            best_fitness=float(max(all_fitness)),
            mean_fitness=float(np.mean(all_fitness)),
            diversity=diversity,
            mutation_rate=current_rate,
            n_agents=self.n_agents,
        )
        self.archive.add_generation(record)

        return gen_metrics

    def _mutate_system(self, system: MultiAgentSystem, rate: float, strength: float) -> MultiAgentSystem:
        import copy
        new_system = MultiAgentSystem.__new__(MultiAgentSystem)
        new_system.config = system.config
        new_system.rng = np.random.RandomState()
        new_system.env = system.env

        new_system.agents = []
        for agent in system.agents:
            mutant = mutate_agent(agent, rate * 2, strength * 2)
            new_system.agents.append(mutant)

        return new_system

    def _crossover_systems(self, sys_a: MultiAgentSystem, sys_b: MultiAgentSystem) -> MultiAgentSystem:
        import copy
        new_system = MultiAgentSystem.__new__(MultiAgentSystem)
        new_system.config = sys_a.config
        new_system.rng = np.random.RandomState()
        new_system.env = sys_a.env

        new_system.agents = []
        for i in range(len(sys_a.agents)):
            child = crossover_agents(sys_a.agents[i], sys_b.agents[i % len(sys_b.agents)])
            new_system.agents.append(child)

        return new_system

    def evaluate(self) -> Dict:
        final_metrics = self.run_generation(self._get_n_generations())
        return final_metrics

    def _get_n_generations(self) -> int:
        return self.n_generations
