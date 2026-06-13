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


class UniverseDialogueExperiment(BaseExperiment):
    def __init__(self, config: Dict, output_dir: str = "./output"):
        super().__init__(config, output_dir)
        ud_config = config.get("universe_dialogue", {})
        self.ud_config = UniverseDialogueConfig(
            n_agents=ud_config.get("n_agents", 6),
            env_size=ud_config.get("env_size", 10),
            n_steps=ud_config.get("n_steps", 100),
            n_generations=ud_config.get("n_generations", 50),
            n_trials=ud_config.get("n_trials", 3),
            policy_hidden_dim=ud_config.get("agent", {}).get("policy_hidden_dim", 16),
            comm_dim=ud_config.get("agent", {}).get("comm_dim", 4),
            memory_size=ud_config.get("agent", {}).get("memory_size", 8),
            use_memory=ud_config.get("agent", {}).get("use_memory", True),
            learn_communication=ud_config.get("agent", {}).get("learn_communication", True),
            channel_type=ud_config.get("communication", {}).get("channel_type", "vector"),
            vocab_size=ud_config.get("communication", {}).get("vocab_size", 16),
            message_length=ud_config.get("communication", {}).get("message_length", 4),
            attention_heads=ud_config.get("communication", {}).get("attention_heads", 2),
            task_type=ud_config.get("task", {}).get("type", "individual"),
            shared_goal_prob=ud_config.get("task", {}).get("shared_goal_prob", 0.3),
            reward_type=ud_config.get("task", {}).get("reward_type", "distance"),
            seed=self.seed,
        )

        evo_config = config.get("rsi_evolution", {})
        self.evo_config = {
            "n_agents": self.ud_config.n_agents,
            "n_generations": self.ud_config.n_generations,
            "mutation_rate": evo_config.get("mutation_rate", 0.1),
            "mutation_strength": evo_config.get("mutation_strength", 0.05),
            "selection": evo_config.get("selection", "tournament"),
            "tournament_size": evo_config.get("tournament_size", 3),
            "elite_ratio": evo_config.get("elite_ratio", 0.1),
            "crossover_rate": evo_config.get("crossover_rate", 0.3),
        }

        self.use_meta = evo_config.get("meta_evolution", {}).get("enabled", False)
        self.meta_interval = evo_config.get("meta_evolution", {}).get("meta_interval", 10)

        self.system: Optional[MultiAgentSystem] = None
        self.analyzer: Optional[UniverseDialogueAnalyzer] = None
        self.meta_learner: Optional[MetaLearner] = None

    def setup(self):
        self.system = MultiAgentSystem(self.ud_config)
        self.analyzer = UniverseDialogueAnalyzer(self.ud_config)
        if self.use_meta:
            self.meta_learner = MetaLearner(n_strategies=5, seed=self.seed + 100)
        self.logger.log_message(f"Universe Dialogue experiment initialized with {self.ud_config.n_agents} agents")

    def run_generation(self, generation: int) -> Dict:
        current_mutation_rate = self.evo_config["mutation_rate"]
        current_selection = self.evo_config["selection"]

        if self.use_meta and self.meta_learner and generation % self.meta_interval == 0:
            strategy = self.meta_learner.select_strategy()
            current_mutation_rate = strategy.mutation_rate
            current_selection = strategy.selection_method
            self.evo_config["selection"] = current_selection

        trial_metrics = []
        all_agent_fitness = []

        for trial in range(self.ud_config.n_trials):
            total_reward, comm_history, action_history, reward_history, metadata = (
                self.system.run_episode()
            )
            trial_result = self.analyzer.analyze_episode(
                self.system, comm_history, action_history, reward_history
            )
            trial_result["trial"] = trial
            trial_metrics.append(trial_result)

            agent_fitness = [self.system.agents[i].total_reward for i in range(self.ud_config.n_agents)]
            all_agent_fitness.append(agent_fitness)

        gen_metrics = self.analyzer.compute_generation_metrics(trial_metrics)
        gen_metrics["n_trials"] = self.ud_config.n_trials

        avg_agent_fitness = np.mean(all_agent_fitness, axis=0).tolist()

        n_survivors = max(1, int(self.ud_config.n_agents * (1 - self.evo_config["elite_ratio"])))
        survivor_indices = select_survivors(
            avg_agent_fitness, n_survivors,
            method="truncation",
            elite_ratio=self.evo_config["elite_ratio"],
        )

        if len(survivor_indices) < 2:
            survivor_indices = list(range(self.ud_config.n_agents))

        n_parents = self.ud_config.n_agents - len(survivor_indices)
        if n_parents > 0 and len(survivor_indices) > 0:
            parent_indices = select_parents(
                [avg_agent_fitness[i] for i in survivor_indices],
                n_parents,
                method=self.evo_config["selection"],
                tournament_size=self.evo_config["tournament_size"],
            )
            parent_indices = [survivor_indices[i] for i in parent_indices]

            new_agents = []
            for idx in parent_indices:
                parent = self.system.agents[idx]
                if self.evo_config["crossover_rate"] > 0 and np.random.random() < self.evo_config["crossover_rate"]:
                    other_idx = np.random.choice([i for i in parent_indices if i != idx])
                    child = crossover_agents(parent, self.system.agents[other_idx])
                else:
                    child = mutate_agent(parent, current_mutation_rate, self.evo_config["mutation_strength"])
                child.id = self.ud_config.n_agents + len(new_agents)
                new_agents.append(child)

            replace_indices = [i for i in range(self.ud_config.n_agents) if i not in survivor_indices]
            for i, idx in enumerate(replace_indices[:len(new_agents)]):
                self.system.agents[idx] = new_agents[i]

        if self.use_meta and self.meta_learner and generation > 0:
            prev_best = self.archive.get_fitness_history()[-1] if self.archive.generations else -float("inf")
            current_best = max(avg_agent_fitness)
            fitness_gain = current_best - prev_best if prev_best > -float("inf") else 0
            self.meta_learner.update_strategy_fitness(0, fitness_gain)
            if generation % self.meta_interval == 0:
                self.meta_learner.mutate_strategies()

        diversity = float(np.std(avg_agent_fitness) / (np.mean(np.abs(avg_agent_fitness)) + 1e-8))

        # Phase transitions count
        phase_n = 0
        for tm in trial_metrics:
            pt = tm.get("phase_transitions", {})
            if isinstance(pt, dict):
                phase_n += pt.get("n_transitions", 0)

        record = GenerationRecord(
            generation=generation,
            metrics=gen_metrics,
            agent_fitness=avg_agent_fitness,
            best_fitness=max(avg_agent_fitness),
            mean_fitness=float(np.mean(avg_agent_fitness)),
            diversity=diversity,
            mutation_rate=current_mutation_rate,
            n_agents=self.ud_config.n_agents,
        )
        self.archive.add_generation(record)

        gen_metrics["diversity"] = diversity
        gen_metrics["phase_transitions_n"] = phase_n

        for i, agent in enumerate(self.system.agents):
            genome = {
                "id": i,
                "fitness": avg_agent_fitness[i] if i < len(avg_agent_fitness) else 0.0,
            }
            self.archive.save_agent_genome(generation, genome)

        self.logger.save_json(gen_metrics, f"generation_{generation:04d}.json")

        return gen_metrics

    def evaluate(self) -> Dict:
        final_metrics = self.run_generation(self._get_n_generations())
        self.logger.log_message("Final evaluation complete.")
        return final_metrics

    def _get_n_generations(self) -> int:
        return self.ud_config.n_generations
