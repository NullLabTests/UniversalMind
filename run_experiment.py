#!/usr/bin/env python3
"""
UniversalMind: Recursive Self-Improvement & Multi-Agent Coordination Research Framework.

Usage:
    python run_experiment.py [--config CONFIG_PATH] [--experiment NAME] [OPTIONS]

Examples:
    # Run default Universe Dialogue experiment
    python run_experiment.py

    # Run RSI Evolution experiment
    python run_experiment.py --experiment rsi_evolution --generations 30

    # Custom config
    python run_experiment.py --config config.local.yaml

    # Quick test (2 generations, 2 agents)
    python run_experiment.py --generations 2 --agents 2 --trials 1
"""
import argparse
import sys
from pathlib import Path

from godelion_extended.config import Config, config as global_config
from godelion_extended.experiments import UniverseDialogueExperiment, RSIEvolutionExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="UniversalMind: RSI & Multi-Agent Coordination Research Framework"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["universe_dialogue", "rsi_evolution"],
                        help="Experiment type to run")
    parser.add_argument("--generations", type=int, default=None,
                        help="Number of generations")
    parser.add_argument("--agents", type=int, default=None,
                        help="Number of agents in the system")
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of trials per generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (2 generations, 2 agents, 1 trial)")
    return parser.parse_args()


def override_config(cfg: Config, args) -> Config:
    if args.experiment:
        cfg.set(args.experiment, "experiment", "name")
    if args.generations:
        for section in ["universe_dialogue", "rsi_evolution"]:
            if cfg.get(section, "n_generations"):
                cfg.set(args.generations, section, "n_generations")
    if args.agents:
        cfg.set(args.agents, "universe_dialogue", "n_agents")
    if args.trials:
        cfg.set(args.trials, "universe_dialogue", "n_trials")
    if args.output:
        cfg.set(args.output, "experiment", "output_dir")
    if args.seed:
        cfg.set(args.seed, "experiment", "seed")
    if args.quick:
        cfg.set(2, "universe_dialogue", "n_generations")
        cfg.set(2, "universe_dialogue", "n_agents")
        cfg.set(1, "universe_dialogue", "n_trials")
        cfg.set(2, "rsi_evolution", "n_generations")
        cfg.set(2, "rsi_evolution", "n_agents")
    return cfg


def main():
    args = parse_args()

    cfg = Config(args.config) if args.config else global_config
    cfg = override_config(cfg, args)

    experiment_name = cfg.get("experiment", "name", default="universe_dialogue")
    output_dir = cfg.get("experiment", "output_dir", default="./output")

    print(f"UniversalMind v{_get_version()}")
    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    print(f"Seed: {cfg.get('experiment', 'seed', default=42)}")
    print()

    if experiment_name == "rsi_evolution":
        experiment = RSIEvolutionExperiment(cfg.to_dict(), output_dir)
    else:
        experiment = UniverseDialogueExperiment(cfg.to_dict(), output_dir)

    archive = experiment.run()

    print(f"\nExperiment complete.")
    print(f"Results saved to: {experiment.logger.get_run_dir()}")
    print(f"Total generations: {len(archive.generations)}")
    if archive.generations:
        print(f"Best fitness: {archive.best_fitness_ever:.4f}")
        print(f"Final best fitness: {archive.generations[-1].best_fitness:.4f}")


def _get_version() -> str:
    try:
        from godelion_extended import __version__
        return __version__
    except ImportError:
        return "0.1.0"


if __name__ == "__main__":
    main()
