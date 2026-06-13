import numpy as np
from godelion_extended.evolution.selection import select_parents, select_survivors
from godelion_extended.evolution.archive import Archive, GenerationRecord


def test_select_parents_tournament():
    fitness = [0.1, 0.5, 0.8, 0.2, 0.9]
    parents = select_parents(fitness, n_parents=3, method="tournament", tournament_size=2)
    assert len(parents) == 3
    assert all(0 <= p < len(fitness) for p in parents)


def test_select_parents_proportional():
    fitness = [0.1, 0.5, 0.8, 0.2, 0.9]
    parents = select_parents(fitness, n_parents=3, method="proportional")
    assert len(parents) == 3


def test_select_parents_elite():
    fitness = [0.1, 0.5, 0.8, 0.2, 0.9]
    parents = select_parents(fitness, n_parents=3, method="elite")
    assert len(parents) == 3
    # Elite should select top 3
    assert 4 in parents  # index of 0.9
    assert 2 in parents  # index of 0.8


def test_select_parents_rank():
    fitness = [0.1, 0.5, 0.8, 0.2, 0.9]
    parents = select_parents(fitness, n_parents=3, method="rank")
    assert len(parents) == 3


def test_select_survivors_truncation():
    fitness = [0.1, 0.5, 0.8, 0.2, 0.9]
    survivors = select_survivors(fitness, n_survivors=2, method="truncation")
    assert len(survivors) == 2
    assert 4 in survivors  # 0.9
    assert 2 in survivors  # 0.8


def test_select_survivors_elite_plus_random():
    fitness = [0.1, 0.5, 0.8, 0.2, 0.9]
    survivors = select_survivors(fitness, n_survivors=3, method="elite_plus_random", elite_ratio=0.3)
    assert len(survivors) == 3


def test_archive():
    archive = Archive(output_dir="/tmp/test_archive")
    record = GenerationRecord(
        generation=0,
        metrics={"score": 1.0},
        agent_fitness=[0.1, 0.5],
        best_fitness=0.5,
        mean_fitness=0.3,
        diversity=0.2,
        mutation_rate=0.1,
        n_agents=2,
    )
    archive.add_generation(record)
    assert len(archive.generations) == 1
    assert archive.best_fitness_ever == 0.5

    record2 = GenerationRecord(
        generation=1,
        metrics={"score": 2.0},
        agent_fitness=[0.2, 0.9],
        best_fitness=0.9,
        mean_fitness=0.55,
        diversity=0.35,
        mutation_rate=0.1,
        n_agents=2,
    )
    archive.add_generation(record2)
    assert archive.best_fitness_ever == 0.9
    assert len(archive.get_fitness_history()) == 2
    assert len(archive.get_mean_fitness_history()) == 2
    assert len(archive.get_diversity_history()) == 2


def test_archive_save_load():
    import os
    archive = Archive(output_dir="/tmp/test_archive_save")
    record = GenerationRecord(generation=0, metrics={"score": 1.0}, agent_fitness=[0.5],
                               best_fitness=0.5, mean_fitness=0.5, diversity=0.0,
                               mutation_rate=0.1, n_agents=1)
    archive.add_generation(record)
    save_path = "/tmp/test_archive_save/archive.json"
    archive.save(save_path)
    assert os.path.exists(save_path)

    loaded = Archive(output_dir="/tmp/test_archive_load")
    loaded.load(save_path)
    assert len(loaded.generations) == 1
    assert loaded.best_fitness_ever == 0.5
