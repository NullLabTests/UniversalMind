import numpy as np
from godelion_extended.environments.grid_world import GridWorld, GridWorldConfig


def test_grid_world_initialization():
    config = GridWorldConfig(size=10, n_agents=5, seed=42)
    env = GridWorld(config)
    obs = env.reset()
    assert len(obs) == 5
    for i in range(5):
        assert obs[i].shape == (4,)


def test_grid_world_step():
    config = GridWorldConfig(size=10, n_agents=3, seed=42)
    env = GridWorld(config)
    obs = env.reset()

    actions = {0: 0, 1: 1, 2: 2}
    next_obs, rewards, done = env.step(actions)

    assert len(next_obs) == 3
    assert len(rewards) == 3
    assert isinstance(done, bool)
    assert rewards[0] < 0  # distance-based negative reward


def test_grid_world_bounds():
    config = GridWorldConfig(size=5, n_agents=1, seed=42)
    env = GridWorld(config)
    env.reset()
    env.positions[0] = np.array([0.0, 0.0])

    actions = {0: 2}  # left from 0
    env.step(actions)
    assert np.all(env.positions[0] >= 0)
    assert np.all(env.positions[0] <= 4)

    env.positions[0] = np.array([4.0, 4.0])
    actions = {0: 3}  # right from max
    env.step(actions)
    assert np.all(env.positions[0] <= 4)


def test_grid_world_max_steps():
    config = GridWorldConfig(size=5, n_agents=1, max_steps=3, seed=42)
    env = GridWorld(config)
    env.reset()

    actions = {0: 0}
    for _ in range(3):
        _, _, done = env.step(actions)
    assert done


def test_grid_world_shared_goal():
    config = GridWorldConfig(size=10, n_agents=3, task_type="shared", seed=42)
    env = GridWorld(config)
    obs = env.reset()

    targets = [env.targets[i] for i in range(3)]
    assert np.allclose(targets[0], targets[1])
    assert np.allclose(targets[1], targets[2])


def test_grid_world_mixed_goals():
    config = GridWorldConfig(size=10, n_agents=5, task_type="mixed", shared_goal_prob=0.6, seed=42)
    env = GridWorld(config)
    obs = env.reset()

    assert len(env.targets) == 5
    for t in env.targets.values():
        assert t.shape == (2,)


def test_grid_world_global_state():
    config = GridWorldConfig(size=10, n_agents=3, seed=42)
    env = GridWorld(config)
    env.reset()
    gs = env.get_global_state()
    assert gs.shape == (12,)


def test_sparse_reward():
    config = GridWorldConfig(size=5, n_agents=1, reward_type="sparse", seed=42)
    env = GridWorld(config)
    env.reset()

    env.positions[0] = env.targets[0].copy()
    _, rewards, _ = env.step({0: 0})
    assert rewards[0] >= 0


def test_get_positions_targets():
    config = GridWorldConfig(size=10, n_agents=3, seed=42)
    env = GridWorld(config)
    env.reset()
    positions = env.get_positions()
    targets = env.get_targets()
    assert len(positions) == 3
    assert len(targets) == 3
