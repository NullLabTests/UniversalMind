import numpy as np
from godelion_extended.universe_dialogue import (
    UniverseDialogueConfig,
    DialogueAgent,
    MultiAgentSystem,
    UniverseDialogueAnalyzer,
    analyze_system,
)


def make_config(**kwargs):
    params = {
        "n_agents": 3,
        "env_size": 5,
        "n_steps": 10,
        "n_generations": 2,
        "n_trials": 1,
        "policy_hidden_dim": 8,
        "comm_dim": 2,
        "memory_size": 4,
        "use_memory": True,
        "learn_communication": True,
        "channel_type": "vector",
        "vocab_size": 8,
        "message_length": 2,
        "attention_heads": 1,
        "task_type": "individual",
        "shared_goal_prob": 0.3,
        "reward_type": "distance",
        "seed": 42,
    }
    params.update(kwargs)
    return UniverseDialogueConfig(**params)


def test_dialogue_agent():
    config = make_config()
    agent = DialogueAgent(0, config)
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    action, comm = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action <= 3
    assert comm.shape == (config.comm_dim,)


def test_multi_agent_system_run():
    config = make_config(n_agents=3, n_steps=10)
    system = MultiAgentSystem(config)
    total_reward, comm_history, action_history, reward_history, metadata = system.run_episode()
    assert isinstance(total_reward, float)
    assert len(comm_history) <= 10
    assert len(action_history) <= 10
    assert len(reward_history) <= 10
    assert "total_reward" in metadata


def test_system_communication():
    config = make_config(n_agents=3, n_steps=20, comm_dim=2)
    system = MultiAgentSystem(config)
    _, comm_history, _, _, _ = system.run_episode()
    comm_matrix = system.get_comm_matrix(comm_history)
    assert comm_matrix.shape == (3, len(comm_history), 2)


def test_system_action_matrix():
    config = make_config(n_agents=3, n_steps=10)
    system = MultiAgentSystem(config)
    _, _, action_history, _, _ = system.run_episode()
    action_matrix = system.get_action_matrix(action_history)
    assert action_matrix.shape == (3, len(action_history))


def test_system_reward_matrix():
    config = make_config(n_agents=3, n_steps=10)
    system = MultiAgentSystem(config)
    _, _, _, reward_history, _ = system.run_episode()
    reward_matrix = system.get_reward_matrix(reward_history)
    assert reward_matrix.shape == (3, len(reward_history))


def test_analyzer():
    config = make_config(n_agents=3, n_steps=20)
    system = MultiAgentSystem(config)
    _, comm_history, action_history, reward_history, _ = system.run_episode()
    analyzer = UniverseDialogueAnalyzer(config)
    metrics = analyzer.analyze_episode(system, comm_history, action_history, reward_history)
    assert "total_system_reward" in metrics
    assert "coordination_score" in metrics
    assert "agent_rewards" in metrics
    assert "n_steps" in metrics


def test_analyze_system():
    config = make_config(n_agents=3, n_steps=20, n_trials=2)
    system = MultiAgentSystem(config)
    result = analyze_system(config, system, n_trials=2)
    assert "total_system_reward_mean" in result
    assert "n_trials" in result
    assert len(result["trial_metrics"]) == 2


def test_shared_goal_mode():
    config = make_config(n_agents=3, n_steps=10, task_type="shared")
    system = MultiAgentSystem(config)
    total_reward, _, _, _, _ = system.run_episode()
    assert isinstance(total_reward, float)


def test_no_communication():
    config = make_config(n_agents=3, n_steps=10, learn_communication=False)
    system = MultiAgentSystem(config)
    total_reward, _, _, _, _ = system.run_episode()
    assert isinstance(total_reward, float)


def test_no_memory():
    config = make_config(n_agents=3, n_steps=10, use_memory=False)
    system = MultiAgentSystem(config)
    total_reward, _, _, _, _ = system.run_episode()
    assert isinstance(total_reward, float)


def test_discrete_communication():
    config = make_config(n_agents=3, n_steps=10, channel_type="discrete",
                          vocab_size=8, message_length=2)
    system = MultiAgentSystem(config)
    total_reward, _, _, _, _ = system.run_episode()
    assert isinstance(total_reward, float)


def test_attention_communication():
    config = make_config(n_agents=3, n_steps=10, channel_type="attention",
                          attention_heads=2, comm_dim=4)
    system = MultiAgentSystem(config)
    total_reward, _, _, _, _ = system.run_episode()
    assert isinstance(total_reward, float)
