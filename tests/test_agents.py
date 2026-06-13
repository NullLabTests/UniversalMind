import numpy as np
from godelion_extended.agents.policy import MLPPolicy, LinearPolicy, PolicyConfig
from godelion_extended.agents.memory import AgentMemory
from godelion_extended.agents.communication import (
    VectorChannel, DiscreteChannel, AttentionChannel, create_channel,
)


def test_mlp_policy_forward():
    cfg = PolicyConfig(input_dim=4, hidden_dim=16, output_dim=4)
    policy = MLPPolicy(cfg)
    x = np.random.randn(4)
    out = policy.forward(x)
    assert out.shape == (4,)


def test_mlp_policy_clone():
    cfg = PolicyConfig(input_dim=4, hidden_dim=16, output_dim=4, seed=42)
    policy = MLPPolicy(cfg)
    clone = policy.clone()
    x = np.random.randn(4)
    np.testing.assert_array_almost_equal(policy.forward(x), clone.forward(x))


def test_mlp_policy_mutate():
    cfg = PolicyConfig(input_dim=4, hidden_dim=16, output_dim=4, seed=42)
    policy = MLPPolicy(cfg)
    x = np.random.randn(4)
    orig_out = policy.forward(x).copy()
    policy.mutate(rate=1.0, strength=1.0)
    new_out = policy.forward(x)
    assert not np.allclose(orig_out, new_out)


def test_mlp_policy_crossover():
    cfg = PolicyConfig(input_dim=4, hidden_dim=16, output_dim=4, seed=42)
    a = MLPPolicy(cfg)
    b = MLPPolicy(PolicyConfig(input_dim=4, hidden_dim=16, output_dim=4, seed=99))
    child = a.crossover(b)
    x = np.random.randn(4)
    assert child.forward(x).shape == (4,)


def test_linear_policy():
    cfg = PolicyConfig(input_dim=4, hidden_dim=8, output_dim=4, policy_type="linear")
    policy = LinearPolicy(cfg)
    x = np.random.randn(4)
    out = policy.forward(x)
    assert out.shape == (4,)


def test_agent_memory():
    mem = AgentMemory(memory_size=8, state_dim=4)
    initial = mem.get_state()
    assert initial.shape == (4,)
    assert np.allclose(initial, np.zeros(4))

    mem.push(np.array([1.0, 2.0, 3.0, 4.0]))
    state = mem.get_state()
    assert not np.allclose(state, np.zeros(4))

    mem.reset()
    state = mem.get_state()
    assert np.allclose(state, np.zeros(4))


def test_agent_memory_full():
    mem = AgentMemory(memory_size=3, state_dim=2)
    for i in range(5):
        mem.push(np.array([float(i), float(i * 2)]))
    state = mem.get_state()
    assert state.shape == (2,)


def test_vector_channel():
    channel = VectorChannel(dim=4, state_dim=4, rng=np.random.RandomState(42))
    state = np.random.randn(4)
    msg = channel.encode(state)
    assert msg.shape == (4,)
    decoded = channel.decode(msg)
    assert decoded.shape == (4,)


def test_discrete_channel():
    channel = DiscreteChannel(vocab_size=16, message_length=4, state_dim=4)
    state = np.random.randn(4)
    msg = channel.encode(state)
    assert msg.shape == (4 * 16,)
    decoded = channel.decode(msg)
    assert decoded.shape == (4,)


def test_attention_channel():
    channel = AttentionChannel(dim=4, n_heads=2, state_dim=4)
    state = np.random.randn(4)
    msg = channel.encode(state)
    assert msg.shape == (4,)
    decoded = channel.decode(msg)
    assert decoded.shape == (4,)


def test_create_channel():
    config = {"comm_dim": 4, "state_dim": 4, "vocab_size": 16, "message_length": 4, "attention_heads": 2}
    rng = np.random.RandomState(42)

    vc = create_channel("vector", config, rng)
    assert isinstance(vc, VectorChannel)

    dc = create_channel("discrete", config, rng)
    assert isinstance(dc, DiscreteChannel)

    ac = create_channel("attention", config, rng)
    assert isinstance(ac, AttentionChannel)


def test_channel_mutate():
    channel = VectorChannel(dim=4, state_dim=4)
    state = np.random.randn(4)
    orig_msg = channel.encode(state).copy()
    channel.mutate(rate=1.0, strength=1.0)
    new_msg = channel.encode(state)
    assert not np.allclose(orig_msg, new_msg)


def test_channel_clone():
    channel = VectorChannel(dim=4, state_dim=4, rng=np.random.RandomState(42))
    clone = channel.clone()
    state = np.random.randn(4)
    np.testing.assert_array_almost_equal(channel.encode(state), clone.encode(state))
