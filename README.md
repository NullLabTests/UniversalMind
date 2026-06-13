<div align="center">
  <h1>🧠 UniversalMind</h1>
</div>

<p align="center">
  <strong>Recursive Self-Improvement & Multi-Agent Coordination Research Framework</strong><br>
  <em>Empirically testing whether distributed agents converge toward globally coordinated behavior beyond local rules</em>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Native-013243?style=for-the-badge&logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/status-research--grade-ff6b35?style=for-the-badge" alt="Status">
</p>

---

## 🔬 What is UniversalMind?

UniversalMind is a rigorous experimental platform for studying **emergent coordination in recursive self-improving multi-agent systems**. It extends the [Godelion](https://github.com/NullLabTests/godelion) evolutionary self-improvement paradigm into a multi-agent context, enabling empirical testing of a fundamental question:

> **Do recursive self-improving multi-agent systems converge toward globally coordinated behavior that cannot be explained by local rules alone?**

### Primary Research Hypothesis

The system is designed to test two competing hypotheses:

| Hypothesis | Prediction |
|---|---|
| **H1 — Local Reducibility** | All coordination is reducible to local interaction + communication. Cross-agent behavioral coupling is fully explained by the information channel bandwidth. |
| **H2 — Emergent Unity** | Evolution produces emergent global structure resembling a unified optimization process. Coordination exceeds what communication bandwidth alone can explain, implying shared latent structure. |

**You are not allowed to assume H2. The system must attempt to falsify it.**

### Key Features

| Feature | Description |
|---|---|
| **🧬 Darwinian Evolution** | Agent populations evolve through selection, mutation, and crossover with fitness-based pressure |
| **🔄 Recursive Self-Improvement** | Mutation strategies, selection pressures, and evaluation heuristics themselves evolve (meta-RSI) |
| **🤝 Multi-Agent Universe** | 5–50 agents in a partially observable grid world with communication channels |
| **💬 Communication Systems** | Vector, discrete symbolic, and attention-based message routing with learnable protocols |
| **📊 Rigorous Metrics** | Mutual information, synchronization index, communication entropy, coordination scores, emergence indicators, phase transition detection |
| **🧪 Falsifiable Design** | Every metric is designed to test whether coordination exceeds communication-bounded predictions |
| **🔁 Meta-Evolution** | Not just agents, but mutation strategies and evolutionary parameters themselves undergo selection |
| **📈 Full Visualization** | Fitness curves, coordination heatmaps, communication entropy plots, phase transition detection |
| **🔐 Reproducible** | Deterministic seeding, configuration-driven experiments, full audit trail |
| **🖥️ Local-First** | Pure Python/NumPy — no external APIs, no Docker requirement. Runs on any machine. |

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Experiments](#-experiments)
- [Metrics](#-metrics)
- [Configuration](#-configuration)
- [Visualization](#-visualization)
- [Extending the Framework](#-extending-the-framework)
- [Project Structure](#-project-structure)
- [Research Background](#-research-background)
- [License](#-license)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Install

```bash
git clone https://github.com/anomalyco/UniversalMind.git
cd UniversalMind

python3 -m venv venv
source venv/bin/activate

pip install -e .
```

### 2. Run Default Experiment

```bash
python run_experiment.py
```

This runs the **Universe Dialogue** experiment with default parameters (6 agents, 50 generations).

### 3. Quick Test

```bash
python run_experiment.py --quick
```

Runs 2 generations with 2 agents — verifies the pipeline in seconds.

### 4. Custom Experiments

```bash
# Universe Dialogue — 10 agents, 100 generations, 5 trials
python run_experiment.py --experiment universe_dialogue --agents 10 --generations 100 --trials 5

# RSI Evolution — 50 agents, 100 generations
python run_experiment.py --experiment rsi_evolution --agents 50 --generations 100

# Custom config
cp config.yaml config.local.yaml
# Edit config.local.yaml
python run_experiment.py --config config.local.yaml

# Reproducible run
python run_experiment.py --seed 12345
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL MIND FRAMEWORK                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐     ┌─────────────────┐                     │
│  │  UNIVERSE        │     │  RSI EVOLUTION   │                     │
│  │  DIALOGUE        │     │  SYSTEM           │                     │
│  │  EXPERIMENT      │     │  (meta-evolution) │                     │
│  │                  │     │                   │                     │
│  │  ┌───────────┐   │     │  ┌─────────────┐  │                     │
│  │  │ GridWorld │   │     │  │ MetaLearner  │  │                     │
│  │  │ Environment│   │     │  │ (strategies  │  │                     │
│  │  └─────┬─────┘   │     │  │  evolve)     │  │                     │
│  │        │         │     │  └──────┬──────┘  │                     │
│  │  ┌─────▼─────┐   │     │         │         │                     │
│  │  │ Dialogue  │   │     │  ┌──────▼──────┐  │                     │
│  │  │ Agents    │───┼─────┼──► Evolutionary │  │                     │
│  │  │ (N=5-50)  │   │     │  │ Engine       │  │                     │
│  │  └─────┬─────┘   │     │  │ (selection,  │  │                     │
│  │        │         │     │  │  mutation,   │  │                     │
│  │  ┌─────▼─────┐   │     │  │  archive)    │  │                     │
│  │  │ Comm       │   │     │  └──────┬──────┘  │                     │
│  │  │ Channels   │   │     │         │         │                     │
│  │  └───────────┘   │     │  ┌──────▼──────┐  │                     │
│  └────────┬─────────┘     │  │ Mutation    │  │                     │
│           │               │  │ Strategy    │  │                     │
│           │               │  │ Evolution   │  │                     │
│           │               │  └─────────────┘  │                     │
│           │               └────────────────────┘                     │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────┐                           │
│  │          UNIFIED METRICS LAYER       │                           │
│  │                                      │                           │
│  │  ┌──────────┐ ┌──────────┐ ┌──────┐ │                           │
│  │  │Coordination │ Emergence │ Info │ │                           │
│  │  │ Metrics   │ │ Detection │Theory│ │                           │
│  │  └──────────┘ └──────────┘ └──────┘ │                           │
│  └──────────────────┬───────────────────┘                           │
│                     │                                               │
│  ┌──────────────────▼───────────────────┐                           │
│  │         ANALYTICS LAYER              │                           │
│  │  ┌────────┐ ┌──────────┐ ┌────────┐  │                           │
│  │  │ Logger  │ │Visualizer│ │Reporter│  │                           │
│  │  └────────┘ └──────────┘ └────────┘  │                           │
│  └──────────────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

### Core Components

#### Environments (`godelion_extended/environments/`)
- **GridWorld:** N agents in an N×N grid with partially observable state, configurable reward structures (distance-based, sparse, potential-based), and task types (individual goals, shared goals, mixed).

#### Agents (`godelion_extended/agents/`)
- **Policy Networks:** MLP and linear policies with configurable architectures, mutation, and crossover
- **AgentMemory:** Episodic memory buffer for state persistence across steps
- **Communication Channels:**
  - *VectorChannel:* Continuous low-dimensional message passing
  - *DiscreteChannel:* Learnable symbolic communication with softmax routing
  - *AttentionChannel:* Self-attention based message encoding

#### Universe Dialogue (`godelion_extended/universe_dialogue/`)
- Full multi-agent system with communication, memory, and analysis
- Supports individual, shared, and mixed objective configurations

#### RSI Evolution (`godelion_extended/evolution/`)
- **Selection:** Tournament, proportional, elite, and rank-based methods
- **Mutation:** Weight perturbation with adaptive rate control
- **Archive:** Full genealogical record of all generations

#### Meta-Evolution (`godelion_extended/rsi/`)
- **MetaLearner:** Population of evolution strategies that compete and evolve
- **Mutation Strategy Evolution:** The mutation rate, selection pressure, and crossover rate themselves undergo evolution
- **Adaptive strategies** that respond to fitness trends

#### Metrics (`godelion_extended/metrics/`)
- **Coordination:** Mutual information matrix, synchronization index, coordination score, redundancy ratio
- **Emergence:** Phase transition detection, cross-prediction error, shared representation score
- **Information Theory:** Entropy estimation, mutual information, KL divergence, total correlation

---

## 🧪 Experiments

### Experiment 1: Universe Dialogue

Tests the core hypothesis in a controlled multi-agent setting.

```bash
python run_experiment.py --experiment universe_dialogue --generations 50 --agents 10 --trials 3
```

**Measures:**
- Does coordination increase over generations?
- Does communication entropy decrease (protocol convergence)?
- Is there evidence of shared representations forming?
- Are phase transitions detectable in collective behavior?

### Experiment 2: RSI Evolution

Tests whether meta-evolution accelerates the emergence of coordination.

```bash
python run_experiment.py --experiment rsi_evolution --generations 100 --agents 50
```

**Measures:**
- Do evolved mutation strategies outperform fixed ones?
- Does meta-RSI lead to faster convergence to coordinated states?
- Are there cascading phase transitions as the meta-system optimizes?

### Experiment 3: Meta-RSI (Coming Soon)

Self-modifying mutation operators — the system evolves how it evolves.

---

## 📊 Metrics

### Individual Metrics
| Metric | Description |
|---|---|
| `agent_rewards` | Per-agent cumulative reward per episode |
| `fitness` | Evolutionary fitness (smoothed reward) |
| `communication_entropy` | Entropy of emitted communication signals |

### System Metrics
| Metric | Description |
|---|---|
| `total_system_reward` | Sum of all agent rewards |
| `convergence_rate` | Rate of fitness improvement over generations |
| `diversity` | Population variance in agent fitness |
| `reward_variance` | Inequality of reward distribution |

### Coordination Metrics
| Metric | Description |
|---|---|
| `mutual_information` | Pairwise MI between agent action sequences |
| `synchronization_index` | Fraction of actions where agents agree |
| `coordination_score` | Normalized measure of non-independence in actions |
| `redundancy_ratio` | Communication similarity ÷ action similarity |

### Emergence Indicators
| Metric | Description |
|---|---|
| `cross_prediction_error` | How well one agent's actions predict another's |
| `shared_representation_score` | Correlation in communication signal trajectories |
| `phase_transitions` | Detected change points in collective dynamics |
| `emergence_index` | Coordination gain ÷ communication bandwidth increase |

### Information-Theoretic Measures
| Metric | Description |
|---|---|
| `entropy` | Estimated entropy of arbitrary signals |
| `mutual_information` | Pairwise dependency quantification |
| `KL_divergence` | Distribution shift between agent behaviors |
| `total_correlation` | Multi-variate generalization of mutual information |

---

## ⚙️ Configuration

Experiments are configured through a hierarchical YAML system:

```yaml
# config.yaml (default)
experiment:
  name: "universe_dialogue"
  seed: 42
  output_dir: "./output"

universe_dialogue:
  n_agents: 6
  env_size: 10
  n_steps: 100
  n_generations: 50
  n_trials: 3
  agent:
    policy_hidden_dim: 16
    comm_dim: 4
    use_memory: true
    learn_communication: true
  communication:
    channel_type: "vector"  # vector | discrete | attention
  task:
    type: "individual"  # individual | shared | mixed
```

Create a `config.local.yaml` to override specific settings without modifying the default config.

Environment variables with prefix `UMIND_` override YAML settings:
```bash
export UMIND_UNIVERSE_DIALOGUE__N_AGENTS=20
export UMIND_UNIVERSE_DIALOGUE__N_GENERATIONS=100
```

---

## 📈 Visualization

All plots are automatically generated and saved to the run output directory:

```
output/universe_dialogue_20250101_120000/
├── plots/
│   ├── fitness_over_time.png
│   ├── coordination_metrics.png
│   ├── communication_analysis.png
│   ├── diversity.png
│   ├── mutual_information.png
│   ├── redundancy_vs_entropy.png
│   ├── phase_transitions.png
│   └── summary.png
├── experiment_report.txt
├── metrics.csv
├── all_metrics.json
├── config.json
└── archive.json
```

Post-hoc analysis:
```bash
python -m analysis.plot_results --run-dir ./output/universe_dialogue_20250101_120000
```

---

## 📁 Project Structure

```
UniversalMind/
├── run_experiment.py                    # Main entry point
├── config.yaml                          # Default configuration
├── config.local.yaml                    # Local overrides (gitignored)
├── pyproject.toml                       # Python packaging
├── requirements.txt                     # Dependencies
├── README.md                            # This file
├── LICENSE                              # Apache 2.0
│
├── godelion_extended/                   # Core framework package
│   ├── __init__.py                      # Package init
│   ├── config.py                        # Configuration loader
│   │
│   ├── environments/                    # Task environments
│   │   ├── grid_world.py                # GridWorld env
│   │   └── __init__.py
│   │
│   ├── agents/                          # Agent components
│   │   ├── policy.py                    # MLP/Linear policies
│   │   ├── memory.py                    # Agent memory
│   │   ├── communication.py            # Comm channels
│   │   └── __init__.py
│   │
│   ├── universe_dialogue/               # Multi-agent system
│   │   ├── environment.py               # Experiment config
│   │   ├── agent.py                     # Dialogue agents
│   │   ├── system.py                    # Multi-agent runner
│   │   ├── analysis.py                  # Episode analysis
│   │   └── __init__.py
│   │
│   ├── evolution/                       # Evolutionary engine
│   │   ├── selection.py                 # Parent/survivor selection
│   │   ├── mutation.py                  # Mutation & crossover
│   │   ├── archive.py                   # Generation archive
│   │   └── __init__.py
│   │
│   ├── rsi/                             # Meta-RSI
│   │   ├── meta_learner.py              # Meta-evolution
│   │   ├── mutation_strategies.py       # Adaptive mutation
│   │   └── __init__.py
│   │
│   ├── metrics/                         # Analysis metrics
│   │   ├── coordination.py              # Coordination measures
│   │   ├── emergence.py                 # Emergence detection
│   │   ├── information_theory.py        # Info-theoretic measures
│   │   └── __init__.py
│   │
│   ├── analytics/                       # Logging & viz
│   │   ├── logger.py                    # Experiment logger
│   │   ├── visualization.py             # Plot generation
│   │   ├── reporting.py                 # Text reports
│   │   └── __init__.py
│   │
│   └── experiments/                     # Runnable experiments
│       ├── base.py                      # Abstract base
│       ├── universe_dialogue.py         # Universe Dialogue
│       ├── rsi_evolution.py             # RSI Evolution
│       └── __init__.py
│
├── analysis/                            # Post-hoc analysis
│   ├── plot_results.py                  # Result plotting
│   └── __init__.py
│
├── tests/                               # Test suite
│   ├── test_environment.py              # Environment tests
│   ├── test_agents.py                   # Agent tests
│   ├── test_universe_dialogue.py        # System tests
│   ├── test_evolution.py                # Evolution tests
│   └── test_metrics.py                  # Metrics tests
│
└── docs/                                # Documentation
    └── architecture.md                  # Detailed architecture
```

---

## 🔬 Research Background

### Foundational Work

This framework builds on and extends several research directions:

- **Godelion** ([arXiv:2505.22954](https://arxiv.org/abs/2505.22954)): Open-ended evolution of self-improving coding agents — the evolutionary RSI foundation
- **Distributed Cognition** (Hutchins, 1995): Cognitive processes distributed across agents and environment
- **Global Workspace Theory** (Baars, 1988): Consciousness as global information integration
- **Integrated Information Theory** (Tononi, 2004): Φ as a measure of irreducible causal interactions
- **Multi-Agent Reinforcement Learning**: Independent learning with shared experience
- **Evolutionary Game Theory**: Emergence of cooperation and coordination

### Key References

1. Clune, J. (2019). AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence. *arXiv:1905.10985*.
2. Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*.
3. Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*.
4. Schmidhuber, J. (2009). Ultimate cognition à la Gödel. *Cognitive Computation*.

### The Falsification Protocol

The framework implements the following empirical protocol:

1. Run baseline with **no communication** — measure baseline coordination
2. Run with **limited bandwidth communication** — measure coordination ceiling
3. Run with **full communication** — measure actual coordination
4. If actual coordination significantly exceeds the communication-bounded ceiling → **H2 supported**
5. If coordination scales linearly with communication bandwidth → **H1 supported**
6. Track across generations: does the gap grow? This indicates emergent structure.

---

## 🧩 Extending the Framework

### Adding a New Agent Type

```python
from godelion_extended.agents.policy import MLPPolicy
from godelion_extended.universe_dialogue.agent import DialogueAgent

class MyCustomAgent(DialogueAgent):
    def act(self, obs, comm_input=None):
        # Custom decision logic
        return action, comm_state
```

### Adding a New Metric

```python
from godelion_extended.metrics.coordination import compute_coordination_score

def my_custom_metric(action_matrix):
    # Custom computation
    return score
```

### Adding a New Environment

```python
from godelion_extended.environments.grid_world import GridWorld

class ContinuousEnvironment:
    def reset(self):
        pass
    def step(self, actions):
        pass
    def observe(self):
        pass
```

### Adding a New Experiment

```python
from godelion_extended.experiments.base import BaseExperiment

class MyExperiment(BaseExperiment):
    def setup(self):
        pass
    def run_generation(self, generation):
        return metrics
    def evaluate(self):
        return final_metrics
```

---

## 🛡️ License

Apache 2.0. See [LICENSE](./LICENSE) for details.

---

<p align="center">
  <strong>Built with 🔬 for open empirical research into emergent coordination in recursive self-improving systems.</strong>
</p>
