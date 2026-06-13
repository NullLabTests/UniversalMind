# Contributing to UniversalMind

## Getting Started

```bash
git clone https://github.com/NullLabTests/UniversalMind.git
cd UniversalMind
pip install -e ".[dev]"
pytest tests/ -v
```

## Development Guidelines

1. **Modularity**: Each component belongs in its own module under `godelion_extended/`
2. **Tests**: Every new feature must include tests in `tests/`
3. **Reproducibility**: All experiments must be seed-controllable
4. **Metrics**: New metrics should go in `godelion_extended/metrics/`
5. **Config**: New parameters need defaults in `config.yaml`

## Adding a New Experiment

Create a class inheriting from `BaseExperiment` in `godelion_extended/experiments/`:

```python
from godelion_extended.experiments.base import BaseExperiment

class MyExperiment(BaseExperiment):
    def setup(self): ...
    def run_generation(self, generation): ...
    def evaluate(self): ...
```

## Code Style

- Python 3.10+ type hints
- Line length: 140
- Follow existing patterns in the codebase
