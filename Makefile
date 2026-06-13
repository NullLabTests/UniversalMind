.PHONY: install test run quick clean

# Install package in development mode
install:
	pip install -e .

# Run all tests
test:
	python -m pytest tests/ -v

# Run default experiment
run:
	python run_experiment.py

# Quick test (2 generations, 2 agents)
quick:
	python run_experiment.py --quick

# Run with custom config
run-config:
	python run_experiment.py --config config.local.yaml

# Clean cache files
clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null
	rm -rf output/
