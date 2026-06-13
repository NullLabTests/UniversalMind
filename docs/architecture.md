# Architecture

## Package Structure

```
godelion_extended/
├── config.py              # YAML-based configuration loader
├── environments/          # Task environments (GridWorld)
├── agents/                # Neural policies, memory, comm channels
├── universe_dialogue/     # Multi-agent system with analysis
├── evolution/             # Darwinian evolution engine
├── rsi/                   # Meta-evolution (evolving evolution)
├── metrics/               # Coordination, emergence, info theory
├── analytics/             # Logging, visualization, reporting
└── experiments/           # Runnable experiment harnesses
```

## Data Flow

```
config.yaml → Experiment → Evolution Engine → Agent Population
                   ↓                              ↓
              Metrics Layer ←─── Simulation ──────┘
                   ↓
         Analytics (logs, plots, report)
```

## Key Design Decisions

1. **No external ML framework** — Pure NumPy keeps dependencies minimal
2. **Deterministic by default** — Seeded RNG everywhere
3. **Configuration-driven** — Every parameter in YAML
4. **Metrics-first** — Every simulation step produces analyzable data
5. **Evolutionary archive** — Full genealogy preserved across generations
