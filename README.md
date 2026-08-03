# MAPF-warehouse

How to run simulation?
```bash
uv run python -m src.main
```

How to train models?

Configure models, view sizes and number of robots in `src.utils.experiment_runner.py`

then run:
```bash
uv run python -m src.utils.experiment_runner
```

To compare results configure `src.utils.experiment_plotter.py`

then run:
```bash
uv run python -m src.utils.experiment_plotter
```

Ideas:
Train model on larger amount of robots, after training results might be better.
Fine tuning with another number of robots might be needed.