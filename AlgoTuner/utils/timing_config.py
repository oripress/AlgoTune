from AlgoTuner.config.loader import load_config

"""Shared timing configuration.
This centralises the default numbers of warm-up iterations, measurement
iterations, and the warm-up-to-timeout multiplier so they can be changed in
one place.
"""

_cfg = load_config()
_bench = _cfg.get("benchmark", {})

# Number of timed measurements per benchmark
RUNS: int = _bench.get("runs", 10)

# Number of un-timed warm-up iterations run before the measurements
WARMUPS: int = 1  # Fixed at 1 warmup per subprocess

# Dataset evaluation specific runs
DEV_RUNS: int = _bench.get("dev_runs", 2)  # For training dataset
EVAL_RUNS: int = _bench.get("eval_runs", 10)  # For test dataset

# Legacy aliases for compatibility
DATASET_RUNS: int = EVAL_RUNS  # Default to eval_runs
DATASET_WARMUPS: int = WARMUPS

# Factor by which warm-up time contributes to the timeout heuristic.
WARMUP_MULTIPLIER: float = 5.0 