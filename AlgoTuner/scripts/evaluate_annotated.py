#!/usr/bin/env python3
"""
Evaluate annotated dataset for a task and produce a minimal report for each run.
Usage:
  python3 scripts/evaluate_annotated.py TASK_NAME --data-dir DATA_DIR --run-id RUN_ID [--subset train|test]
"""

import argparse
import json
from pathlib import Path
import itertools
from AlgoTuneTasks.factory import TaskFactory
from AlgoTuneTasks.base import load_dataset_streaming
from AlgoTuner.utils.evaluator.main import evaluate_problems
from AlgoTuner.config.loader import load_config  # for default test runs/warmups


def main():
    parser = argparse.ArgumentParser(description="Evaluate annotated dataset for a task.")
    parser.add_argument("task_name", help="Name of the registered task to evaluate")
    parser.add_argument("--data-dir", required=True, help="Directory where dataset JSONL files are stored")
    parser.add_argument("--run-id", type=int, required=True, help="Identifier for this evaluation run")
    parser.add_argument("--subset", choices=["train", "test"], default="train",
                        help="Dataset subset to evaluate (train or test)")
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Number of runs per instance (defaults from config)")
    parser.add_argument("--warmup-runs", type=int, default=None,
                        help="Number of warmup runs per instance (defaults from config)")
    args = parser.parse_args()

    task = TaskFactory(args.task_name)
    dataset_dir = Path(args.data_dir) / args.task_name
    pattern = f"{args.subset}_*.jsonl"
    generators = []
    for jpath in sorted(dataset_dir.glob(pattern)):
        generators.append(load_dataset_streaming(str(jpath)))
    if not generators:
        print(json.dumps({
            "task_name": args.task_name,
            "run_id": args.run_id,
            "status": "NO_DATA",
        }))
        return

    # Load benchmark defaults from config
    cfg = load_config()
    bench_cfg = cfg.get('benchmark', {})
    default_runs = bench_cfg.get('test_runs', 5)
    default_warmups = bench_cfg.get('test_warmups', 3)

    # Determine runs/warmups (CLI or config defaults)
    num_runs = args.num_runs if args.num_runs is not None else default_runs
    warmup_runs = args.warmup_runs if args.warmup_runs is not None else default_warmups

    # Chain generators together
    all_records_iterable = itertools.chain(*generators)

    # Run the shared solve-loop, passing the iterable and unpacking the count
    per_problem, aggregate, timing_report, num_evaluated = evaluate_problems(
        task,
        all_records_iterable,
        num_runs,
        warmup_runs
    )

    report = {
        "task_name": args.task_name,
        "run_id": args.run_id,
        "status": "OK",
        "num_instances": num_evaluated,
        "avg_solve_ms": aggregate.get("avg_solver_time_ms"),
        "avg_oracle_ms": aggregate.get("avg_oracle_time_ms"),
    }
    print(json.dumps(report))

if __name__ == "__main__":
    main() 