#!/usr/bin/env python3
"""
Run two identical dataset annotation phases and report annotation, reannotation times and their average difference.
Usage:
  python3 scripts/benchmark_two_phases.py TASK_NAME --data-dir DATA_DIR [--num-runs R] [--warmup-runs W]
"""

import argparse
import json
import logging
from pathlib import Path

from AlgoTuneTasks.factory import TaskFactory
from AlgoTuneTasks.base import load_dataset_streaming
from AlgoTuner.config.loader import load_config
from AlgoTuner.utils.evaluator.runner import run_oracle_evaluation


def run_single_phase(task_name: str, data_dir: str, num_runs: int, warmup_runs: int):
    # Load dataset configuration defaults
    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})
    train_size = ds_cfg.get("train_size", 100)
    test_size  = ds_cfg.get("test_size", 100)
    seed       = ds_cfg.get("random_seed", 42)

    # Initialize task and generate dataset
    task = TaskFactory(task_name, oracle_time_limit=None)
    task.load_dataset(
        train_size=train_size,
        test_size=test_size,
        random_seed=seed,
        data_dir=data_dir
    )
    dataset_dir = Path(data_dir) / task_name

    # Phase 1: annotation (baseline)
    annotation_times = []
    for subset in ("train", "test"):
        for jpath in sorted(dataset_dir.glob(f"{subset}_*.jsonl")):
            for rec in load_dataset_streaming(str(jpath)):
                res = run_oracle_evaluation(
                    problem=rec["problem"],
                    task_instance=task,
                    capture_output=False,
                    needs_casting=True,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs
                )
                annotation_times.append(res.get("elapsed_ms", 0))
    annotation_avg = sum(annotation_times) / len(annotation_times) if annotation_times else 0.0

    # Phase 2: reannotation (reload and re-run annotation identical to Phase 1)
    reannotation_times = []
    for subset in ("train", "test"):
        for jpath in sorted(dataset_dir.glob(f"{subset}_*.jsonl")):
            for rec in load_dataset_streaming(str(jpath)):
                res = run_oracle_evaluation(
                    problem=rec["problem"],
                    task_instance=task,
                    capture_output=False,
                    needs_casting=True,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs
                )
                reannotation_times.append(res.get("elapsed_ms", 0))
    reannotation_avg = sum(reannotation_times) / len(reannotation_times) if reannotation_times else 0.0

    return annotation_avg, reannotation_avg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark two identical dataset annotation phases.")
    parser.add_argument("task_name", help="Name of the task to benchmark")
    parser.add_argument("--data-dir", required=True, help="Directory where datasets are stored")
    parser.add_argument("--num-runs",    type=int, default=5, help="Number of runs per instance for timing")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs per instance")
    args = parser.parse_args()

    # Set up file logging to tests/logs
    log_dir = Path("tests") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    python_log = log_dir / f"{args.task_name}_benchmark_python.log"
    file_handler = logging.FileHandler(python_log, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Detailed logs will be written to {python_log}")

    # Run Phase 0
    ann0, reann0 = run_single_phase(args.task_name, args.data_dir, args.num_runs, args.warmup_runs)
    logging.info(f"Phase 0: annotation_avg_ms={ann0:.3f}, reannotation_avg_ms={reann0:.3f}")

    # Run Phase 1
    ann1, reann1 = run_single_phase(args.task_name, args.data_dir, args.num_runs, args.warmup_runs)
    logging.info(f"Phase 1: annotation_avg_ms={ann1:.3f}, reannotation_avg_ms={reann1:.3f}")

    # Compute average difference
    diff0 = reann0 - ann0
    diff1 = reann1 - ann1
    avg_diff = (diff0 + diff1) / 2.0

    # Prepare summary
    summary = {
        "task_name": args.task_name,
        "phase0_annotation_avg_ms": ann0,
        "phase0_reannotation_avg_ms": reann0,
        "phase1_annotation_avg_ms": ann1,
        "phase1_reannotation_avg_ms": reann1,
        "avg_diff_ms": avg_diff
    }

    # Write summary JSON
    report_dir = Path("tests") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"result_{args.task_name}_two_phase.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main() 