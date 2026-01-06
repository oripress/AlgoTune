#!/usr/bin/env python3
"""
Prepare retry for tasks with N/A results for a specific model.
Deletes local logs/outputs/errors for those task+model entries and removes the
model entry from reports/agent_summary.json. Outputs tasks to retry (one per line).
"""

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare retry for N/A tasks.")
    parser.add_argument("--model", required=True, help="Full model name in summary.json")
    parser.add_argument(
        "--model-display", required=True, help="Model display name in log filenames"
    )
    parser.add_argument("--logs-dir", required=True, help="Local logs directory")
    parser.add_argument("--job-map", required=True, help="Path to reports/job_map.json")
    parser.add_argument("--summary-file", required=True, help="Path to reports/agent_summary.json")
    parser.add_argument("--out", required=True, help="Output file for tasks to retry")
    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    summary = load_json(summary_path)
    tasks_to_retry = []

    for task_name, models in summary.items():
        if not isinstance(models, dict):
            continue
        metrics = models.get(args.model)
        if not isinstance(metrics, dict):
            continue
        speedup = metrics.get("final_speedup")
        if speedup is None or speedup == "N/A":
            tasks_to_retry.append(task_name)

    if not tasks_to_retry:
        Path(args.out).write_text("")
        return 0

    # Delete local logs for the model display name
    logs_dir = Path(args.logs_dir)
    if logs_dir.exists():
        for log_path in logs_dir.glob("*.log"):
            name = log_path.name
            if (
                name.startswith(tuple(f"{task}_" for task in tasks_to_retry))
                and args.model_display in name
            ):
                try:
                    log_path.unlink()
                except Exception:
                    pass

    # Delete outputs/errors for matching job_map entries
    job_map_path = Path(args.job_map)
    job_map = load_json(job_map_path)
    outputs_dir = Path(args.summary_file).parent.parent / "aws" / "outputs"
    errors_dir = Path(args.summary_file).parent.parent / "aws" / "errors"

    for job_id, meta in job_map.items():
        if not isinstance(meta, dict):
            continue
        task = meta.get("task")
        model = meta.get("model")
        if task in tasks_to_retry and model == args.model_display:
            out_path = outputs_dir / f"batch-{job_id}.out"
            err_path = errors_dir / f"batch-{job_id}.err"
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            if err_path.exists():
                try:
                    err_path.unlink()
                except Exception:
                    pass

    # Remove model entries from summary for retry tasks
    for task in tasks_to_retry:
        models = summary.get(task)
        if isinstance(models, dict) and args.model in models:
            del models[args.model]
            if not models:
                summary.pop(task, None)

    save_json(summary_path, summary)

    Path(args.out).write_text("\n".join(tasks_to_retry) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
