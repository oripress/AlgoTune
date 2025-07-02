#!/usr/bin/env python3
"""
Aggregate baseline and eval JSON reports into a concise summary.
Usage: python3 scripts/aggregate_task_statuses.py <reports_dir> <output_summary_json>
"""

import sys
import os
import glob
import json


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 aggregate_task_statuses.py <reports_dir> <output_summary_json>", file=sys.stderr)
        sys.exit(1)
    report_dir = sys.argv[1]
    output_file = sys.argv[2]
    if not os.path.isdir(report_dir):
        print(f"Error: Report directory not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all result_*.json files
    pattern = os.path.join(report_dir, 'result_*.json')
    result_files = glob.glob(pattern)

    # Group by task_name
    by_task = {}
    for rf in result_files:
        try:
            data = json.load(open(rf))
            task = data.get('task_name')
            run_id = data.get('run_id')
            if task is None or run_id is None:
                continue
            by_task.setdefault(task, {})[run_id] = data
        except Exception as e:
            continue

    # Build summary list
    summary = []
    for task in sorted(by_task.keys()):
        runs = by_task[task]
        base = runs.get(0, {})
        e1 = runs.get(1, {})
        e2 = runs.get(2, {})
        status = base.get('status', 'NO_DATA')
        baseline_avg = base.get('baseline_avg_ms')
        eval1_avg = e1.get('avg_solve_ms')
        eval2_avg = e2.get('avg_solve_ms')
        # Compute pairwise differences
        diff_1_2 = None
        if eval1_avg is not None and eval2_avg is not None:
            diff_1_2 = abs(eval2_avg - eval1_avg)
        diff_1_0 = None
        if eval1_avg is not None and baseline_avg is not None:
            diff_1_0 = abs(eval1_avg - baseline_avg)
        diff_2_0 = None
        if eval2_avg is not None and baseline_avg is not None:
            diff_2_0 = abs(eval2_avg - baseline_avg)
        summary.append({
            'task': task,
            'status': status,
            'baseline_avg_ms': baseline_avg,
            'eval1_avg_ms': eval1_avg,
            'eval2_avg_ms': eval2_avg,
            'diff_1_2_ms': diff_1_2,
            'diff_1_0_ms': diff_1_0,
            'diff_2_0_ms': diff_2_0
        })

    # Write summary to output file
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main() 