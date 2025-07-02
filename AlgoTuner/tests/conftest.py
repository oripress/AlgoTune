import json
import os

# Pytest hooks to collect test outcomes and write global summary

# Module-level summary storage
_summary_results = []

def pytest_configure(config):
    """Initialize storage for summary results."""
    global _summary_results
    _summary_results.clear()


def pytest_runtest_logreport(report):
    """Collect the outcome of each test 'call' phase."""
    if report.when != 'call':
        return
    nodeid = report.nodeid
    # Only collect dataset generation tests
    if 'test_generate_save_load_consistency' in nodeid:
        # Extract the parameterized task name
        if '[' in nodeid and ']' in nodeid:
            task_name = nodeid.split('[', 1)[1].rstrip(']')
        else:
            task_name = None
        summary = {'task_name': task_name, 'outcome': report.outcome}
        if report.outcome == 'failed':
            # Include the error message or traceback for failed dataset creation
            error_msg = getattr(report, 'longreprtext', str(report.longrepr))
            summary['error'] = error_msg
        # Include duration if available
        duration = getattr(report, 'duration', None)
        if duration is not None:
            summary['duration'] = duration
        _summary_results.append(summary)


def pytest_sessionfinish(session, exitstatus):
    """Aggregate per-task JSON reports into a summary with requested metrics."""
    import glob
    # Determine where to write the summary
    summary_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.environ.get('SUMMARY_FILE', os.path.join(summary_dir, 'summary.json'))

    # Read individual per-task report files
    report_files = glob.glob(os.path.join(summary_dir, 'result_*.json'))
    tasks_summary = []
    success_count = 0
    na_count = 0
    bad_dataset_count = 0
    for rf in report_files:
        # Initialize abs_diff_target to default to avoid UnboundLocalError
        abs_diff_target = 'N/A'
        try:
            with open(rf) as f:
                rd = json.load(f)
        except Exception:
            continue
        task_name = rd.get('task_name')
        # Determine task status, defaulting to SUCCESS
        status = rd.get('status', 'SUCCESS')
        if status == 'FAILED':
            run_diff = 'N/A'
            loaded_diff = 'N/A'
            run_diff_median = 'N/A'
            loaded_diff_median = 'N/A'
            na_count += 1
        elif status == 'BAD_DATASET':
            run_diff = 'N/A'
            loaded_diff = 'N/A'
            run_diff_median = 'N/A'
            loaded_diff_median = 'N/A'
            bad_dataset_count += 1
        else:  # SUCCESS
            r1 = rd.get('run1_avg_time_ms')
            r2 = rd.get('run2_avg_time_ms')
            lr1 = rd.get('run1_loaded_solve_avg_time_ms')
            run_diff = abs(r1 - r2) if isinstance(r1, (int, float)) and isinstance(r2, (int, float)) else 'N/A'
            loaded_diff = abs(lr1 - r1) if isinstance(lr1, (int, float)) and isinstance(r1, (int, float)) else 'N/A'
            # Median-based diffs
            m1 = rd.get('run1_median_time_ms')
            m2 = rd.get('run2_median_time_ms')
            lm1 = rd.get('run1_loaded_solve_median_time_ms')
            run_diff_median = abs(m1 - m2) if isinstance(m1, (int, float)) and isinstance(m2, (int, float)) else 'N/A'
            loaded_diff_median = abs(lm1 - m1) if isinstance(lm1, (int, float)) and isinstance(m1, (int, float)) else 'N/A'
            # Extract the absolute difference between run1 average and target time
            abs_diff_target = rd.get('abs_diff_target_run1_avg_ms', 'N/A')
            success_count += 1
        tasks_summary.append({
            'task_name': task_name,
            'status': status,
            'run_diff_avg_time_ms': run_diff,
            'loaded_diff_vs_run1_ms': loaded_diff,
            'run_diff_median_ms': run_diff_median,
            'loaded_diff_vs_run1_median_ms': loaded_diff_median,
            'abs_diff_target_run1_avg_ms': abs_diff_target
        })
    # Build final summary
    summary = {
        'exitstatus': exitstatus,
        'num_success': success_count,
        'num_na': na_count,
        'num_bad_dataset': bad_dataset_count,
        'tasks': tasks_summary
    }
    # Write summary to JSON
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2) 