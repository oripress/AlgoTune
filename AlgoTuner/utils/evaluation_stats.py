import statistics
from typing import Any


def calculate_evaluation_stats(
    time_ratios: list[float],
    solution_diffs: list[float],
    llm_times: list[int],
    optimal_times: list[int],
    total_problems: int,
    optimal_count: int,
    non_optimal_count: int,
    timeout_count: int,
) -> dict[str, Any]:
    """Calculate comprehensive statistics for evaluation results."""
    # Calculate actual average times without penalties
    actual_llm_time = statistics.mean(llm_times) if llm_times else 0
    actual_optimal_time = statistics.mean(optimal_times) if optimal_times else 0

    # Calculate performance ratio without penalties
    actual_time_ratios = [min(ratio, 10.0) for ratio in time_ratios]  # Cap at 10x for stats
    actual_mean_ratio = statistics.mean(actual_time_ratios) if actual_time_ratios else 0

    stats = {
        "total_problems": total_problems,
        "optimal_solutions": optimal_count,
        "non_optimal_solutions": non_optimal_count,
        "timeouts": timeout_count,
        "optimal_percentage": (optimal_count / total_problems) * 100,
        "non_optimal_percentage": (non_optimal_count / total_problems) * 100,
        "timeout_percentage": (timeout_count / total_problems) * 100,
        "mean_llm_time": actual_llm_time,  # Actual average time
        "median_llm_time": statistics.median(llm_times) if llm_times else None,
        "mean_optimal_time": actual_optimal_time,  # Actual optimal time
        "median_optimal_time": statistics.median(optimal_times) if optimal_times else None,
        "actual_time_ratio": actual_mean_ratio,  # Actual average ratio without penalties
    }
    return stats


def calculate_aggregate_metrics(
    time_ratios: list[float], solution_diffs: list[float]
) -> tuple[float, float | None]:
    """Calculate aggregate metrics from evaluation results."""
    # For scoring purposes, keep the 10x penalty
    mean_time_ratio = sum(time_ratios) / len(time_ratios) if time_ratios else 0
    mean_solution_diff = sum(solution_diffs) / len(solution_diffs) if solution_diffs else None
    return mean_time_ratio, mean_solution_diff


def check_all_timeouts(time_ratios: list[float]) -> bool:
    """Check if all problems timed out."""
    return all(ratio >= 10.0 for ratio in time_ratios)


def create_evaluation_result(
    success: bool,
    time_ratios: list[float],
    solution_diffs: list[float],
    stats: dict[str, Any],
    last_output_logs: dict[str, Any],
    command_source: str | None = None,
    error: str | None = None,
    traceback: str | None = None,
) -> dict[str, Any]:
    """Create a standardized evaluation result dictionary."""
    if not success:
        return {
            "success": False,
            "error": error,
            "traceback": traceback,
            "output_logs": error or "",
            "command_source": command_source,
        }

    mean_time_ratio, mean_solution_diff = calculate_aggregate_metrics(time_ratios, solution_diffs)
    all_timeouts = check_all_timeouts(time_ratios)

    return {
        "success": True,
        "average_time_ratio": mean_time_ratio,
        "average_solution_diff": mean_solution_diff,
        "total_evaluated": len(time_ratios),
        "perfect_score": (mean_solution_diff == 0 and mean_time_ratio <= 1.0),
        "stats": stats,
        "output_logs": last_output_logs,
        "all_timeouts": all_timeouts,
        "command_source": command_source,
    }


def create_timeout_stats(
    elapsed: int,
    optimal_time: int,
) -> dict[str, Any]:
    """Create statistics for timeout case."""
    return {
        "total_problems": 1,
        "optimal_solutions": 0,
        "non_optimal_solutions": 0,
        "timeouts": 1,
        "optimal_percentage": 0,
        "non_optimal_percentage": 0,
        "timeout_percentage": 100,
        "mean_llm_time": elapsed,
        "median_llm_time": elapsed,
        "mean_optimal_time": optimal_time,
        "median_optimal_time": optimal_time,
    }
