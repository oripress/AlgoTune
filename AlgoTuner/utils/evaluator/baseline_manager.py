"""
Baseline Manager for handling baseline time generation and caching.
"""

import glob
import logging
import os
import threading
from typing import Any

from AlgoTuner.utils.timing_config import DEV_RUNS, EVAL_RUNS


class BaselineManager:
    """
    Manages baseline times for evaluation tasks.

    Provides a unified interface for:
    - Generating baseline times when needed
    - Caching baseline times in memory
    - Thread-safe access to baseline times
    - Automatic regeneration when cache is incomplete
    """

    def __init__(self, task_instance: Any):
        """
        Initialize the BaselineManager.

        Args:
            task_instance: The task instance to manage baselines for
        """
        self.task_instance = task_instance
        self._cache = {"train": None, "test": None}
        self._lock = threading.Lock()

    def get_baseline_times(
        self,
        subset: str,
        force_regenerate: bool = False,
        test_mode: bool = False,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """
        Get baseline times for a dataset subset, generating if needed.
        Retries up to MAX_RETRIES times on incomplete results.
        Requires 100% of baselines to be generated - no tolerance for missing items.

        Args:
            subset: 'train' or 'test'
            force_regenerate: Force regeneration even if cached
            test_mode: Whether in test mode
            max_samples: Maximum samples to process (for test mode)

        Returns:
            Dictionary mapping problem IDs to baseline times in milliseconds
        """
        MAX_RETRIES = 3

        with self._lock:
            # Check cache
            if not force_regenerate and self._cache[subset] is not None:
                logging.info(
                    f"Using cached baseline times for {subset}: {len(self._cache[subset])} entries"
                )
                return self._cache[subset]

            # Retry loop for baseline generation
            for attempt in range(MAX_RETRIES):
                logging.info(
                    f"Generating baseline times for subset '{subset}' (attempt {attempt + 1}/{MAX_RETRIES})"
                )

                # Progressive timeout increase on retries
                # Attempt 1: 60s, Attempt 2: 90s, Attempt 3: 120s
                timeout_multiplier = 1.0 + (attempt * 0.5)
                logging.info(f"Using timeout multiplier {timeout_multiplier:.1f}x for attempt {attempt + 1}")

                # Generate baseline times
                baseline_times = self._generate_baseline_times(
                    subset, test_mode, max_samples, timeout_multiplier=timeout_multiplier
                )

                # Determine expected count (need to load dataset to know size)
                expected_count = self._get_expected_baseline_count(subset, test_mode, max_samples)
                actual_count = len(baseline_times)

                logging.info(
                    f"Baseline generation attempt {attempt + 1}: {actual_count}/{expected_count} items"
                )

                # Check if results are complete - require 100% match
                if actual_count == expected_count:
                    # Success - cache and return
                    self._cache[subset] = baseline_times
                    logging.info(f"Successfully generated all {actual_count} baseline times")
                    logging.info("BaselineManager.get_baseline_times completed, returning to caller")
                    return baseline_times

                # Incomplete results
                if attempt < MAX_RETRIES - 1:
                    # Not the last attempt - clear cache and retry
                    missing_count = expected_count - actual_count
                    logging.warning(
                        f"Incomplete baseline generation ({actual_count}/{expected_count}, missing {missing_count}). "
                        f"Clearing cache and retrying (attempt {attempt + 2}/{MAX_RETRIES})..."
                    )

                    # Force garbage collection to free memory
                    import gc
                    import time

                    gc.collect()
                    logging.info("Forced garbage collection before retry")

                    self.clear_cache(subset)

                    # Longer delay to allow system resources to settle
                    time.sleep(5)
                else:
                    # Last attempt failed after 3 retries - this is a FATAL SYSTEM ERROR
                    missing_count = expected_count - actual_count
                    error_msg = (
                        f"FATAL SYSTEM ERROR: BASELINE GENERATION FAILED after {MAX_RETRIES} attempts. "
                        f"Only {actual_count}/{expected_count} baseline times generated (missing {missing_count} items). "
                        f"This indicates oracle solver is too slow or task is misconfigured. "
                        f"This is a system/infrastructure failure - job cannot continue."
                    )
                    logging.critical(error_msg)
                    # Exit immediately - this is not recoverable by the agent
                    raise SystemExit(1)

            # Should never reach here
            raise SystemExit(1)

    def _get_expected_baseline_count(
        self, subset: str, test_mode: bool, max_samples: int | None
    ) -> int:
        """
        Determine the expected number of baseline entries based on dataset size.

        Args:
            subset: 'train' or 'test'
            test_mode: Whether in test mode
            max_samples: Maximum samples to process

        Returns:
            Expected count of baseline times
        """
        from AlgoTuner.utils.evaluator.main import stream_jsonl
        import os
        import glob

        # Check for JSONL files first (memory efficient)
        data_dir = self.task_instance.data_dir or self.task_instance.get_task_directory()
        jsonl_pattern = os.path.join(
            data_dir, f"{self.task_instance.task_name}_T*ms_n*_size*_{subset}.jsonl"
        )
        jsonl_files = glob.glob(jsonl_pattern)

        if jsonl_files:
            # Count lines in JSONL file
            jsonl_path = jsonl_files[0]
            dataset_iter = stream_jsonl(jsonl_path)
            dataset_list = list(dataset_iter)
        else:
            # Load dataset and count
            train_iter, test_iter = self.task_instance.load_dataset()
            dataset_iter = train_iter if subset == "train" else test_iter
            dataset_list = list(dataset_iter)

        expected_count = len(dataset_list)

        # Apply test mode filtering
        if test_mode and max_samples:
            expected_count = min(expected_count, max_samples)

        return expected_count

    def _generate_baseline_times(
        self, subset: str, test_mode: bool, max_samples: int | None, timeout_multiplier: float = 1.0
    ) -> dict[str, float]:
        """
        Generate baseline times by running the oracle solver.

        Args:
            subset: 'train' or 'test'
            test_mode: Whether in test mode
            max_samples: Maximum samples to process
            timeout_multiplier: Multiplier for timeout (1.0 = 60s, 1.5 = 90s, 2.0 = 120s)

        Returns:
            Dictionary mapping problem IDs to baseline times
        """
        # Determine number of runs based on subset
        num_runs = DEV_RUNS if subset == "train" else EVAL_RUNS

        # Check for JSONL files first (memory efficient)
        data_dir = self.task_instance.data_dir or self.task_instance.get_task_directory()
        jsonl_pattern = os.path.join(
            data_dir, f"{self.task_instance.task_name}_T*ms_n*_size*_{subset}.jsonl"
        )
        jsonl_files = glob.glob(jsonl_pattern)

        # Run baseline evaluation directly and get results in memory
        from AlgoTuner.utils.evaluator.main import stream_jsonl

        if jsonl_files:
            # Use JSONL file for memory efficiency
            jsonl_path = jsonl_files[0]
            logging.info(f"Using JSONL file for baseline generation: {jsonl_path}")
            dataset_iter = stream_jsonl(jsonl_path)
        else:
            # Load dataset and generate baselines
            logging.info("Loading dataset for baseline generation")
            train_iter, test_iter = self.task_instance.load_dataset()
            dataset_iter = train_iter if subset == "train" else test_iter

        # Convert iterator to list to enable using different problems for warmup
        dataset_list = list(dataset_iter)
        if test_mode and max_samples:
            dataset_list = dataset_list[:max_samples]

        # Run oracle evaluation to generate baseline times
        logging.info("Running oracle evaluation to generate baseline times")

        # Import required functions

        from AlgoTuner.utils.benchmark import run_benchmark
        from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark

        # Get the oracle solver function
        oracle_solver = getattr(self.task_instance, "solve", None)
        if not oracle_solver:
            logging.error("BaselineManager: Task instance has no solve method")
            return {}

        # Check if we should use isolated execution (matches solver evaluation logic)
        agent_mode = os.environ.get("AGENT_MODE", "0")
        use_isolated = agent_mode != "0" or os.environ.get("ISOLATED_EVAL", "1") == "1"

        # Process each problem and collect baseline times
        baseline_times = {}

        problem_count = len(dataset_list)
        for i, item in enumerate(dataset_list):
            # Extract problem ID the same way as evaluation orchestrator
            if isinstance(item, dict):
                # Build metadata exactly like evaluation_orchestrator._extract_problem_data (line 300)
                # Use seed as ID if available (it's the unique identifier in datasets)
                metadata = {
                    "id": item.get("id", item.get("seed", item.get("k", None))),
                    "baseline_time_ms": item.get("baseline_time_ms"),
                    "baseline_time_us": item.get("baseline_time_us"),
                }
                problem_data = item.get("problem", item)
            else:
                metadata = {"id": None, "baseline_time_ms": None, "baseline_time_us": None}
                problem_data = item

            # Use same ID extraction logic as evaluation_orchestrator.py:90
            problem_id = metadata.get("id", f"problem_{i + 1}")

            # Ensure problem_id is a string for consistent dictionary lookups
            if problem_id is not None:
                problem_id = str(problem_id)

            # Get warmup problem (use next problem in dataset, wrapping around)
            warmup_idx = (i + 1) % problem_count
            warmup_item = dataset_list[warmup_idx]
            warmup_problem_data = (
                warmup_item.get("problem", warmup_item)
                if isinstance(warmup_item, dict)
                else warmup_item
            )

            # Log whether problems are different
            if i > 0:
                logging.debug(
                    f"BaselineManager: Problem {problem_id} using different warmup problem (index {warmup_idx})"
                )
            else:
                logging.debug(
                    f"BaselineManager: Problem {problem_id} using same problem for warmup (first problem)"
                )

            # Run oracle solver with appropriate execution method
            try:
                logging.info(f"BaselineManager: Running oracle benchmark for {problem_id}")

                if use_isolated:
                    # Use isolated benchmark with different problems for warmup and timing
                    logging.debug(f"BaselineManager: Using isolated benchmark for {problem_id}")

                    # Get task directory and name for isolated benchmark
                    task_name = getattr(self.task_instance, "task_name", "unknown_task")
                    code_dir = self.task_instance.get_task_directory()

                    # Calculate timeout based on target time if available
                    timeout_seconds = 60.0  # Default
                    if (
                        hasattr(self.task_instance, "target_time_ms")
                        and self.task_instance.target_time_ms
                    ):
                        target_time_s = self.task_instance.target_time_ms / 1000.0
                        timeout_seconds = max(
                            60.0, target_time_s * 10.0
                        )  # 10x target time, minimum 60s

                    # Apply timeout multiplier for retries
                    timeout_seconds *= timeout_multiplier

                    benchmark_result = run_isolated_benchmark(
                        task_name=task_name,
                        code_dir=code_dir,
                        warmup_problem=warmup_problem_data,
                        timed_problem=problem_data,
                        num_runs=num_runs,
                        timeout_seconds=timeout_seconds,
                    )

                    # Extract timing from isolated benchmark result
                    if benchmark_result.get("success"):
                        min_time_ms = benchmark_result.get("min_time_ms", 0)
                        if min_time_ms > 0:
                            baseline_times[problem_id] = float(min_time_ms)
                            if i < 5:  # Log first few
                                logging.info(
                                    f"BaselineManager: Problem {problem_id} oracle time: {min_time_ms}ms (isolated)"
                                )
                        else:
                            # No valid oracle time - abort and let retry mechanism handle it
                            error_msg = (
                                f"BASELINE GENERATION FAILED: Oracle returned no valid time for problem {problem_id} ({i+1}/{problem_count}). "
                                f"This should never happen. Successfully generated {len(baseline_times)} baselines before failure."
                            )
                            logging.error(error_msg)
                            raise RuntimeError(error_msg)
                    else:
                        # Check if this is a timeout error (oracle should NEVER timeout)
                        error = benchmark_result.get('error', '')
                        is_timeout = (
                            benchmark_result.get('timeout_occurred', False) or
                            'timeout' in error.lower() or
                            'timed out' in error.lower()
                        )

                        if is_timeout:
                            # Oracle timeout - abort and let retry mechanism handle it
                            error_msg = (
                                f"BASELINE GENERATION FAILED: Oracle timed out for problem {problem_id} ({i+1}/{problem_count}). "
                                f"Oracle should NEVER timeout - this indicates the task is misconfigured or oracle is too slow. "
                                f"Error: {error}. Successfully generated {len(baseline_times)} baselines before failure."
                            )
                            logging.error(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            # Non-timeout error - log but continue (might be a flaky problem)
                            logging.error(
                                f"BaselineManager: Oracle isolated benchmark failed for {problem_id}: {error}"
                            )

                else:
                    # Use regular benchmark (may fall back to in-process)
                    logging.debug(f"BaselineManager: Using regular benchmark for {problem_id}")

                    # Create a wrapper function that binds the problem
                    def oracle_wrapper():
                        return oracle_solver(problem_data)

                    # Run the benchmark with auto-decided execution mode
                    benchmark_result = run_benchmark(
                        func=oracle_wrapper,
                        num_runs=num_runs,
                        num_warmups=1,
                        capture_output=False,
                        force_subprocess=None,  # Let it auto-decide based on environment
                    )

                    # Extract timing from result
                    if benchmark_result.get("success"):
                        # Get the minimum time in milliseconds
                        min_time_ms = benchmark_result.get("min_time_ms", 0)
                        if min_time_ms > 0:
                            baseline_times[problem_id] = float(min_time_ms)
                            if i < 5:  # Log first few
                                logging.info(
                                    f"BaselineManager: Problem {problem_id} oracle time: {min_time_ms}ms (regular)"
                                )
                        else:
                            # No valid oracle time - abort and let retry mechanism handle it
                            error_msg = (
                                f"BASELINE GENERATION FAILED: Oracle returned no valid time for problem {problem_id} ({i+1}/{problem_count}). "
                                f"This should never happen. Successfully generated {len(baseline_times)} baselines before failure."
                            )
                            logging.error(error_msg)
                            raise RuntimeError(error_msg)
                    else:
                        # Check if this is a timeout error (oracle should NEVER timeout)
                        error = benchmark_result.get('error', '')
                        is_timeout = (
                            benchmark_result.get('timeout_occurred', False) or
                            'timeout' in error.lower() or
                            'timed out' in error.lower()
                        )

                        if is_timeout:
                            # Oracle timeout - abort and let retry mechanism handle it
                            error_msg = (
                                f"BASELINE GENERATION FAILED: Oracle timed out for problem {problem_id} ({i+1}/{problem_count}). "
                                f"Oracle should NEVER timeout - this indicates the task is misconfigured or oracle is too slow. "
                                f"Error: {error}. Successfully generated {len(baseline_times)} baselines before failure."
                            )
                            logging.error(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            # Non-timeout error - log but continue (might be a flaky problem)
                            logging.error(
                                f"BaselineManager: Oracle benchmark failed for {problem_id}: {error}"
                            )

            except Exception as e:
                logging.error(f"BaselineManager: Error evaluating problem {problem_id}: {e}")
                import traceback

                logging.error(f"BaselineManager: Traceback: {traceback.format_exc()}")

        logging.info(
            f"BaselineManager: Generated baseline times for {len(baseline_times)} out of {problem_count} problems"
        )

        logging.info(f"Generated {len(baseline_times)} baseline times")

        # Validate baseline times
        if not baseline_times:
            logging.error("Generated empty baseline times!")
        else:
            # Log sample of baseline times
            sample_baseline = {k: baseline_times[k] for k in list(baseline_times.keys())[:3]}
            logging.info(f"Sample baseline times: {sample_baseline}")

        return baseline_times

    def clear_cache(self, subset: str | None = None):
        """
        Clear cached baseline times.

        Args:
            subset: Specific subset to clear, or None to clear all
        """
        with self._lock:
            if subset:
                old_count = len(self._cache[subset]) if self._cache[subset] else 0
                self._cache[subset] = None
                logging.info(f"Cleared baseline cache for {subset} (was {old_count} entries)")
            else:
                old_counts = {k: len(v) if v else 0 for k, v in self._cache.items()}
                self._cache = {"train": None, "test": None}
                logging.info(
                    f"Cleared all baseline caches (was train={old_counts['train']}, test={old_counts['test']})"
                )

    def get_cache_status(self) -> dict[str, int | None]:
        """
        Get the current cache status.

        Returns:
            Dict with subset names and number of cached entries (or None)
        """
        with self._lock:
            return {subset: len(times) if times else None for subset, times in self._cache.items()}
