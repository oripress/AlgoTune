"""
task_base.py
============

A lean Task base for AlgoTune.

Key policies
------------
1. **Single-k**: We pick one value of `k` and stick to it.  
2. **No attempt limits**: We continue generating/validating problems until
   `train_size` + `test_size` valid instances have been collected.  
3. **Atomic JSONL writes** and streaming loaders remain unchanged.
"""

from __future__ import annotations

import glob
import inspect
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import orjson

from AlgoTuner.config.loader import load_config
from AlgoTuner.utils.serialization import dataset_decoder, _is_large_ndarray, DatasetEncoder
from AlgoTuner.utils.serialization import externalize_large_arrays
from AlgoTuner.utils.streaming_json import stream_jsonl
from AlgoTuner.utils.casting import parse_string  # noqa: F401
from AlgoTuner.utils.type_inspection import describe_type
from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark
from AlgoTuner.utils.multiprocessing_utils import _pool_worker_initializer, load_pool_config, RESOURCE_AVAILABLE
from AlgoTuner.utils.k_search import find_k_for_time

_bench_cfg      = load_config().get("benchmark", {})
DATASET_RUNS    = _bench_cfg.get("runs",    5)
DATASET_WARMUPS = _bench_cfg.get("warmups", 3)

class BadDatasetError(RuntimeError):
    """Raised when dataset generation hits an unrecoverable error."""
    pass


def register_task(name: str):
    """
    Decorator to register a Task subclass in the global TASK_REGISTRY.
    """
    def decorator(cls):
        if name in TASK_REGISTRY:
            logging.debug("Task '%s' already registered; skipping.", name)
            return cls
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


INT64_MIN = -(2**63)
INT64_MAX = (2**63) - 1
def _convert_large_ints_to_str(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = str(k) if not isinstance(k, str) else k
            new_dict[new_key] = _convert_large_ints_to_str(v)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [_convert_large_ints_to_str(item) for item in obj]
    elif isinstance(obj, int) and (obj < INT64_MIN or obj > INT64_MAX):
        logging.debug(f"Converting large integer {obj} to string for serialization.")
        return str(obj)
    return obj

import multiprocessing
import sys
import time
import queue
import traceback

from AlgoTuner.utils.multiprocessing_utils import _pool_worker_initializer, load_pool_config, RESOURCE_AVAILABLE


def _pool_oracle_target(solve_func, problem):
    """Target function for oracle execution within a Pool worker."""
    pid = os.getpid()
    problem_repr = repr(problem)[:200]
    logging.info(f"GVP WORKER (PID: {pid}): Starting execution of solve_func for problem: {problem_repr}...")
    result_dict = {}
    solve_func_returned = False
    try:
        logging.info(f"GVP WORKER (PID: {pid}): Calling solve_func directly...")
        result = solve_func(problem)
        solve_func_returned = True
        logging.info(f"GVP WORKER (PID: {pid}): solve_func returned. Result type: {type(result)}")
        result_dict = {"success": True, "result": result}
    except Exception as e:
        solve_func_returned = True
        tb_str = traceback.format_exc()
        logging.error(f"GVP WORKER (PID: {pid}): Exception during solve_func: {e}\n{tb_str}")
        result_dict = {"success": False, "error": str(e), "traceback": tb_str, "error_type": "oracle_exception"}
    finally:
        if not solve_func_returned:
            logging.error(f"GVP WORKER (PID: {pid}): solve_func appears to have not returned (e.g., hung or crashed hard). This log is from the finally block.")
    logging.info(f"GVP WORKER (PID: {pid}): Finished execution. Success: {result_dict.get('success')}")
    return result_dict


class Task:
    """Base class for all AlgoTune tasks."""

    def generate_problem(self, n: int, random_seed: int = 1):
        raise NotImplementedError
    def solve(self, problem):
        raise NotImplementedError
    def is_solution(self, problem, solution) -> bool:
        raise NotImplementedError

    def __init__(self, n: Optional[int] = None, dataset_size: Optional[int] = None, target_time_ms: Optional[int] = None, data_dir: Optional[str] = None, **kwargs):
        """Initializes Task, potentially storing dataset parameters and data directory."""
        class_name = self.__class__.__name__
        self.task_name = class_name
        self.k: Optional[int] = None
        self.oracle = self.solve
        
        self.n = n
        self.dataset_size = dataset_size
        self.target_time_ms = target_time_ms
        
        self.data_dir = data_dir

        self._cached_train_file_path: Optional[Path] = None
        self._cached_test_file_path: Optional[Path] = None
        self._cached_data_params: Optional[Tuple[Optional[int], Optional[int], int, int]] = None
        self._estimated_oracle_time_s = None

    def get_task_directory(self) -> str:
        import inspect, os, logging
        try:
            return os.path.dirname(inspect.getfile(self.__class__))
        except TypeError:
            module = getattr(self.__class__, "__module__", None)
            if module:
                try:
                    mod = __import__(module, fromlist=["dummy"])
                    file = getattr(mod, "__file__", None)
                    if file:
                        return os.path.dirname(file)
                except Exception:
                    pass
            logging.error(f"Could not determine file for class {self.__class__}. Returning CWD as fallback.")
            return os.getcwd()

    def extract_solution(self, solve_output: Any) -> Any:
        """
        Best-effort extraction of the element matching the signature hint
        of `is_solution`.
        """
        import typing
        from inspect import Parameter

        sig   = inspect.signature(self.is_solution)
        hint  = None
        for p in sig.parameters.values():
            if p.name == "solution":
                hint = p.annotation
                break

        if hint is None or hint is Parameter.empty:
            return solve_output

        origin = typing.get_origin(hint)
        try:
            if origin and isinstance(solve_output, origin):
                return solve_output
            if not origin and isinstance(solve_output, hint):
                return solve_output
        except Exception:
            pass

        if isinstance(solve_output, np.ndarray) and (origin is np.ndarray or hint is np.ndarray):
            return solve_output

        if isinstance(solve_output, tuple):
            matches = [el for el in solve_output if self._matches_hint(el, hint)]
            if len(matches) == 1:
                return matches[0]

        return solve_output

    def _matches_hint(self, obj: Any, hint: Any) -> bool:
        import typing
        origin = typing.get_origin(hint)
        args   = typing.get_args(hint)

        if origin is None:
            try:
                return isinstance(obj, hint)
            except Exception:
                return True

        if origin in (list, tuple):
            if not isinstance(obj, origin):
                return False
            if not args:
                return True
            inner_hint = args[0]
            return all(self._matches_hint(el, inner_hint) for el in obj)

        if origin is dict:
            if not isinstance(obj, dict):
                return False
            if len(args) == 2:
                k_hint, v_hint = args
                return all(
                    self._matches_hint(k, k_hint) and self._matches_hint(v, v_hint)
                    for k, v in obj.items()
                )
            return True

        try:
            return isinstance(obj, origin)
        except Exception:
            return True

    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return self._convert_to_json_serializable(obj.tolist())
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complexfloating, np.complex64, np.complex128, complex)):
            return {"__complex__": True, "real": float(obj.real), "imag": float(obj.imag)}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, 'tolist'):
            return self._convert_to_json_serializable(obj.tolist())
        return obj

    def _convert_from_json(self, obj):
        if isinstance(obj, dict):
            if "__complex__" in obj:
                return complex(obj["real"], obj["imag"])
            return {k: self._convert_from_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_from_json(i) for i in obj]
        return obj

    def _get_target_time_ms(self) -> int:
        if hasattr(self, 'target_time_ms') and self.target_time_ms is not None:
            return self.target_time_ms
        cfg = load_config()
        task_name_for_config = self.task_name 
        return cfg.get("tasks", {}).get(task_name_for_config, {}).get(
            "oracle_time_limit",
            cfg.get("global", {}).get("oracle_time_limit", 1000)
        )

    def _benchmark_problem(self, problem_obj, target_time_ms, target_num_runs,
                           target_warmup_runs, current_seed, dataset_type):
        """
        Run a pre-check and a timed benchmark of self.solve.
        """
        from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark

        if target_time_ms and target_time_ms > 0:
            total_runs = target_num_runs + target_warmup_runs
            timeout_s  = max(1.0, total_runs * (target_time_ms / 1000.0) * 10.0)
        else:
            timeout_s = 3600.0

        precheck_timeout_s = (
            max(15.0, (target_time_ms / 1000.0) * 10.0 + 10.0)
            if (target_time_ms and target_time_ms > 0) else 60.0
        )

        pre = run_isolated_benchmark(
            task_name=self.task_name,
            code_dir=self.get_task_directory(),
            warmup_problem=problem_obj,
            timed_problem=problem_obj,
            num_runs=1,
            timeout_seconds=precheck_timeout_s
        )
        if not pre.get("success", False):
            pre["precheck_failed"] = True
            return pre

        main = run_isolated_benchmark(
            task_name=self.task_name,
            code_dir=self.get_task_directory(),
            warmup_problem=problem_obj,
            timed_problem=problem_obj,
            num_runs=target_num_runs,
            timeout_seconds=timeout_s
        )
        main["precheck_failed"] = False
        return main

    def load_dataset(
        self,
        train_size: int = 100,
        test_size: int = 100,
        random_seed: int = 42,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """Loads, validates, and potentially generates dataset based on parameters.
        Returns:
            Tuple[Generator, Generator]: Training and test data generators.
        """
        current_task_name = self.task_name 
        logging.info(f"load_dataset called for task '{current_task_name}' with train_size={train_size}, test_size={test_size}")

        if self.target_time_ms is not None:
            target_time_ms_for_load = self.target_time_ms
            logging.info(f"load_dataset: Using pre-configured self.target_time_ms = {target_time_ms_for_load}")
        else:
            target_time_ms_for_load = self._get_target_time_ms()
            logging.info(f"load_dataset: Determined target_time_ms via _get_target_time_ms() = {target_time_ms_for_load}")
        
        k_for_cache_check = self.k
        cached_params_tuple = (k_for_cache_check, target_time_ms_for_load, train_size, test_size)

        if self._cached_data_params == cached_params_tuple and \
           self._cached_train_file_path is not None and self._cached_test_file_path is not None and \
           self._cached_train_file_path.exists() and self._cached_test_file_path.exists():
            logging.info(f"Using cached dataset file paths for {self.task_name} (k={k_for_cache_check}, T={target_time_ms_for_load}ms, train={train_size}, test={test_size})")
            base_dir = self._cached_train_file_path.parent
            train_gen = stream_jsonl(str(self._cached_train_file_path), decoder_base_dir=str(base_dir))
            test_gen = stream_jsonl(str(self._cached_test_file_path), decoder_base_dir=str(base_dir))
            return train_gen, test_gen
        
        # _find_dataset_files will attempt to find files. 
        # It uses target_time_ms=None for wildcard search for T. 
        # It will cache paths and update self.k and self._cached_data_params (with actual T from filename) if files are found.
        found_train_path, found_test_path, k_from_file, reason_for_next_step = self._find_dataset_files(
            target_time_ms=None, 
            train_size=train_size,
            test_size=test_size,
            random_seed=random_seed
        )

        if found_train_path and found_test_path and k_from_file is not None:
            # If _find_dataset_files found them, it also cached them with the correct target_time_ms parsed from filename.
            # The self.k and self._cached_data_params are now up-to-date.
            logging.info(f"load_dataset: Using dataset files identified and cached by _find_dataset_files (k={self.k}): {self._cached_train_file_path}, {self._cached_test_file_path}")
            base_dir = self._cached_train_file_path.parent
            train_gen = stream_jsonl(str(self._cached_train_file_path), decoder_base_dir=str(base_dir))
            test_gen = stream_jsonl(str(self._cached_test_file_path), decoder_base_dir=str(base_dir))
            return train_gen, test_gen
        else:
            logging.info(f"load_dataset: No suitable pre-generated dataset found. Reason: {reason_for_next_step}. Proceeding to generation.")
            if os.environ.get("SKIP_DATASET_GEN") == "1":
                logging.info(f"load_dataset: SKIP_DATASET_GEN set; skipping dataset generation for task '{self.task_name}'")
                raise FileNotFoundError(f"Skipping dataset generation for {self.task_name} due to SKIP_DATASET_GEN. Original reason: {reason_for_next_step}")

                self._estimated_oracle_time_s = None # Reset before potential k-estimation
            
            k_for_generation = self.k
            if k_for_generation is None:
                logging.info(f"Task {self.task_name}: Estimating k for generation using target_time_ms={target_time_ms_for_load}ms...")
                from AlgoTuner.utils.k_search import find_k_for_time
                target_s_for_k_estimation = target_time_ms_for_load / 1000.0
                try:
                    cfg = load_config()
                    task_cfg = cfg.get('tasks', {}).get(self.task_name, {})
                    timing_cfg = cfg.get('timing', {})
                    min_k = task_cfg.get('min_k', timing_cfg.get('default_min_k', 1))
                    max_k = task_cfg.get('max_k', timing_cfg.get('default_max_k', 9999999))
                    n_examples_for_time = timing_cfg.get('n_examples_for_time', 3)
                    random_seed_for_timing = timing_cfg.get('random_seed', 42)
                    
                    pool_params_for_k_finding = load_pool_config(pool_config_name="validation_pool")
                    memory_limit_gb_for_k_finding = pool_params_for_k_finding.get("memory_limit_gb_per_worker", 14)
                    memory_limit_mb_for_k_finding = int(memory_limit_gb_for_k_finding * 1024)

                    k_est, stats = find_k_for_time(
                        self, target_s_for_k_estimation, min_k=min_k, max_k=max_k,
                        n_examples=n_examples_for_time, random_seed=random_seed_for_timing,
                        memory_limit_mb=memory_limit_mb_for_k_finding
                    )
                    k_for_generation = k_est
                    self.k = k_for_generation
                    mean_time = stats.get("mean_time") or stats.get("mean")
                    self._estimated_oracle_time_s = mean_time
                    if mean_time is not None:
                        logging.info(f"Task {self.task_name}: Estimated k = {k_for_generation} for generation (avg_time={mean_time:.4f}s if available).")
                    else:
                        logging.info(f"Task {self.task_name}: Estimated k = {k_for_generation} for generation (avg_time=unknown).")
                except Exception as e:
                    cfg = load_config()
                    task_cfg = cfg.get('tasks', {}).get(self.task_name, {})
                    timing_cfg = cfg.get('timing', {})
                    fallback_k = task_cfg.get('max_k', timing_cfg.get('default_max_k', 1000))
                    logging.warning(f"Task {self.task_name}: k estimation failed: {e}. Using fallback k={fallback_k} for generation.", exc_info=True)
                    k_for_generation = fallback_k
                    self.k = k_for_generation
            
            if k_for_generation is None:
                logging.error(f"Task {self.task_name}: Could not determine a valid k for dataset generation.")
                raise BadDatasetError(f"Failed to find or estimate a working k for task {self.task_name}")

            generation_data_dir = Path(self.data_dir) if self.data_dir else None
            if not generation_data_dir:
                base_data_dir_from_env = os.environ.get("DATA_DIR", load_config().get("DATA_DIR"))
                if not base_data_dir_from_env:
                    raise ValueError("DATA_DIR must be set (env or config) or task.data_dir must be pre-configured if self.data_dir is not set.")
                generation_data_dir = Path(base_data_dir_from_env) / current_task_name
            
            try:
                logging.info(f"Starting dataset generation for task {self.task_name} with k={k_for_generation}, train_size={train_size}, test_size={test_size} in dir {generation_data_dir}")
                self.create_dataset(
                    k=k_for_generation,
                    train_size=train_size,
                    test_size=test_size,
                    random_seed=random_seed, # Use the random_seed from load_dataset args
                    data_dir=str(generation_data_dir)
                )
                
                generated_train_file = generation_data_dir / f"{current_task_name}_T{target_time_ms_for_load}ms_n{k_for_generation}_size{train_size}_train.jsonl"
                generated_test_file = generation_data_dir / f"{current_task_name}_T{target_time_ms_for_load}ms_n{k_for_generation}_size{test_size}_test.jsonl"

                if not generated_train_file.exists() or not generated_test_file.exists():
                    missing_files_msg = f"Dataset generation finished, but expected files not found. Searched for: {generated_train_file}, {generated_test_file}"
                    logging.error(missing_files_msg)
                    raise BadDatasetError(missing_files_msg)

                self._cached_train_file_path = generated_train_file
                self._cached_test_file_path = generated_test_file
                self._cached_data_params = (k_for_generation, target_time_ms_for_load, train_size, test_size)
                
                logging.info(f"Dataset generation complete. Loading generated files: {generated_train_file}, {generated_test_file}")
                base_dir = generated_train_file.parent
                train_gen = stream_jsonl(str(generated_train_file), decoder_base_dir=str(base_dir))
                test_gen = stream_jsonl(str(generated_test_file), decoder_base_dir=str(base_dir))
                return train_gen, test_gen

            except Exception as e:
                logging.error(f"Failed to generate or load dataset for task {self.task_name}: {e}", exc_info=True)
                self._cached_train_file_path = None
                self._cached_test_file_path = None
                self._cached_data_params = None
                raise BadDatasetError(f"Dataset generation or loading failed for task {self.task_name}") from e

        final_error_msg = f"load_dataset: Unhandled state. Could not find or generate dataset for {current_task_name}. Reason from find: {reason_for_next_step}"
        logging.error(final_error_msg)
        raise FileNotFoundError(final_error_msg)

    def _generate_and_validate_problems(
        self,
        target_size: int,
        k: int,
        random_seed_base: int,
        random_seed_offset: int,
        validation_timeout_ms: int,
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield validated problems until *target_size* is reached.
        Uses a multiprocessing.Pool for efficient validation.
        """
        logging.info(f"GVP START: target_size={target_size}, k={k}, seed_base={random_seed_base}, offset={random_seed_offset}, val_timeout_ms={validation_timeout_ms}")
        current_time_start_gvp = time.perf_counter()

        produced = 0
        attempt = 0

        if self._estimated_oracle_time_s is not None and self._estimated_oracle_time_s > 0:
            base_timeout_s = 50.0 * self._estimated_oracle_time_s
            logging.info(f"GVP TIMEOUT_CALC: Using 50x estimated oracle time for validation timeout ({base_timeout_s:.2f}s)")
        else:
            target_time_ms_for_timeout = self._get_target_time_ms()
            base_timeout_s = 10.0 * (target_time_ms_for_timeout / 1000.0)
            logging.info(f"GVP TIMEOUT_CALC: Using 10x target oracle time for validation timeout ({base_timeout_s:.2f}s, from target_time_ms={target_time_ms_for_timeout})")
        min_timeout_s = 10.0
        validation_timeout_s = max(base_timeout_s, min_timeout_s)
        if validation_timeout_s != base_timeout_s:
            logging.info(f"GVP TIMEOUT_CALC: Adjusted validation timeout for generate_problem to minimum {validation_timeout_s:.2f}s")

        pool_params = load_pool_config(pool_config_name="validation_pool", force_num_workers=None)
        num_workers = pool_params["num_workers"]
        maxtasks = pool_params["maxtasksperchild"]
        mem_limit_bytes = pool_params["mem_limit_bytes"]
        disable_rlimit_as = pool_params.get("disable_rlimit_as", False)
        logging.info(f"GVP POOL_SETUP: num_workers={num_workers}, maxtasksperchild={maxtasks}, mem_limit_bytes={mem_limit_bytes}, disable_rlimit_as={disable_rlimit_as}")

        try:
            mp_context = multiprocessing.get_context('fork')
            logging.info(f"GVP POOL_INIT: Successfully got 'fork' multiprocessing context.")
        except Exception as e:
            logging.warning(f"GVP POOL_INIT: Failed to get 'fork' context, falling back to default. Error: {e}")
            mp_context = multiprocessing

        logging.info(f"GVP POOL_INIT: Creating multiprocessing.Pool (using {mp_context.get_start_method()} context) with num_workers={num_workers}, maxtasksperchild={maxtasks}, mem_limit_bytes={mem_limit_bytes}, disable_rlimit_as={disable_rlimit_as}")
        pool = mp_context.Pool(
            processes=num_workers,
            initializer=_pool_worker_initializer,
            initargs=(mem_limit_bytes, disable_rlimit_as),
            maxtasksperchild=maxtasks
        )
        active_validations: Dict[int, multiprocessing.pool.AsyncResult] = {}
        logging.info(f"GVP POOL_INIT: Pool created.")

        try:
            logging.debug(f"GVP MAIN_LOOP: Starting. Target size: {target_size}. Produced: {produced}. Attempt: {attempt}")
            loop_iter_count = 0
            not_ready_counts = defaultdict(int)

            while produced < target_size:
                loop_iter_count += 1
                current_time_loop_start = time.perf_counter()
                logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Checking completed validations (active: {len(active_validations)}). Elapsed: {(current_time_loop_start - current_time_start_gvp):.2f}s")
                
                completed_seeds = []
                any_task_was_ready_this_iteration = False
                if not active_validations:
                    logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: No active validations to check.")
                for seed_val, async_result_val in active_validations.items():
                    is_ready = async_result_val.ready()
                    logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Checking seed {seed_val}. Ready? {is_ready}")
                    if is_ready:
                        any_task_was_ready_this_iteration = True
                        not_ready_counts[seed_val] = 0
                        logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Seed {seed_val} IS READY.")
                        completed_seeds.append(seed_val)
                        current_time_before_get = time.perf_counter()
                        GET_TIMEOUT_S = max(60.0, validation_timeout_s * 5.0) 
                        logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Calling get() for seed {seed_val} with timeout {GET_TIMEOUT_S:.1f}s. Elapsed in loop: {(current_time_before_get - current_time_loop_start):.2f}s")
                        try:
                            res = async_result_val.get(timeout=GET_TIMEOUT_S)
                            current_time_after_get = time.perf_counter()
                            logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Seed {seed_val} get() returned in {(current_time_after_get - current_time_before_get):.2f}s. Result success: {res.get('success')}")
                            if res.get("success", False):
                                problem_to_yield = async_result_val._problem_ref if hasattr(async_result_val, '_problem_ref') else {}
                                logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Seed {seed_val} validation solve SUCCESS. Yielding problem. k={k}, seed={seed_val}")
                                yield {
                                    "k": k,
                                    "seed": seed_val,
                                    "problem": problem_to_yield,
                                    "median_oracle_time_ms": -1,
                                }
                                produced += 1
                                logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Produced {produced}/{target_size} for k={k}. Last seed: {seed_val}")
                            else:
                                error_msg = res.get("error", "Unknown validation error")
                                error_type = res.get("error_type", "unknown")
                                logging.info(f"GVP MAIN_LOOP iter {loop_iter_count}: Seed {seed_val} validation solve FAILED. Type: {error_type}, Msg: {error_msg}")
                        except multiprocessing.TimeoutError:
                             logging.error(f"GVP MAIN_LOOP iter {loop_iter_count}: Timeout WAITING FOR .get() for seed {seed_val} after {GET_TIMEOUT_S:.1f}s. Problem solve may be stuck or too slow.")
                        except Exception as e:
                            logging.error(f"GVP MAIN_LOOP iter {loop_iter_count}: Error getting result from completed validation (seed {seed_val}): {e}", exc_info=True)
                    else:
                        not_ready_counts[seed_val] += 1
                        if not_ready_counts[seed_val] % 50 == 0:
                            logging.warning(f"GVP MAIN_LOOP iter {loop_iter_count}: Seed {seed_val} has been NOT READY for {not_ready_counts[seed_val]} checks (approx. {not_ready_counts[seed_val]*0.1:.1f}s). Waiting...")
                
                if not active_validations and loop_iter_count > 1 and produced < target_size:
                     logging.warning(f"GVP MAIN_LOOP iter {loop_iter_count}: No active validations, but target not met ({produced}/{target_size}). This might indicate all submissions are failing before becoming active.")
                elif active_validations and not any_task_was_ready_this_iteration:
                    logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Active validations exist ({list(active_validations.keys())}), but NONE were ready this iteration.")

                if completed_seeds:
                    logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Removing {len(completed_seeds)} completed seeds: {completed_seeds}")
                for seed_del in completed_seeds:
                    del active_validations[seed_del]
                logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Active validations after removal: {len(active_validations)}")

                current_time_before_submit_loop = time.perf_counter()
                logging.debug(f"GVP SUBMIT_LOOP iter {loop_iter_count}: Checking if new tasks can be submitted. Active: {len(active_validations)}, Workers: {num_workers}, Target: {target_size}, Produced: {produced}. Elapsed in loop: {(current_time_before_submit_loop - current_time_loop_start):.2f}s")
                submit_attempt_count_this_iter = 0
                while len(active_validations) < num_workers and produced + len(active_validations) < target_size:
                    submit_attempt_count_this_iter += 1
                    seed = random_seed_base + random_seed_offset + attempt
                    logging.info(f"Generating problem {produced + len(active_validations) + 1}/{target_size} for task {self.task_name} (seed={seed}, k={k})")
                    attempt += 1
                    gen_problem_success = False
                    problem = None
                    try:
                        current_time_before_gen = time.perf_counter()
                        logging.info(f"GVP SUBMIT_LOOP iter {loop_iter_count}: DIRECT CALL to generate_problem (FIXED). Seed={seed}, k={k}.")
                        
                        problem = self.generate_problem(k, random_seed=seed)
                        
                        current_time_after_gen = time.perf_counter()
                        gen_duration = current_time_after_gen - current_time_before_gen
                        logging.debug(f"GVP SUBMIT_LOOP iter {loop_iter_count}: generate_problem (seed {seed}, k={k}) returned in {gen_duration:.2f}s.")
                        
                        if problem is None:
                            logging.warning(f"GVP SUBMIT_LOOP iter {loop_iter_count}: Problem generation (seed {seed}) returned None.")
                            continue
                        
                        problem_keys = list(problem.keys()) if isinstance(problem, dict) else "NOT_A_DICT"
                        logging.info(f"GVP SUBMIT_LOOP iter {loop_iter_count}: Problem generation SUCCESS for seed {seed}, k={k}. Keys: {problem_keys}")
                        gen_problem_success = True

                        current_time_before_apply_async = time.perf_counter()
                        logging.debug(f"GVP SUBMIT_LOOP iter {loop_iter_count}: Submitting to pool.apply_async for validation solve. Seed={seed}, k={k}")
                        async_result = pool.apply_async(_pool_oracle_target, (self.solve, problem))
                        active_validations[seed] = async_result
                        async_result._problem_ref = problem
                        current_time_after_apply_async = time.perf_counter()
                        logging.debug(f"GVP SUBMIT_LOOP iter {loop_iter_count}: Submitted to pool.apply_async for seed {seed} in {(current_time_after_apply_async - current_time_before_apply_async):.2f}s. Active validations: {len(active_validations)}")

                    except Exception as e:
                        logging.warning(f"GVP SUBMIT_LOOP iter {loop_iter_count}: Exception during problem generation/submission for seed {seed}: {e}", exc_info=True)
                        continue
                if submit_attempt_count_this_iter == 0:
                    logging.debug(f"GVP SUBMIT_LOOP iter {loop_iter_count}: No new tasks submitted in this iteration (pool full or target met). Active: {len(active_validations)}")
                
                current_time_before_sleep_check = time.perf_counter()
                if len(active_validations) >= num_workers:
                    logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: Pool is full (active: {len(active_validations)} >= workers: {num_workers}). Sleeping 0.1s. Elapsed in loop: {(current_time_before_sleep_check - current_time_loop_start):.2f}s")
                    time.sleep(0.1)

                if attempt > target_size * 100 and produced == 0:
                    logging.error(f"GVP FAILSAFE: Validation seems stuck. Attempted {attempt} times, produced {produced}. Aborting for task {self.task_name}, k={k}.")
                    raise BadDatasetError(f"Dataset generation failed: validation stuck for task {self.task_name}, k={k}")
                current_time_loop_end = time.perf_counter()
                logging.debug(f"GVP MAIN_LOOP iter {loop_iter_count}: End. Produced: {produced}/{target_size}. Total loop time: {(current_time_loop_end - current_time_loop_start):.2f}s. Total GVP time: {(current_time_loop_end - current_time_start_gvp):.2f}s")

        finally:
            current_time_finally = time.perf_counter()
            logging.info(f"GVP FINALLY: Shutting down validation worker pool. Elapsed: {(current_time_finally - current_time_start_gvp):.2f}s")
            pool.terminate()
            pool.join()
            current_time_after_shutdown = time.perf_counter()
            logging.info(f"GVP FINALLY: Validation worker pool shut down. Shutdown took: {(current_time_after_shutdown - current_time_finally):.2f}s. Total GVP time: {(current_time_after_shutdown - current_time_start_gvp):.2f}s")

    def create_dataset(
        self,
        k: int,
        train_size: int = 100,
        test_size: int = 100,
        random_seed: int = 1,
        data_dir: Optional[str] = None,
    ) -> None:
        """
        Create and save training and test datasets of the specified sizes.

        Args:
            k: The problem size parameter.
            train_size: The number of training examples.
            test_size: The number of test examples.
            random_seed: The random seed for reproducibility.
            data_dir: The directory where datasets will be saved, defaults to DATA_DIR env var.
        """
        if data_dir is None:
            data_dir = self.data_dir or os.environ.get("DATA_DIR")
            if not data_dir:
                raise ValueError("DATA_DIR must be set via env, config, or passed explicitly to create_dataset().")

        task_dir = Path(data_dir)
        subdir_candidate = task_dir / self.task_name
        if subdir_candidate.is_dir() or not task_dir.exists():
            task_dir = subdir_candidate

        task_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving datasets under %s", task_dir)

        if self.k is None:
             self.k = k
        elif self.k != k:
             logging.warning(f"create_dataset called with k={k}, but self.k is already set to {self.k}. Using provided k={k}.")

        logging.info(f"Generating training dataset (k={k}, size={train_size})...")
        validation_timeout_ms = self._get_target_time_ms() * 10

        train_tmp_file = None
        test_tmp_file = None
        encoder = DatasetEncoder()
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=".jsonl", dir=task_dir) as tmp_f:
                train_tmp_file = Path(tmp_f.name)
                train_gen = self._generate_and_validate_problems(
                    target_size=train_size,
                    k=k,
                    random_seed_base=random_seed,
                    random_seed_offset=0,
                    validation_timeout_ms=validation_timeout_ms
                )
                count = 0
                for record in train_gen:
                    processed_record_ext = externalize_large_arrays(record, task_dir)
                    processed_record_pre = encoder._preprocess_for_json(processed_record_ext)
                    processed_record_final = _convert_large_ints_to_str(processed_record_pre)
                    json_bytes = orjson.dumps(processed_record_final, default=encoder.default, option=orjson.OPT_APPEND_NEWLINE)
                    tmp_f.write(json_bytes)
                    count += 1
                logging.info(f"Saved {count} training records to {train_tmp_file}")
            
            logging.info(f"Generating test dataset (k={k}, size={test_size})...")
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=".jsonl", dir=task_dir) as tmp_f:
                test_tmp_file = Path(tmp_f.name)
                test_gen = self._generate_and_validate_problems(
                    target_size=test_size,
                    k=k,
                    random_seed_base=random_seed,
                    random_seed_offset=train_size,
                    validation_timeout_ms=validation_timeout_ms
                )
                count = 0
                for record in test_gen:
                    processed_record_ext = externalize_large_arrays(record, task_dir)
                    processed_record_pre = encoder._preprocess_for_json(processed_record_ext)
                    processed_record_final = _convert_large_ints_to_str(processed_record_pre)
                    json_bytes = orjson.dumps(processed_record_final, default=encoder.default, option=orjson.OPT_APPEND_NEWLINE)
                    tmp_f.write(json_bytes)
                    count += 1
                logging.info(f"Saved {count} test records to {test_tmp_file}")

            filename_task_name = self.task_name
            train_tmp_file.rename(task_dir / f"{filename_task_name}_T{self._get_target_time_ms()}ms_n{k}_size{train_size}_train.jsonl")
            test_tmp_file.rename(task_dir / f"{filename_task_name}_T{self._get_target_time_ms()}ms_n{k}_size{test_size}_test.jsonl")
            logging.info(f"Renamed temp files to final dataset files: {task_dir / f'{filename_task_name}_T{self._get_target_time_ms()}ms_n{k}_size{train_size}_train.jsonl'}, {task_dir / f'{filename_task_name}_T{self._get_target_time_ms()}ms_n{k}_size{test_size}_test.jsonl'}")

        except Exception as e:
            logging.error(f"Error during dataset creation: {e}", exc_info=True)
            if train_tmp_file and train_tmp_file.exists():
                train_tmp_file.unlink()
            if test_tmp_file and test_tmp_file.exists():
                test_tmp_file.unlink()
            raise BadDatasetError(f"Dataset generation failed for task {self.task_name}") from e
        finally:
            if train_tmp_file and train_tmp_file.exists():
                 logging.warning(f"Temporary train file {train_tmp_file} still exists after create_dataset completion/failure. Removing.")
                 try: train_tmp_file.unlink()
                 except OSError: pass
            if test_tmp_file and test_tmp_file.exists():
                 logging.warning(f"Temporary test file {test_tmp_file} still exists after create_dataset completion/failure. Removing.")
                 try: test_tmp_file.unlink()
                 except OSError: pass

    def _find_dataset_files(self, target_time_ms: Optional[int], train_size: int, test_size: int, random_seed: int) -> Tuple[Optional[Path], Optional[Path], Optional[int], str]:
        """Finds existing dataset files based on target_time_ms and sizes.

        Returns:
            (train_path, test_path, k, reason) where k is derived from filename, or Nones if not found.
            Reason explains why generation might be needed.
        """
        k: Optional[int] = None

        data_dir_to_use: Optional[str | os.PathLike] = self.data_dir
        if not data_dir_to_use:
            data_dir_to_use = os.environ.get("DATA_DIR")
            if not data_dir_to_use:
                logging.warning(
                    "_find_dataset_files: data_dir not provided via init or DATA_DIR env var. "
                    "Using default relative path '../data'."
                )
                data_dir_to_use = "../data"

        task_specific_data_dir = Path(data_dir_to_use)
        subdir_candidate = task_specific_data_dir / self.task_name
        if subdir_candidate.is_dir():
            task_specific_data_dir = subdir_candidate

        search_task_name_for_filename = self.task_name or self.__class__.__name__

        filename_base = search_task_name_for_filename
        logging.info(
            f"_find_dataset_files: Using task name '{filename_base}' for file search (original was '{self.task_name}')"
        )


        ignore_target_time = target_time_ms is None

        if not task_specific_data_dir.is_dir():
            return None, None, None, (
                f"Task data directory not found: {task_specific_data_dir.resolve()}"
            )

        logging.info(
            f"_find_dataset_files: Absolute task-specific path being searched: {task_specific_data_dir.resolve()}"
        )

        if ignore_target_time:
            train_pattern_glob = f"{filename_base}_T*ms_n*_size{train_size}_train.jsonl"
            logging.info(f"_find_dataset_files: Ignoring target_time_ms. Searching with glob pattern: '{train_pattern_glob}' in '{task_specific_data_dir}'")
        else:
            train_pattern_glob = f"{filename_base}_T{target_time_ms}ms_n*_size{train_size}_train.jsonl"
            logging.info(f"_find_dataset_files: Searching with glob pattern: '{train_pattern_glob}' in '{task_specific_data_dir}'")

        found_train_files_raw = list(task_specific_data_dir.glob(train_pattern_glob))
        logging.info(f"_find_dataset_files: Glob found {len(found_train_files_raw)} potential files: {[p.name for p in found_train_files_raw]}")

        found_train_files = sorted(
            found_train_files_raw,
            key=lambda p: int(re.search(r"_n(\d+)_", p.name).group(1)) if re.search(r"_n(\d+)_", p.name) else -1,
            reverse=True
        )
        logging.info(f"_find_dataset_files: Sorted potential files (descending k): {[p.name for p in found_train_files]}")

        train_file_to_load = None
        test_file_to_load = None
        k_from_file = None
        reason_for_generation = "No suitable existing files found."

        if found_train_files:
            logging.info(f"Found {len(found_train_files)} potential existing train files (target_time {'ANY' if ignore_target_time else f'{target_time_ms}ms'}), size={train_size}.")
            selected_train_file = found_train_files[0]
            logging.info(f"_find_dataset_files: Attempting to use best candidate train file: '{selected_train_file.name}'")
            match = re.search(r"_n(\d+)_", selected_train_file.name)
            if match:
                parsed_k = int(match.group(1))
                logging.info(f"_find_dataset_files: Parsed k={parsed_k} from filename '{selected_train_file.name}'")
                expected_test_file = selected_train_file.parent / selected_train_file.name.replace("_train.jsonl", "_test.jsonl")
                logging.info(f"_find_dataset_files: Checking for corresponding test file: '{expected_test_file}'")
                if expected_test_file.exists():
                    k_from_file = parsed_k
                    train_file_to_load = selected_train_file
                    test_file_to_load = expected_test_file
                    logging.info(f"_find_dataset_files: Selected best available dataset (k={k_from_file}): {train_file_to_load.name}")
                else:
                    logging.warning(f"_find_dataset_files: Selected train file {selected_train_file} but matching test file {expected_test_file} missing. Will attempt generation.")
                    reason_for_generation = f"Matching test file '{expected_test_file.name}' not found for selected train file '{selected_train_file.name}'."
            else:
                logging.warning(f"_find_dataset_files: Could not parse k from selected train file: {selected_train_file}. Will attempt generation.")
                reason_for_generation = f"Could not parse k from candidate train file '{selected_train_file.name}'."
        else:
            logging.info(f"_find_dataset_files: No existing dataset files found for target_time {'ANY' if ignore_target_time else f'{target_time_ms}ms'}, size={train_size}. Will attempt generation.")
            
        if k_from_file is not None and train_file_to_load and test_file_to_load:
            k = k_from_file
            self.k = k
            logging.info(f"_find_dataset_files: Set k={k} from found files. Updated self.k.")
            
            self._cached_train_file_path = train_file_to_load
            self._cached_test_file_path = test_file_to_load
            parsed_T_for_cache = target_time_ms
            if parsed_T_for_cache is None and train_file_to_load:
                t_match = re.search(r"_T(\d+)ms_", train_file_to_load.name)
                if t_match:
                    parsed_T_for_cache = int(t_match.group(1))
                else:
                    logging.warning(f"Could not parse T from filename {train_file_to_load.name} for caching key, using provided {target_time_ms}")
            
            self._cached_data_params = (k, parsed_T_for_cache, train_size, test_size)
            logging.info(f"_find_dataset_files: Successfully found and cached file paths (k={k}, T={parsed_T_for_cache}ms). Returning paths.")
            return train_file_to_load, test_file_to_load, k, "Found existing dataset files and cached their paths."
        else:
            logging.info(f"_find_dataset_files: No loadable dataset pair found. Returning to caller to potentially generate. Reason: {reason_for_generation}")
            return None, None, None, reason_for_generation

TASK_REGISTRY = {}
        