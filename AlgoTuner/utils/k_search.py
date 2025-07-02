"""Near-target *k* search utilities

This is the modern home of the logic that was previously in
``AlgoTuner.utils.timing``.  Only the public helper ``find_k_for_time`` (and
its internal helpers) are preserved; everything relies on the canonical
``precise_timing.time_execution_ns`` implementation for actual measurements.
"""

# ----------------------------------------------------------------------
# stdlib
# ----------------------------------------------------------------------
import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import pickle
from typing import Dict, List, Optional, Tuple, Callable

import resource  # Unix only; present on typical Slurm nodes

# ----------------------------------------------------------------------
# third-party
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
# project-local
# ----------------------------------------------------------------------
from AlgoTuner.utils.precise_timing import (
    _calculate_confidence_interval,
    time_execution_ns,
)
from AlgoTuner.utils.timing_config import RUNS, WARMUPS

# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------
LOG = logging.getLogger(__name__)

# ======================================================================
#  sandbox helpers
# ======================================================================

def _child_worker_isolated(q, task_bytes, k, warmup_seed, timed_seed):
    """Sub-process: generate + solve with isolated warmup and timed problems."""
    mem_bytes = int(os.environ.get("MEM_LIMIT_BYTES", "0"))
    if mem_bytes > 0:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except ValueError:
            pass

    try:
        task = pickle.loads(task_bytes)
    except Exception as exc:
        q.put(("error", f"unpickle task: {exc}"))
        return

    try:
        # Generate separate problems for warmup and timed measurement
        warmup_problem = task.generate_problem(k, random_seed=warmup_seed)
        timed_problem = task.generate_problem(k, random_seed=timed_seed)
        
        # Do warmup run on warmup problem
        warmup_result = time_execution_ns(
            func=task.solve,
            args=(warmup_problem,),
            num_runs=1,
            warmup_runs=WARMUPS,
            capture_output=False,
        )
        
        if not warmup_result.get("success"):
            q.put(("error", f"Warmup failed: {warmup_result.get('error', 'Unknown warmup error')}"))
            return
        
        # Do timed run on different problem (no warmup needed since we just warmed up)
        timing_result = time_execution_ns(
            func=task.solve,
            args=(timed_problem,),
            num_runs=RUNS,
            warmup_runs=0,  # No warmup needed, we already warmed up
            capture_output=False,
        )

        if timing_result.get("success"):
            min_ns = timing_result.get("min_ns")
            if min_ns is not None:
                q.put(("ok", min_ns / 1e9))
                return
            q.put(("error", "Timing succeeded but min_ns is None"))
        else:
            q.put(("error", timing_result.get("error", "Unknown timing error")))
    except MemoryError:
        q.put(("oom", None))
    except Exception as exc:
        q.put(("error", repr(exc)))


def _run_probe_safely_isolated(task, k, warmup_seed, timed_seed, timeout_s, memory_limit_mb):
    """Run one isolated generate+solve probe in a sandboxed subprocess with separate warmup and timed problems."""
    mem_bytes = memory_limit_mb * 1024 * 1024
    env = os.environ.copy()
    env["MEM_LIMIT_BYTES"] = str(mem_bytes)

    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_child_worker_isolated, args=(q, pickle.dumps(task), k, warmup_seed, timed_seed))
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout", None

    if p.exitcode is not None and p.exitcode < 0:
        sig = -p.exitcode
        if sig in (signal.SIGKILL, signal.SIGSEGV):
            return "oom", None
        return "error", None

    try:
        status, payload = q.get_nowait()
    except queue.Empty:
        return "error", None

    return status, payload

# ======================================================================
#  measure_solve_time — helper used by the k-search
# ======================================================================

def measure_solve_time(
    task,
    k: int,
    target_time: float,
    n_examples: int = 10,
    random_seed: int = 0,
    early_exit_multiplier: float = 10.0,
    timing_num_runs: int = 5,
    timing_warmup_runs: int = 3,
    timeout_s: float = 60.0,
    memory_limit_mb: int = 4096,
    log_level: int = logging.DEBUG,
) -> Tuple[Optional[float], Dict[str, any]]:
    """Measure mean solve time for *k* using 10x isolated probes with separate warmup and timed problems."""
    LOG.setLevel(log_level)
    times: List[float] = []
    errors = timeouts = ooms = 0
    early_exit = False
    error_messages = []  # Collect actual error messages

    # Run 10 isolated processes instead of n_examples
    # Each process uses different problems for warmup vs timed measurement
    for i in range(RUNS * 2):  # Use 2x RUNS for better statistics
        # Use different seeds for warmup and timed problems within each isolated run
        base_seed = random_seed + i * 1000  # Large offset to ensure different problems
        warmup_seed = base_seed
        timed_seed = base_seed + 500  # Different problem for timed measurement
        
        try:
            status, reported_s = _run_probe_safely_isolated(
                task,
                k,
                warmup_seed,
                timed_seed,
                timeout_s,
                memory_limit_mb,
            )
        except Exception as exc:
            LOG.error(f"isolated probe (k={k}, warmup_seed={warmup_seed}, timed_seed={timed_seed}) crashed: {exc}", exc_info=True)
            errors += 1
            break

        if status == "ok":
            times.append(reported_s)
            if i == 0 and reported_s > early_exit_multiplier * target_time:
                early_exit = True
                break
        elif status == "timeout":
            timeouts += 1
            break
        elif status == "oom":
            ooms += 1
            break
        else:
            errors += 1
            if reported_s:  # reported_s contains the error message when status is "error"
                error_messages.append(str(reported_s))
            break

    if early_exit or timeouts or ooms or errors:
        return None, {
            "errors": errors,
            "timeouts": timeouts,
            "oom": bool(ooms),
            "early_exit": early_exit,
            "num_runs": RUNS * 2,  # Updated to reflect isolated runs
            "warmup_runs": WARMUPS,  # Each isolated run does config warmups
            "error_messages": error_messages,  # Include actual error messages
        }

    if not times:
        LOG.warning(f"[measure_solve_time] k={k}: no successful runs")
        return None, {
            "errors": errors,
            "timeouts": timeouts,
            "oom": bool(ooms),
            "early_exit": early_exit,
            "num_runs": RUNS * 2,
            "warmup_runs": WARMUPS,
            "error_messages": error_messages,  # Include actual error messages
        }

    mean_time = sum(times) / len(times)
    try:
        ci_low, ci_high = _calculate_confidence_interval(times)
    except Exception:
        ci_low = ci_high = None

    return mean_time, {
        "times": times,
        "mean_time": mean_time,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "errors": 0,
        "timeouts": 0,
        "oom": False,
        "early_exit": False,
        "num_runs": RUNS * 2,
        "warmup_runs": WARMUPS,
    }

# ======================================================================
#  Public search API
# ======================================================================

def find_k_for_time(
    task,
    target_time: float,
    min_k: int = 1,
    max_k: int = 9_999_999,
    n_examples: int = 10,
    random_seed: int = 1,
    early_exit_multiplier: float = 10.0,
    n_initial: int = 16,
    n_refine: int = 8,
    memory_limit_mb: int = 8192,
    timing_num_runs: int = 5,
    timing_warmup_runs: int = 3,
) -> Tuple[Optional[int], Dict[str, any]]:
    """Search for *k* whose mean solve time ≈ *target_time* (seconds)."""
    assert target_time > 0
    min_k = max(1, min_k)
    max_k = max(min_k, max_k)

    cache: Dict[int, Tuple[Optional[float], Dict[str, any]]] = {}

    def probe(k: int):
        if k in cache:
            return cache[k]

        probe_timeout_s = max(50.0 * target_time, 1.0)  # 50× rule
        mean, st = measure_solve_time(
            task,
            k,
            target_time,
            n_examples,
            random_seed,
            early_exit_multiplier,
            timing_num_runs=timing_num_runs,
            timing_warmup_runs=timing_warmup_runs,
            timeout_s=probe_timeout_s,
            memory_limit_mb=memory_limit_mb,
            log_level=logging.DEBUG,
        )
        cache[k] = (mean, st)
        if mean is not None:
            msg = f"mean={mean:.4f}s"
        else:
            # Build failure reason from status dict
            failure_reasons = []
            if st.get("early_exit", False):
                failure_reasons.append("early_exit")
            if st.get("timeouts", 0) > 0:
                failure_reasons.append(f"timeout({st['timeouts']})")
            if st.get("oom", False):
                failure_reasons.append("oom")
            if st.get("errors", 0) > 0:
                failure_reasons.append(f"error({st['errors']})")
            
            reason = ",".join(failure_reasons) if failure_reasons else "unknown"
            
            # Add actual error messages if available
            error_details = ""
            if st.get("error_messages"):
                error_details = f" - {'; '.join(st['error_messages'])}"
            
            msg = f"FAILED: {reason}{error_details}"
        
        LOG.info(f"[probe] k={k:>5}: {msg}")
        return cache[k]

    if min_k == max_k:
        mean, st = probe(min_k)
        return (min_k, st) if mean is not None else (None, st)

    # 1) coarse log-spaced sweep
    exps = np.logspace(math.log10(min_k), math.log10(max_k), num=n_initial)
    sample_ks = sorted({int(max(min_k, min(max_k, round(x)))) for x in exps})
    sample_ks[0] = min_k
    sample_ks[-1] = max_k

    last_success_k = None
    upper_k = None
    last_probe_k = None

    for k in sample_ks:
        last_probe_k = k
        mean, _ = probe(k)
        if mean is None or mean > target_time:
            if last_success_k is not None:
                upper_k = k
                break
            continue
        last_success_k = k

    successes = [(k, mean) for k, (mean, _) in cache.items() if mean is not None]
    if not successes:
        return None, cache[last_probe_k][1]  # type: ignore[index]

    def err(x):
        return abs(x - target_time)

    best_k, best_mean = min(successes, key=lambda kv: (err(kv[1]), kv[0]))
    best_stats = cache[best_k][1]

    # 2) fine binary refinement
    if upper_k is not None and last_success_k is not None and n_refine > 0:
        low, high = last_success_k, upper_k
        for _ in range(n_refine):
            if high - low <= 1:
                break
            mid = (low + high) // 2
            mean_mid, st_mid = probe(mid)
            if mean_mid is None:
                high = mid - 1
                continue
            if err(mean_mid) < err(best_mean) or (
                math.isclose(err(mean_mid), err(best_mean)) and mid < best_k
            ):
                best_k, best_mean, best_stats = mid, mean_mid, st_mid
            if mean_mid > target_time:
                high = mid - 1
            else:
                low = mid

    return best_k, best_stats 