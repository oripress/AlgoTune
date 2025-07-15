#!/usr/bin/env python3
"""
Memory Leak Patch for isolated_benchmark.py

This module provides the actual patches to fix memory leaks in the existing
isolated_benchmark.py without rewriting the entire file.

Apply these changes to isolated_benchmark.py to fix the memory leaks.
"""

# The following changes should be applied to isolated_benchmark.py:

# 1. Add this import at the top of the file:
# from AlgoTuner.utils.resource_manager import ResourceManager, BenchmarkResultAccumulator

# 2. Replace the _run_with_manager function (around line 826) with this version:

def _run_with_manager_fixed(ctx):
    """Fixed version with proper memory management."""
    # Initialize local variables
    run_results = []
    last_result = None
    
    # Load manager refresh interval from config
    try:
        from AlgoTuner.config.config import load_config
        config = load_config()
        MANAGER_REFRESH_INTERVAL = config.get("benchmark", {}).get("manager_refresh_interval", 50)
        logger.debug(f"[isolated_benchmark] Using manager_refresh_interval={MANAGER_REFRESH_INTERVAL} from config")
    except Exception as e:
        MANAGER_REFRESH_INTERVAL = 50
        logger.debug(f"[isolated_benchmark] Failed to load manager_refresh_interval from config: {e}. Using default: {MANAGER_REFRESH_INTERVAL}")
    
    # Track Manager usage
    manager_usage = 0
    mgr = None
    
    try:
        for idx in range(num_runs):
            # Create or refresh Manager if needed
            if mgr is None or manager_usage >= MANAGER_REFRESH_INTERVAL:
                if mgr is not None:
                    logger.debug(f"[isolated_benchmark] Refreshing Manager after {manager_usage} uses")
                    try:
                        mgr.shutdown()
                    except Exception as e:
                        logger.warning(f"[isolated_benchmark] Error shutting down Manager: {e}")
                    del mgr
                    gc.collect()  # Force cleanup of Manager resources
                
                logger.debug(f"[isolated_benchmark] Creating new Manager for run {idx+1}/{num_runs}")
                mgr = ctx.Manager()
                manager_usage = 0
                
            # Each run: one fork does warmup+timed sequentially, then dies
            with tempfile.TemporaryDirectory() as tmp_dir:
                ret = mgr.dict()
                try:
                    proc = ctx.Process(
                        target=_fork_run_worker,
                        args=(task_name, code_dir, tmp_dir, warmup_payload, timed_payload, payload_is_pickled, ret),
                        daemon=False,
                    )
                    proc.start()
                    proc.join(timeout_seconds)

                    if proc.is_alive():
                        timeout_error = f"Process timed out after {timeout_seconds}s"
                        logger.warning(
                            f"[isolated_benchmark] Run {idx+1}/{num_runs} timed out after {timeout_seconds}s – will skip and continue"
                        )
                        # Try terminate first (SIGTERM) for cleaner shutdown
                        proc.terminate()
                        proc.join(timeout=0.5)  # Wait up to 0.5s for clean exit
                        if proc.is_alive():
                            # If still alive, force kill (SIGKILL)
                            proc.kill()
                            proc.join()

                        # Add a small delay to allow system cleanup after killing the process
                        time.sleep(0.1)  # 100ms delay

                        run_results.append({
                            "warmup_ns": None,
                            "timed_ns": None,
                            "timeout": True,
                        })
                        continue  # attempt next run

                    if not ret.get("success", False):
                        run_error = ret.get('error')
                        if not run_error:
                            ret_keys = list(ret.keys()) if ret else []
                            run_error = f"Process failed without error message. Return dict keys: {ret_keys}. Process may have crashed or timed out."
                        clean_error = _extract_clean_error_with_context(run_error)
                        logger.warning(
                            f"[isolated_benchmark] Run {idx+1}/{num_runs} failed: {clean_error}"
                        )
                        
                        run_results.append({
                            "warmup_ns": None,
                            "timed_ns": None,
                            "error": clean_error,
                        })
                        continue  # try next run

                    warmup_ns = ret.get("warmup_ns")
                    timed_ns = ret.get("timed_ns")
                    if warmup_ns is None or timed_ns is None:
                        timing_error = "No timing information returned from worker process"
                        logger.warning(
                            f"[isolated_benchmark] Run {idx+1}/{num_runs} did not return timing info – skipping"
                        )
                        run_results.append({
                            "warmup_ns": None,
                            "timed_ns": None,
                            "error": timing_error,
                        })
                        continue

                    # Capture stdout if available
                    captured_stdout = ret.get("stdout", "")
                    
                    run_results.append({
                        "warmup_ns": int(warmup_ns),
                        "timed_ns": int(timed_ns),
                        "warmup_ms": warmup_ns / 1e6,
                        "timed_ms": timed_ns / 1e6,
                        "stdout": captured_stdout,
                    })
                    
                    # Only unpickle the last result to save memory
                    if idx == num_runs - 1:  # Last run
                        try:
                            last_result = pickle.loads(ret.get("out_pickle", b""))
                        except Exception:
                            last_result = None
                    
                    # Increment manager usage counter
                    manager_usage += 1
                    
                finally:
                    # CRITICAL FIX: Clear and delete the shared dictionary
                    ret.clear()
                    del ret
                    
            # Periodic garbage collection
            if idx % 5 == 0:
                gc.collect()

    finally:
        # Clean up Manager if it exists
        if mgr is not None:
            logger.debug(f"[isolated_benchmark] Cleaning up Manager after {manager_usage} total uses")
            try:
                mgr.shutdown()
            except Exception as e:
                logger.warning(f"[isolated_benchmark] Error shutting down Manager during cleanup: {e}")
            del mgr
            gc.collect()

    return {"run_results": run_results, "last_result": last_result}


# 3. In the _fork_run_worker function (around line 644), replace the pickling section with:

"""
# Pickle the result for validation - only if this is likely the last run
try:
    # Check if we should skip pickling to save memory
    skip_pickle = os.environ.get("ALGOTUNER_SKIP_RESULT_PICKLE", "").lower() == "true"
    if not skip_pickle:
        ret_dict["out_pickle"] = pickle.dumps(timed_result)
    else:
        ret_dict["out_pickle"] = b""  # Empty bytes as placeholder
        ret_dict["had_result"] = True
except Exception as e:
    logging.warning(f"[isolated_worker] Failed to pickle result: {e}")
    ret_dict["out_pickle"] = b""
    ret_dict["had_result"] = True
"""

# 4. For the _run_with_manager_fetch function (around line 1128), apply similar fixes:
# - Add ret.clear() after line 1222
# - Add del ret after the clear()
# - Add periodic gc.collect() every few runs

# 5. Update the configuration to use a smaller refresh interval:
# In config.yaml, change:
#   manager_refresh_interval: 50
# To:
#   manager_refresh_interval: 10


if __name__ == "__main__":
    print("Memory Leak Patch Instructions")
    print("==============================")
    print()
    print("Apply the changes in this file to isolated_benchmark.py")
    print()
    print("Key fixes:")
    print("1. Clear Manager dictionaries after each use (ret.clear())")
    print("2. Delete dictionary references (del ret)")
    print("3. Only pickle the last result (memory optimization)")
    print("4. More aggressive Manager refresh (every 10 runs)")
    print("5. Regular garbage collection (every 5 runs)")
    print()
    print("These changes will prevent OOM kills during benchmark execution.")