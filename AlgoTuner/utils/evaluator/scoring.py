"""
Scoring utilities for evaluating task performance.
"""

import logging
from typing import Optional

def calculate_input_speedup(solver_time_ms: float, baseline_time_ms: float, is_valid: bool) -> Optional[float]:
    """
    Calculates the speedup for a single input based on solver time vs oracle time.

    Speedup = (Oracle Time) / (Solver Time)
    Higher is better.

    - If the solution is invalid, speedup is None.
    - If oracle time is invalid (<= 0), speedup is None.
    - If solver time is 0 for a valid solution, speedup is infinite.

    Args:
        solver_time_ms: Time taken by the solver in milliseconds.
        baseline_time_ms: Reference time taken by the oracle solver in milliseconds.
        is_valid: Boolean indicating if the solver's solution was valid.

    Returns:
        The calculated speedup (float) or None if the speedup is undefined or invalid.
    """
    # COMPREHENSIVE SPEEDUP CALCULATION DEBUG: Log inputs with types
    logging.info(f"SPEEDUP_CALC_DEBUG: Inputs - solver_time_ms={solver_time_ms} (type: {type(solver_time_ms)}), baseline_time_ms={baseline_time_ms} (type: {type(baseline_time_ms)}), is_valid={is_valid} (type: {type(is_valid)})")
    
    # Additional checks for debugging
    if solver_time_ms is None:
        logging.debug(f"SPEEDUP_CALC_DEBUG: *** CRITICAL *** solver_time_ms is None! This is the root cause of the issue!")
    if baseline_time_ms is None:
        logging.debug(f"SPEEDUP_CALC_DEBUG: *** CRITICAL *** baseline_time_ms is None!")
    if is_valid is None:
        logging.debug(f"SPEEDUP_CALC_DEBUG: *** CRITICAL *** is_valid is None!")
    
    if not is_valid:
        logging.info("SPEEDUP_CALC_DEBUG: Invalid solution. Speedup is None.")
        return None

    if baseline_time_ms is None:
        logging.info("SPEEDUP_CALC_DEBUG: Baseline time is None. Speedup is None.")
        return None

    if baseline_time_ms <= 0:
        logging.warning(f"SPEEDUP_CALC_DEBUG: Baseline time <= 0 ({baseline_time_ms}ms). Speedup is None.")
        return None

    if solver_time_ms is None: # Check for None explicitly
         logging.warning(f"SPEEDUP_CALC_DEBUG: solver_time_ms is None. Speedup is None.")
         return None
         
    if solver_time_ms <= 0:
        logging.info(f"SPEEDUP_CALC_DEBUG: Solver time <= 0 ({solver_time_ms}ms) and solution is valid. Speedup is infinite.")
        return float('inf')

    # Regular calculation
    speedup = baseline_time_ms / solver_time_ms
    logging.info(f"SPEEDUP_CALC_DEBUG: Calculation: {baseline_time_ms} / {solver_time_ms} = {speedup}") # Log calculation
    return speedup 