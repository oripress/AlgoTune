from typing import Any
import numpy as np
import numba

# Previous attempts using parallelization with Numba have proven unstable in the
# evaluation environment. This solution returns to a single-threaded approach,
# focusing on maximizing its performance.
#
# The most memory-efficient single-threaded design is a single, fused loop that
# calculates the cumulative difference and sums the absolute values in one pass.
# This minimizes memory bandwidth by avoiding intermediate arrays.
#
# To boost performance beyond previous single-threaded attempts (which plateaued
# around 4.5x speedup), this version adds the `fastmath=True` option to the
# Numba JIT decorator. This flag allows the compiler to make aggressive
# floating-point optimizations (e.g., reordering operations) that are not
# strictly IEEE 754 compliant. This can unlock significant performance gains,
# particularly by enabling SIMD vectorization that might otherwise be disallowed.
# The potential for minor floating-point inaccuracies is an acceptable trade-off
# for a large speed improvement in this context.
@numba.njit(cache=True, fastmath=True)
def _calculate_wasserstein_fused_fast(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculates the 1D Wasserstein distance using a single, fused Numba loop
    with fastmath optimizations enabled for maximum single-threaded performance.
    """
    n = u.shape[0]
    # The loop runs up to n-1. If n<=1, the loop is empty and 0.0 is returned,
    # correctly handling edge cases.
    
    cumulative_diff = 0.0
    total_distance = 0.0
    
    # The loop is over n-1 elements because the last element of the
    # cumulative sum of differences is always zero and doesn't contribute.
    for i in range(n - 1):
        cumulative_diff += u[i] - v[i]
        total_distance += abs(cumulative_diff)
        
    return total_distance

class Solver:
    def solve(self, problem: dict[str, list[float]], **kwargs) -> Any:
        """
        Computes the 1-Wasserstein distance for 1D discrete distributions
        using a high-performance, single-threaded Numba JIT-compiled function
        with fastmath optimizations.
        """
        u_np = np.array(problem["u"], dtype=np.float64)
        v_np = np.array(problem["v"], dtype=np.float64)

        return _calculate_wasserstein_fused_fast(u_np, v_np)