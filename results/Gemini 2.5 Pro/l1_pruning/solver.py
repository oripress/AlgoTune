import numpy as np
from typing import Any
import numba

# This is a single, unified JIT kernel that performs all core logic.
# It takes pre-computed values from NumPy to avoid redundant work.
# Merging the logic into one function eliminates the Python overhead of
# calling two separate kernels.
@numba.jit(nopython=True, fastmath=True, cache=True)
def project_core(v_flat: np.ndarray, u: np.ndarray, s: float, k: float) -> np.ndarray:
    """
    Core projection logic. Finds theta and applies soft-thresholding in one pass.
    """
    n = v_flat.shape[0]
    
    # --- Part 1: Serial Theta Search ---
    # This part is sequential as each iteration depends on the previous one.
    theta = (s - k) / n
    
    # Newton-like method converges very quickly. 32 iterations is a safe upper bound.
    for _ in range(32):
        n_support = 0
        sum_support = 0.0
        for i in range(n):
            if u[i] > theta:
                n_support += 1
                sum_support += u[i]
        
        if n_support == 0:
            break
            
        theta_new = (sum_support - k) / n_support
        
        if abs(theta_new - theta) < 1e-9:
            theta = theta_new
            break
            
        theta = theta_new
    
    if theta < 0.0:
        theta = 0.0

    # --- Part 2: Fused Projection ---
    # This loop is performed immediately after finding theta, inside the same kernel.
    w = np.empty_like(v_flat)
    for i in range(n):
        # Soft-thresholding: val = max(u[i] - theta, 0)
        val = u[i] - theta
        if val > 0:
            # Restore sign: val * sign(v_flat[i])
            if v_flat[i] > 0:
                w[i] = val
            else:
                w[i] = -val
        else:
            w[i] = 0.0
            
    return w

class Solver:
  def solve(self, problem: dict[str, Any], **kwargs) -> Any:
    v = np.array(problem.get("v"), dtype=np.float64)
    k = float(problem.get("k"))

    v_flat = v.flatten()
    
    # Step 1: Use highly optimized NumPy functions for pre-computation.
    u = np.abs(v_flat)
    s = np.sum(u)

    # Step 2: Early exit if already in the L1 ball.
    if s <= k:
        return {"solution": v_flat.tolist()}

    # Step 3: Call the single, unified JIT kernel to do all the heavy lifting.
    w_solution = project_core(v_flat, u, s, k)

    return {"solution": w_solution.tolist()}