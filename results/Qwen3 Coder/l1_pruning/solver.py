import numpy as np
from typing import Any, Dict, List
from numba import jit

@jit(nopython=True)
def find_theta(sorted_u, k):
    """Find the threshold theta using numba-compiled function."""
    cumsum = 0.0
    theta = 0.0
    for j in range(len(sorted_u)):
        cumsum += sorted_u[j]
        if sorted_u[j] < (cumsum - k) / (j + 1):
            theta = (cumsum - k) / (j + 1)
            break
    return theta

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the L1 proximal operator (projection onto L1 ball).
        
        Args:
            problem: Dictionary with keys:
                - "v": A list of n real numbers
                - "k": the L1 norm bound
            
        Returns:
            A dictionary with key "solution": the projected vector
        """
        v = np.asarray(problem["v"], dtype=np.float64)
        k = float(problem["k"])
        
        # If k is large enough, no projection is needed
        l1_norm = np.sum(np.abs(v))
        if k >= l1_norm:
            return {"solution": v.tolist()}
        
        # For projection onto L1 ball, we use the standard algorithm
        # 1. Work with absolute values
        u = np.abs(v)
        
        # 2. Sort in descending order
        sorted_u = np.sort(u)[::-1]
        
        # 3. Find threshold using numba-compiled function
        theta = find_theta(sorted_u, k)
        
        # 4. Apply soft thresholding
        b = np.maximum(u - theta, 0)
        
        # 5. Apply original signs
        solution = b * np.sign(v)
        
        return {"solution": solution.tolist()}