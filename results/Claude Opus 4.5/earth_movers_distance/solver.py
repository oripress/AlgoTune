import numpy as np
from ot.lp.emd_wrap import emd_c

class Solver:
    def __init__(self):
        # Cache the function reference
        self._emd_c = emd_c
        # Warm up
        a = np.array([0.5, 0.5], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)
        M = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        self._emd_c(a, b, M, 100000, 1)
    
    def solve(self, problem, **kwargs):
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]
        
        # Ensure correct types - emd_c requires contiguous float64 arrays
        if not isinstance(a, np.ndarray) or a.dtype != np.float64:
            a = np.ascontiguousarray(a, dtype=np.float64)
        if not isinstance(b, np.ndarray) or b.dtype != np.float64:
            b = np.ascontiguousarray(b, dtype=np.float64)
        if not isinstance(M, np.ndarray) or M.dtype != np.float64 or not M.flags['C_CONTIGUOUS']:
            M = np.ascontiguousarray(M, dtype=np.float64)
        
        # Call the C function directly
        G, cost, u, v, result_code = self._emd_c(a, b, M, 100000, 1)
        
        return {"transport_plan": G}