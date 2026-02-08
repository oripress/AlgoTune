from typing import Any
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

# Try to find the internal _highs_wrapper
try:
    from scipy.optimize._highspy._h import _highs_wrapper
    HAS_INTERNAL = True
except ImportError:
    try:
        from scipy.optimize._linprog_highs import _highs_wrapper
        HAS_INTERNAL = True
    except ImportError:
        try:
            from scipy.optimize._highspy import _highs_wrapper
            HAS_INTERNAL = True
        except ImportError:
            HAS_INTERNAL = False

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        n = c.shape[0]
        
        bounds = [(0.0, 1.0)] * n
        
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        return {"solution": result.x.tolist()}