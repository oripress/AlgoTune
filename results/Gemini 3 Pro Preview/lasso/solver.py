from typing import Any
import numpy as np
import lasso_cython

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        X = np.array(problem["X"], dtype=np.float32)
        y = np.array(problem["y"], dtype=np.float32)
        
        if not X.flags.f_contiguous:
            X = np.asfortranarray(X)
            
        return lasso_cython.solve_lasso(X, y, 0.1).tolist()