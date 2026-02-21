import numpy as np
from scipy.optimize._linprog_highs import _highs_wrapper
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        num_row, num_col = A.shape
        
        # Convert A to CSC format manually
        cols, rows = np.nonzero(A.T)
        data = A.T[cols, rows]
        
        counts = np.bincount(cols, minlength=num_col)
        indptr = np.zeros(num_col + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])
        indices = rows.astype(np.int32)
        
        lhs = np.full(num_row, -np.inf, dtype=np.float64)
        rhs = b
        lb = np.zeros(num_col, dtype=np.float64)
        ub = np.ones(num_col, dtype=np.float64)
        
        options = {
            'presolve': 'on',
            'sense': 1, # 1 for minimize
            'solver': 'simplex',
            'parallel': 'off',
            'output_flag': False,
        }
        
        integrality = np.empty(0, dtype=np.uint8)
        res = _highs_wrapper(
            c, indptr, indices, data,
            lhs, rhs, lb, ub, integrality, options
        )
        
        # res is a dict with 'x' as the solution
        return {"solution": res['x'].tolist()}