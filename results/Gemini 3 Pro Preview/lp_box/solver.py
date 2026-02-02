import numpy as np
import highspy
from scipy.sparse import csc_matrix
from typing import Any

class Solver:
    def __init__(self):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("presolve", "off")
        self.h.setOptionValue("solver", "simplex")
        self.lp = highspy.HighsLp()
        
        # Pre-allocate large arrays to avoid allocation overhead
        # Assuming max dimension is reasonable, e.g., 10000
        # If problem is larger, we fallback to allocation
        self.max_dim = 10000
        self.zeros = np.zeros(self.max_dim, dtype=np.float64)
        self.ones = np.ones(self.max_dim, dtype=np.float64)
        self.inf = np.full(self.max_dim, -highspy.kHighsInf, dtype=np.float64)
        
        # Cache for indices and start
        self.cache_indices = {}
        self.cache_start = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        c = np.array(problem["c"], dtype=np.float64, copy=False)
        b = np.array(problem["b"], dtype=np.float64, copy=False)
        A = problem["A"]
        
        if not isinstance(A, np.ndarray):
            # Create Fortran contiguous array to make ravel('F') a view
            A = np.array(A, dtype=np.float64, order='F')
        else:
            # If already array, ensure float64. 
            # If we want view, we need to check flags, but converting to F order might copy anyway.
            # If it's C order, asfortranarray makes a copy.
            # But ravel('F') on C order makes a copy too.
            # So we might as well ensure F order if we can reuse it, but we don't reuse A.
            # Just let ravel handle it.
            if A.dtype != np.float64:
                A = A.astype(np.float64)
        
        num_col = c.shape[0]
        num_row = b.shape[0]
        
        # Flatten A in column-major order
        values = A.ravel(order='F')
        
        # Get cached indices or create
        key_indices = (num_row, num_col)
        if key_indices in self.cache_indices:
            indices = self.cache_indices[key_indices]
        else:
            # Faster indices creation using broadcast_to
            # indices = [0, 1, ..., num_row-1] repeated num_col times
            # This corresponds to column indices in the flattened array?
            # No, CSC row indices.
            # In column-major, the first num_row elements are column 0 (rows 0 to num_row-1).
            # So yes, it is 0, 1, ..., num_row-1 repeated.
            indices = np.tile(np.arange(num_row, dtype=np.int32), num_col)
            self.cache_indices[key_indices] = indices
            
        # Get cached start or create
        key_start = (num_row, num_col)
        if key_start in self.cache_start:
            start = self.cache_start[key_start]
        else:
            start = np.arange(num_col + 1, dtype=np.int32) * num_row
            self.cache_start[key_start] = start
            
        # Use addCols and addRows instead of passModel
        self.h.clearModel()
        
        # Prepare bounds
        if num_col <= self.max_dim:
            col_lower = self.zeros[:num_col]
            col_upper = self.ones[:num_col]
        else:
            col_lower = np.zeros(num_col, dtype=np.float64)
            col_upper = np.ones(num_col, dtype=np.float64)
            
        if num_row <= self.max_dim:
            row_lower = self.inf[:num_row]
        else:
            row_lower = np.full(num_row, -highspy.kHighsInf, dtype=np.float64)
        
        row_upper = b
        
        # Add rows (bounds only) first so row indices are valid
        self.h.addRows(num_row, row_lower, row_upper, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        
        # Add columns with matrix entries
        num_nz = len(values)
        self.h.addCols(num_col, c, col_lower, col_upper, num_nz, start, indices, values)
        
        self.h.run()
        
        sol = self.h.getSolution()
        return {"solution": list(sol.col_value)}