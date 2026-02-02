import numpy as np
from solver_cython import solve_fast

class Solver:
    def __init__(self):
        self.n_cap = 0
        self.af = None
        self.bf = None
        
    def solve(self, problem, **kwargs):
        A, B = problem
        # A and B are likely lists or arrays.
        # Get size.
        n = len(A)
        
        if n > self.n_cap:
            self.n_cap = max(n, self.n_cap * 2, 64)
            # Allocate F-contiguous buffers
            self.af = np.empty((self.n_cap, self.n_cap), dtype=np.float32, order='F')
            self.bf = np.empty((self.n_cap, self.n_cap), dtype=np.float32, order='F')
            
        # Copy data into buffers
        # We use views to ensure we pass correct size to cython
        af_view = self.af[:n, :n]
        bf_view = self.bf[:n, :n]
        
        # Copy A and B into views
        # This handles conversion to float32 and layout
        af_view[:] = A
        bf_view[:] = B
        
        vals = solve_fast(af_view, bf_view)
        
        # Sort descending by real, then descending by imag.
        vals.sort()
        return vals[::-1].tolist()