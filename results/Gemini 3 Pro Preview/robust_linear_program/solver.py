import numpy as np
import scipy.sparse as sp
import ecos
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        c = np.array(problem["c"])
        b = np.array(problem["b"])
        P = np.array(problem["P"])
        q = np.array(problem["q"])
        
        n = len(c)
        m = len(b)
        
        # ECOS formulation:
        # minimize c^T x
        # subject to h - G*x in K
        # Where K is a product of second-order cones.
        # Each constraint i: || P_i^T x || <= b_i - q_i^T x
        # corresponds to (b_i - q_i^T x, P_i^T x) in SOC_{n+1}
        # So h_i = [b_i; 0...0]
        #    G_i = [q_i^T; -P_i^T]
        
        # Construct h
        # Size m * (n + 1)
        h = np.zeros(m * (n + 1))
        # Subtract a small epsilon to ensure feasibility within tolerance
        # Subtract a small epsilon to ensure feasibility within tolerance
        # Modify b in-place to avoid temporary array creation
        b -= 1e-6
        h[::n+1] = b
        
        # Construct G efficiently in CSC format
        # G_data will hold the values in column-major order
        # Shape (m*(n+1), n) but we access it via a view to fill it
        G_data = np.empty((m * (n + 1), n), order='F')
        
        # Reshape to (n+1, m, n) to match the memory layout of F-ordered G_data
        # The first dimension (n+1) corresponds to the rows within each block
        # The second dimension (m) corresponds to the block index
        # The third dimension (n) corresponds to the columns
        G_view = G_data.reshape((n + 1, m, n), order='F')
        
        # Fill q
        G_view[0, :, :] = q
        
        # Fill P
        # We want to assign -P[i, k, j] to G_view[j+1, i, k]
        # P has shape (m, n, n) -> (i, k, j) (since symmetric, k and j are interchangeable)
        # We transpose P to (n, m, n) -> (j, i, k)
        G_view[1:, :, :] = -P.transpose(2, 0, 1)
        
        # Create CSC matrix directly
        data = G_data.ravel(order='F')
        # Use np.tile as it is faster than broadcast_to + ravel
        indices = np.tile(np.arange(m * (n + 1)), n)
        indptr = np.arange(0, n * m * (n + 1) + 1, m * (n + 1))
        
        # Create CSC matrix directly bypassing validation for speed
        # We access _shape directly to avoid property setter overhead/issues
        G = sp.csc_matrix.__new__(sp.csc_matrix)
        G.data = data
        G.indices = indices
        G.indptr = indptr
        G._shape = (m * (n + 1), n)
        
        dims = {'l': 0, 'q': [n + 1] * m, 'e': 0}
        
        # Solve
        sol = ecos.solve(c, G, h, dims, verbose=False)
        
        # Check status
        # 0: Optimal, 10: Optimal inaccurate
        if sol['info']['exitFlag'] in [0, 10]:
            return {
                "objective_value": sol['info']['pcost'],
                "x": sol['x']
            }
        else:
            return {
                "objective_value": float("inf"),
                "x": [np.nan] * n
            }