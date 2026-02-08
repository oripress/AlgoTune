import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)
        
        if n == 0:
            return []
        
        w = np.array(weights, dtype=np.float64)
        adj = np.array(adj_matrix, dtype=np.int8)
        
        # Build edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append((i, j))
        
        m = len(edges)
        
        if m == 0:
            # No edges - take all non-negative weight vertices
            return sorted([i for i in range(n) if w[i] >= 0])
        
        # Solve with scipy milp (HiGHS)
        c = -w  # minimize negative weights = maximize weights
        
        if m > 0:
            rows = np.empty(2 * m, dtype=np.int32)
            cols = np.empty(2 * m, dtype=np.int32)
            data = np.ones(2 * m, dtype=np.float64)
            
            for kk, (ii, jj) in enumerate(edges):
                rows[2 * kk] = kk
                rows[2 * kk + 1] = kk
                cols[2 * kk] = ii
                cols[2 * kk + 1] = jj
            
            A = csr_matrix((data, (rows, cols)), shape=(m, n))
            constraints = LinearConstraint(A, ub=np.ones(m))
        else:
            constraints = []
        
        integrality = np.ones(n)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success:
            return sorted([i for i in range(n) if result.x[i] > 0.5])
        
        return []