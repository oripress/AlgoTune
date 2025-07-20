import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}
        
        # Vectorized adjacency matrix construction
        A = np.zeros((n, n), dtype=np.float32)
        indices_i = []
        indices_j = []
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                if i < j:
                    indices_i.append(i)
                    indices_j.append(j)
        A[indices_i, indices_j] = 1
        A[indices_j, indices_i] = 1
        
        # Compute matrix exponential using Padé approximation
        expA = expm(A).astype(np.float64)
        
        # Optimized dictionary construction
        comm_dict = {i: dict(enumerate(expA[i].tolist())) for i in range(n)}
        
        return {"communicability": comm_dict}
        
        # Build output dictionary with optimized construction
        comm_dict = {}
        for i in range(n):
            comm_dict[i] = dict(enumerate(expA[i].tolist()))
        
        return {"communicability": comm_dict}
        expA_np = np.array(expA)
        comm_dict = {}
        for i in range(n):
            comm_dict[i] = dict(enumerate(expA_np[i].tolist()))
        
        return {"communicability": comm_dict}