import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

# Switch to dense eigen or sparse exponential
DENSE_THRESHOLD = 200

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        adj_list = problem.get("adjacency_list")
        # Empty graph case
        if not adj_list:
            return {"communicability": {}}

        n = len(adj_list)
        # Compute matrix exponential
        if n <= DENSE_THRESHOLD:
            # Build dense adjacency
            A = np.zeros((n, n), dtype=float)
            for u, neigh in enumerate(adj_list):
                for v in neigh:
                    A[u, v] = 1.0
            # Symmetrize adjacency
            A = np.maximum(A, A.T)
            # Eigen-decomposition (symmetric)
            vals, vecs = np.linalg.eigh(A)
            exp_vals = np.exp(vals)
            # exp(A) = V * diag(exp(vals)) * V.T
            C = (vecs * exp_vals) @ vecs.T
        else:
            # Build sparse adjacency
            rows, cols, data = [], [], []
            for u, neigh in enumerate(adj_list):
                for v in neigh:
                    rows.append(u); cols.append(v); data.append(1.0)
            A_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
            # exp(A) applied to identity yields full exp(A)
            C = expm_multiply(A_sparse, np.eye(n, dtype=float))

        # Build nested dict output: leverage tolist for faster conversion
        comm = {i: dict(enumerate(C[i].tolist())) for i in range(n)}
        return {"communicability": comm}