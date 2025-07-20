import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Load and cast inputs
        a = np.ascontiguousarray(problem["source_weights"], dtype=np.float64)
        b = np.ascontiguousarray(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        # Kernel matrix with numerical stability
        K = np.exp(-M / reg)
        tiny = np.finfo(np.float64).tiny
        np.maximum(K, tiny, out=K)
        # Initialize Sinkhorn variables
        # Initialize Sinkhorn variables
        n, m = K.shape
        u = np.ones(n, dtype=np.float64)
        v = np.ones(m, dtype=np.float64)
        max_iter = 1000
        tol = 1e-7
        # Sinkhorn iterations
        for _ in range(max_iter):
            u_prev = u
            # K@v and update u
            Kv = K @ v
            u = a / Kv
            # K.T@u and update v
            KTu = K.T @ u
            v = b / KTu
            # check convergence (infinity norm)
            if np.max(np.abs(u - u_prev)) < tol:
                break
        # transport plan
        P = u[:, None] * K * v[None, :]
        return {"transport_plan": P, "error_message": None}