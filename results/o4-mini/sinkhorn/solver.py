from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            # Load inputs
            a = np.asarray(problem["source_weights"], dtype=np.float64)
            b = np.asarray(problem["target_weights"], dtype=np.float64)
            M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
            reg = float(problem["reg"])
            # Dimensions and kernel
            n, m = a.size, b.size
            K = np.exp(-M / reg)
            # Initialize scalings and buffers
            u = np.ones(n, dtype=np.float64)
            v = np.ones(m, dtype=np.float64)
            u_old = np.empty_like(u)
            tmp_n = np.empty_like(u)
            tmp_m = np.empty_like(v)
            eps = 1e-300
            # Iteration parameters tuned for speed vs accuracy
            max_iter = 500
            tol = 1e-7
            # Sinkhorn loop with in-place BLAS-backed matmuls
            for _ in range(max_iter):
                u_old[:] = u
                # v update: use u @ K instead of K.T @ u
                np.matmul(u, K, out=tmp_m)
                tmp_m += eps
                np.divide(b, tmp_m, out=v)
                # u update
                np.matmul(K, v, out=tmp_n)
                tmp_n += eps
                np.divide(a, tmp_n, out=u)
                if np.max(np.abs(u - u_old)) < tol:
                    break
            # Construct transport plan
            G = (u[:, None] * K) * v[None, :]
            if not np.all(np.isfinite(G)):
                raise ValueError("Non-finite values in transport plan")
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}