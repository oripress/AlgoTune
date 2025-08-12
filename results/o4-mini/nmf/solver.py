from typing import Any, Dict, List
import numpy as np
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        # parse inputs
        X = np.array(problem.get("X", []), dtype=float)
        if X.ndim != 2:
            return {"W": [], "H": []}
        m, n = X.shape
        k = int(problem.get("n_components", 0))
        # trivial fallback
        if k <= 0 or m == 0 or n == 0:
            return {
                "W": [[0.0] * k for _ in range(m)],
                "H": [[0.0] * n for _ in range(k)],
            }
        try:
            model = NMF(n_components=k, init="random", random_state=0)
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            return {
                "W": [[0.0] * k for _ in range(m)],
                "H": [[0.0] * n for _ in range(k)],
            }
        VT = Vt * sqrt_S[:, np.newaxis]
        W = np.zeros((m, k), dtype=float)
        H = np.zeros((k, n), dtype=float)
        eps = 1e-8
        for j in range(k):
            u = UT[:, j]
            v = VT[j, :]
            u_p = np.maximum(u, 0.0)
            u_n = np.maximum(-u, 0.0)
            v_p = np.maximum(v, 0.0)
            v_n = np.maximum(-v, 0.0)
            n_u_p = np.linalg.norm(u_p)
            n_u_n = np.linalg.norm(u_n)
            n_v_p = np.linalg.norm(v_p)
            n_v_n = np.linalg.norm(v_n)
            if n_u_p * n_v_p >= n_u_n * n_v_n:
                if n_u_p > eps and n_v_p > eps:
                    W[:, j] = u_p / (n_u_p + eps)
                    H[j, :] = v_p / (n_v_p + eps)
            else:
                if n_u_n > eps and n_v_n > eps:
                    W[:, j] = u_n / (n_u_n + eps)
                    H[j, :] = v_n / (n_v_n + eps)
        # A few multiplicative update iterations, dynamic for speed/quality
        prod = m * n * k
        if prod <= 1e6:
            n_iter = 5
        elif prod <= 1e7:
            n_iter = 3
        else:
            n_iter = 1
        for _ in range(n_iter):
            # update W
            W *= (X @ H.T) / (W @ (H @ H.T) + eps)
            # update H
            H *= (W.T @ X) / ((W.T @ W) @ H + eps)
        return {"W": W.tolist(), "H": H.tolist()}