from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the communicability matrix exp(A) for an undirected graph.

        Input:
            {"adjacency_list": List[List[int]]}

        Output:
            {"communicability": {u: {v: float}}}
        """
        if not isinstance(problem, dict):
            return {"communicability": {}}
        adj_list = problem.get("adjacency_list")
        if not isinstance(adj_list, list):
            return {"communicability": {}}

        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build symmetric adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for u, neighbors in enumerate(adj_list):
            if not isinstance(neighbors, (list, tuple, set)):
                continue
            for v in neighbors:
                try:
                    vi = int(v)
                except Exception:
                    continue
                if 0 <= vi < n:
                    A[u, vi] = 1.0
                    A[vi, u] = 1.0

        # Compute matrix exponential using eigendecomposition for symmetric A
        expA = None
        try:
            w, Q = np.linalg.eigh(A)
            # Clip eigenvalues to avoid overflow in exp
            w_clipped = np.clip(w, -700, 700)
            exp_w = np.exp(w_clipped)
            # Reconstruct exp(A) = Q * diag(exp_w) * Q.T
            expA = (Q * exp_w) @ Q.T
            expA = np.real_if_close(expA, tol=1000)
        except Exception:
            # Fallback to scipy.linalg.expm if available
            try:
                from scipy.linalg import expm  # type: ignore
                expA = expm(A)
            except Exception:
                # Final fallback: Taylor series (slow but safe)
                I = np.eye(n, dtype=float)
                term = I.copy()
                S = I.copy()
                for k in range(1, 500):
                    term = term @ (A / k)
                    S += term
                    if np.linalg.norm(term, ord=np.inf) < 1e-16:
                        break
                expA = S

        arr = np.asarray(expA, dtype=float)

        # Ensure finite numbers
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(
                arr,
                nan=0.0,
                posinf=np.finfo(float).max,
                neginf=-np.finfo(float).max,
            )

        # Build nested dict output
        comm: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row = arr[i]
            comm[i] = {j: float(row[j]) for j in range(n)}

        return {"communicability": comm}