import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: dict, **kwargs) -> Dict[str, Any]:
        """
        Fast approximate projection onto the CVaR constraint set.

        Strategy:
        - Let k = int((1-beta) * m). CVaR constraint:
            sum_largest(A x, k) / k <= kappa  <=>  sum_largest(A x, k) <= alpha
          with alpha = kappa * k.
        - Iteratively:
            For current x compute top-k indices S for y = A x.
            If v^T x <= alpha (v = sum_{i in S} A_i) then feasible.
            Else project x onto halfspace {z: v^T z <= alpha}:
                x <- x - ((v^T x - alpha) / ||v||^2) * v
          Repeat until feasible or iteration/stall limit reached.
        - If iterative procedure fails, fall back to scaling x0 toward zero to enforce feasibility.
        This implementation uses only matrix-vector products and argpartition for speed.
        """
        # Parse input
        try:
            x0 = np.asarray(problem["x0"], dtype=float)
            A = np.asarray(problem["loss_scenarios"], dtype=float)
            beta = float(problem.get("beta", 0.95))
            kappa = float(problem.get("kappa", 0.0))
        except Exception:
            return {"x_proj": []}

        # Basic validation
        if x0.ndim != 1 or A.ndim != 2:
            return {"x_proj": []}
        m, n = A.shape
        if x0.size != n:
            return {"x_proj": []}

        # Determine k and alpha
        k = int((1.0 - beta) * m)
        if k <= 0:
            # Empty tail -> trivially feasible
            return {"x_proj": x0.tolist()}
        alpha = float(kappa * k)

        # Helpers
        def avg_topk(y):
            y = np.asarray(y).flatten()
            if k >= y.size:
                return float(np.mean(y))
            idx = np.argpartition(-y, k - 1)[:k]
            return float(np.mean(y[idx]))

        def topk_indices(y):
            y = np.asarray(y).flatten()
            if k >= y.size:
                return np.arange(y.size, dtype=int)
            return np.argpartition(-y, k - 1)[:k]

        # Quick feasibility check at x0
        y0 = A.dot(x0)
        if avg_topk(y0) <= kappa + 1e-12:
            return {"x_proj": x0.tolist()}

        # Iterative projection onto violating halfspaces (Kaczmarz-like)
        x = x0.copy()
        prev_overshoot = np.inf
        stall = 0
        max_iters = 1000
        tol = 1e-10

        for it in range(max_iters):
            y = A.dot(x)
            idx = topk_indices(y)
            # aggregate scenario row
            try:
                v = A[idx].sum(axis=0)
            except Exception:
                v = np.sum(A[np.asarray(idx, dtype=int)], axis=0)
            v = np.asarray(v, dtype=float)
            denom = float(np.dot(v, v))
            overshoot = float(np.dot(v, x) - alpha)

            if overshoot <= tol:
                return {"x_proj": x.tolist()}

            if denom <= 1e-16:
                # cannot correct with this hyperplane
                break

            # projection step onto {z: v^T z <= alpha}
            step = overshoot / denom
            x = x - step * v

            # Evaluate improvement (on same hyperplane)
            new_overshoot = float(np.dot(v, x) - alpha)
            if new_overshoot >= prev_overshoot - 1e-14:
                stall += 1
            else:
                stall = 0
            prev_overshoot = new_overshoot
            if stall >= 50:
                break

        # Final feasibility check
        y_final = A.dot(x)
        if avg_topk(y_final) <= kappa + 1e-7 and np.all(np.isfinite(x)):
            return {"x_proj": x.tolist()}

        # Fallback: scale x0 toward zero to reduce losses (conservative feasible point)
        try:
            cur = avg_topk(y0)
            if cur <= 0:
                c = 1.0
            else:
                c = min(1.0, float(kappa / cur))
            x_scaled = (c * x0).astype(float)
            if avg_topk(A.dot(x_scaled)) <= kappa + 1e-8:
                return {"x_proj": x_scaled.tolist()}
        except Exception:
            pass

        # Last resort: sanitize x and return
        x = np.nan_to_num(x, nan=0.0, posinf=1e100, neginf=-1e100)
        return {"x_proj": x.tolist()}