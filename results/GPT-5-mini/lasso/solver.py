from typing import Any, List, Dict
import numpy as np

# Prefer scikit-learn's Lasso (fast, well-tested). If unavailable, fall back.
try:
    from sklearn.linear_model import Lasso  # type: ignore

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Elementwise soft-thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve Lasso:
            (1/(2n)) * ||y - X w||^2_2 + alpha * ||w||_1

        Strategy:
          - Use sklearn.linear_model.Lasso when available (matches reference).
          - Fallback: efficient coordinate descent (cyclic, residual updates) on
            column-major X for fast column access and quick convergence.
        Optional kwargs:
          - alpha: regularization parameter (default 0.1)
          - max_iter: maximum iterations / sweeps (default 1000)
          - tol: stopping tolerance on max coordinate change (default 1e-6)
        """
        # Parse inputs
        try:
            X = np.asarray(problem["X"], dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64)
        except Exception:
            return []

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = y.reshape(-1)

        n, d = X.shape
        if y.size != n:
            return [0.0] * d

        alpha = float(kwargs.get("alpha", 0.1))
        max_iter = int(kwargs.get("max_iter", 1000))
        tol = float(kwargs.get("tol", 1e-4))
        # Edge cases
        if d == 0:
            return []
        if n == 0:
            return [0.0] * d
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            return [0.0] * d

        # If no regularization, return least squares solution
        if alpha == 0.0:
            try:
                w_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
                return np.asarray(w_ls, dtype=float).reshape(-1).tolist()
            except Exception:
                return [0.0] * d

        # Prefer scikit-learn solver when available.
        if _HAS_SKLEARN:
            try:
                clf = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol)
                clf.fit(X, y)
                coef = np.asarray(clf.coef_, dtype=float).reshape(-1)
                return coef.tolist()
            except Exception:
                # Fall through to NumPy fallback if sklearn fails for any reason.
                pass

        # Fallback: Coordinate Descent (cyclic) with residual updates.
        # Use column-major layout to make column access contiguous.
        Xf = np.asfortranarray(X)
        y = np.asarray(y, dtype=np.float64)

        # Precompute column norms c_j = (1/n) * sum_i x_ij^2
        # Avoid full Gram matrix to save memory/time.
        col_sq = np.sum(Xf * Xf, axis=0)  # shape (d,)
        cj = col_sq / float(n)
        # Guard against zero norms
        eps = 1e-12
        zero_mask = cj <= 0.0
        if np.any(zero_mask):
            cj[zero_mask] = eps

        # Initialize
        w = np.zeros(d, dtype=np.float64)
        r = y.copy()  # residual = y - Xw (w is zero initially -> r = y)

        # Ordering: update coordinates with largest absolute correlation first
        # This improves initial convergence.
        Xty = Xf.T.dot(y)  # cost O(n*d) once
        order = np.argsort(-np.abs(Xty))

        # Determine number of sweeps adaptively to avoid long time on huge problems
        size = n * d
        if size > 2_000_000:
            max_sweeps = min(200, max(10, max_iter // 10))
        elif size > 500_000:
            max_sweeps = min(400, max(20, max_iter // 5))
        else:
            max_sweeps = max_iter

        # Coordinate descent sweeps
        for sweep in range(max_sweeps):
            max_delta = 0.0
            # Loop coordinates in chosen order
            for j in order:
                xj = Xf[:, j]  # view, contiguous in Fortran order
                # b_j = (1/n) * x_j^T r + c_j * w_j
                dot = float(xj.dot(r))
                bj = (dot / float(n)) + cj[j] * w[j]
                # soft-threshold and divide by c_j
                w_new = float(_soft_threshold(bj, alpha)) / cj[j]
                delta = w_new - w[j]
                if delta != 0.0:
                    # Update residual quickly: r <- r - x_j * delta
                    r -= xj * delta
                    w[j] = w_new
                    if abs(delta) > max_delta:
                        max_delta = abs(delta)
            if max_delta <= tol:
                break

        return w.tolist()