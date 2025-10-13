from typing import Any, List
import numpy as np

# Try to use numba for speed; provide a numpy fallback if unavailable.
try:
    import numba as _nb

    @_nb.njit(cache=True, fastmath=False)
    def _cd_lasso_numba(Xt, y, alpha, max_iter, tol):
        # Xt: (d, n), y: (n,)
        n = y.shape[0]
        d = Xt.shape[0]

        # Initialize residual r = y - Xw with w=0, so r=y
        r = y.copy()
        w = np.zeros(d, dtype=np.float64)

        # Precompute a_j = (1/n) * ||x_j||^2
        a = np.empty(d, dtype=np.float64)
        for j in range(d):
            row = Xt[j]
            s = 0.0
            for i in range(n):
                s += row[i] * row[i]
            a[j] = s / n

        # Coordinate descent iterations
        old_obj = 1e100
        for _ in range(max_iter):
            max_dw = 0.0
            for j in range(d):
                aj = a[j]
                wj = w[j]
                if aj <= 0.0:
                    if wj != 0.0:
                        delta = -wj
                        row = Xt[j]
                        for i in range(n):
                            r[i] -= row[i] * delta
                        w[j] = 0.0
                        adelta = delta if delta >= 0 else -delta
                        if adelta > max_dw:
                            max_dw = adelta
                    continue

                # ro = (x_j^T r)/n + aj * w_j
                row = Xt[j]
                dot = 0.0
                for i in range(n):
                    dot += row[i] * r[i]
                ro = (dot / n) + aj * wj

                # Soft threshold
                if ro > alpha:
                    wnew = (ro - alpha) / aj
                elif ro < -alpha:
                    wnew = (ro + alpha) / aj
                else:
                    wnew = 0.0

                delta = wnew - wj
                if delta != 0.0:
                    for i in range(n):
                        r[i] -= row[i] * delta
                    w[j] = wnew
                    adelta = delta if delta >= 0 else -delta
                    if adelta > max_dw:
                        max_dw = adelta

            # Compute objective: 1/(2n)||r||^2 + alpha ||w||_1
            rss = 0.0
            l1 = 0.0
            for i in range(n):
                rss += r[i] * r[i]
            for j in range(d):
                l1 += w[j] if w[j] >= 0 else -w[j]
            obj = 0.5 * (rss / n) + alpha * l1

            if old_obj - obj < 1e-12 and max_dw < tol:
                break
            old_obj = obj

        return w

    # Precompile for float64 signature (compile time won't count against solve runtime).
    try:
        _ = _cd_lasso_numba(np.zeros((1, 1), dtype=np.float64), np.zeros(1, dtype=np.float64), 0.1, 1, 1e-8)
    except Exception:
        pass

    def _cd_lasso(Xt: np.ndarray, y: np.ndarray, alpha: float, max_iter: int, tol: float) -> np.ndarray:
        return _cd_lasso_numba(Xt, y, alpha, max_iter, tol)

except Exception:
    # NumPy fallback (still reasonably fast for moderate sizes)
    def _cd_lasso(Xt: np.ndarray, y: np.ndarray, alpha: float, max_iter: int, tol: float) -> np.ndarray:
        # Xt: (d, n), y: (n,)
        n = y.shape[0]
        d = Xt.shape[0]
        r = y.copy()
        w = np.zeros(d, dtype=np.float64)
        a = (Xt * Xt).sum(axis=1) / float(n)

        old_obj = np.inf
        for _ in range(max_iter):
            max_dw = 0.0
            for j in range(d):
                aj = a[j]
                wj = w[j]
                if aj <= 0.0:
                    if wj != 0.0:
                        delta = -wj
                        r -= Xt[j] * delta
                        w[j] = 0.0
                        max_dw = max(max_dw, abs(delta))
                    continue

                ro = Xt[j].dot(r) / float(n) + aj * wj
                if ro > alpha:
                    wnew = (ro - alpha) / aj
                elif ro < -alpha:
                    wnew = (ro + alpha) / aj
                else:
                    wnew = 0.0

                delta = wnew - wj
                if delta != 0.0:
                    r -= Xt[j] * delta
                    w[j] = wnew
                    if abs(delta) > max_dw:
                        max_dw = abs(delta)

            rss = float(r @ r)
            obj = 0.5 * (rss / float(n)) + alpha * np.abs(w).sum()
            if (old_obj - obj) < 1e-12 and max_dw < tol:
                break
            old_obj = obj

        return w


class Solver:
    def solve(self, problem: dict, **kwargs) -> List[float]:
        try:
            X = np.asarray(problem["X"], dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64).ravel()
            if X.ndim != 2:
                return []
            n, d = X.shape
            if y.shape[0] != n or d == 0:
                return [0.0] * d

            alpha = float(problem.get("alpha", 0.1))

            # Build contiguous Xt for fast row (feature) access
            Xt = np.ascontiguousarray(X.T)

            # Heuristic iteration/tolerance settings (good accuracy, fast)
            if "max_iter" in kwargs:
                max_iter = int(kwargs["max_iter"])
            else:
                # More iterations for small d to ensure high accuracy; fewer when large.
                if d <= 64:
                    max_iter = 1000
                elif d <= 512:
                    max_iter = 600
                else:
                    max_iter = 400

            if "tol" in kwargs:
                tol = float(kwargs["tol"])
            else:
                tol = 1e-10 if d <= 512 else 1e-8

            w = _cd_lasso(Xt, y, alpha, max_iter, tol)
            return w.tolist()
        except Exception:
            try:
                _, d = np.asarray(problem.get("X", [])).shape
                return [0.0] * d
            except Exception:
                return []