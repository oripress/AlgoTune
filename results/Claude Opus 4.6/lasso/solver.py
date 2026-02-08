from typing import Any
import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True, boundscheck=False)
def _cd_lasso(X, y, alpha, max_iter, tol):
    n, d = X.shape
    w = np.zeros(d)
    r = y.copy()
    inv_n = 1.0 / n

    col_norms_sq_over_n = np.empty(d)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j] * X[i, j]
        col_norms_sq_over_n[j] = s * inv_n

    converged_on_active = False
    
    for iteration in range(max_iter):
        max_change = 0.0
        for j in range(d):
            denom = col_norms_sq_over_n[j]
            if denom == 0.0:
                continue

            old_w = w[j]
            rho = 0.0
            for i in range(n):
                rho += X[i, j] * r[i]
            rho = rho * inv_n + old_w * denom

            if rho > alpha:
                new_w = (rho - alpha) / denom
            elif rho < -alpha:
                new_w = (rho + alpha) / denom
            else:
                new_w = 0.0

            if new_w != old_w:
                diff = new_w - old_w
                for i in range(n):
                    r[i] -= diff * X[i, j]
                w[j] = new_w
                abs_change = abs(diff)
                if abs_change > max_change:
                    max_change = abs_change

        if max_change < tol:
            break

    return w


@nb.njit(cache=True, fastmath=True, boundscheck=False)
def _cd_lasso_gram(G, Xty, alpha, max_iter, tol):
    d = G.shape[0]
    w = np.zeros(d)
    q = Xty.copy()

    for iteration in range(max_iter):
        max_change = 0.0
        for j in range(d):
            denom = G[j, j]
            if denom == 0.0:
                continue

            old_w = w[j]
            rho = q[j] + denom * old_w

            if rho > alpha:
                new_w = (rho - alpha) / denom
            elif rho < -alpha:
                new_w = (rho + alpha) / denom
            else:
                new_w = 0.0

            if new_w != old_w:
                diff = new_w - old_w
                for k in range(d):
                    q[k] -= diff * G[k, j]
                w[j] = new_w
                abs_change = abs(diff)
                if abs_change > max_change:
                    max_change = abs_change

        if max_change < tol:
            break

    return w


@nb.njit(cache=True, fastmath=True, boundscheck=False)
def _cd_lasso_active(X, y, alpha, max_iter, tol):
    """Coordinate descent with active set strategy for high-d problems."""
    n, d = X.shape
    w = np.zeros(d)
    r = y.copy()
    inv_n = 1.0 / n

    col_norms_sq_over_n = np.empty(d)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j] * X[i, j]
        col_norms_sq_over_n[j] = s * inv_n

    # Active set indices
    active = np.empty(d, dtype=nb.int64)
    n_active = d
    for j in range(d):
        active[j] = j
    
    for iteration in range(max_iter):
        max_change = 0.0
        
        # Full pass or active pass
        if iteration == 0 or iteration % 5 == 0:
            # Full pass
            n_active_new = 0
            for jj in range(d):
                j = jj
                denom = col_norms_sq_over_n[j]
                if denom == 0.0:
                    continue

                old_w = w[j]
                rho = 0.0
                for i in range(n):
                    rho += X[i, j] * r[i]
                rho = rho * inv_n + old_w * denom

                if rho > alpha:
                    new_w = (rho - alpha) / denom
                elif rho < -alpha:
                    new_w = (rho + alpha) / denom
                else:
                    new_w = 0.0

                if new_w != old_w:
                    diff = new_w - old_w
                    for i in range(n):
                        r[i] -= diff * X[i, j]
                    w[j] = new_w
                    abs_change = abs(diff)
                    if abs_change > max_change:
                        max_change = abs_change
                
                if new_w != 0.0 or old_w != 0.0:
                    active[n_active_new] = j
                    n_active_new += 1
            
            n_active = n_active_new
        else:
            # Active set pass
            for jj in range(n_active):
                j = active[jj]
                denom = col_norms_sq_over_n[j]
                if denom == 0.0:
                    continue

                old_w = w[j]
                rho = 0.0
                for i in range(n):
                    rho += X[i, j] * r[i]
                rho = rho * inv_n + old_w * denom

                if rho > alpha:
                    new_w = (rho - alpha) / denom
                elif rho < -alpha:
                    new_w = (rho + alpha) / denom
                else:
                    new_w = 0.0

                if new_w != old_w:
                    diff = new_w - old_w
                    for i in range(n):
                        r[i] -= diff * X[i, j]
                    w[j] = new_w
                    abs_change = abs(diff)
                    if abs_change > max_change:
                        max_change = abs_change

        if max_change < tol:
            # Verify with full pass
            if iteration % 5 != 0:
                continue  # Will trigger full pass next
            break

    return w


# Trigger compilation at import time
_dummy_X = np.asfortranarray(np.zeros((2, 2), dtype=np.float64))
_dummy_y = np.zeros(2, dtype=np.float64)
_cd_lasso(_dummy_X, _dummy_y, 0.1, 1, 1e-4)
_dummy_G = np.zeros((2, 2), dtype=np.float64)
_dummy_Xty = np.zeros(2, dtype=np.float64)
_cd_lasso_gram(_dummy_G, _dummy_Xty, 0.1, 1, 1e-4)
_cd_lasso_active(_dummy_X, _dummy_y, 0.1, 1, 1e-4)


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        try:
            X_raw = problem["X"]
            y_raw = problem["y"]

            if isinstance(X_raw, np.ndarray):
                X = np.asfortranarray(X_raw, dtype=np.float64)
                y = np.ascontiguousarray(y_raw, dtype=np.float64)
            else:
                X = np.asfortranarray(X_raw, dtype=np.float64)
                y = np.array(y_raw, dtype=np.float64)

            n, d = X.shape
            alpha = 0.1
            max_iter = 1000

            # Use Gram matrix when d is small relative to n
            if d < 500 and n > 2 * d:
                inv_n = 1.0 / n
                G = (X.T @ X) * inv_n
                Xty = (X.T @ y) * inv_n
                w = _cd_lasso_gram(G, Xty, alpha, max_iter, 1e-4)
            elif d > 500:
                w = _cd_lasso_active(X, y, alpha, max_iter, 1e-4)
            else:
                w = _cd_lasso(X, y, alpha, max_iter, 1e-4)
            return w.tolist()
        except Exception as e:
            d = len(problem["X"][0]) if problem["X"] else 0
            return [0.0] * d