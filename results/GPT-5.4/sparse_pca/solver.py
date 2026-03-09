import numpy as np
from numba import njit

try:
    from scipy.linalg import eigh as scipy_eigh
except Exception:  # pragma: no cover
    scipy_eigh = None

try:
    from scipy.sparse.linalg import eigsh as scipy_eigsh
except Exception:  # pragma: no cover
    scipy_eigsh = None

@njit(cache=True)
def _jacobi_eigh_small(A):
    n = A.shape[0]
    D = A.copy()
    V = np.eye(n, dtype=np.float64)

    for _ in range(24):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = D[p, q]
                if abs(apq) <= 1e-12:
                    continue

                app = D[p, p]
                aqq = D[q, q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))

                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c

                for r in range(n):
                    if r != p and r != q:
                        drp = D[r, p]
                        drq = D[r, q]
                        new_rp = c * drp - s * drq
                        new_rq = s * drp + c * drq
                        D[r, p] = new_rp
                        D[p, r] = new_rp
                        D[r, q] = new_rq
                        D[q, r] = new_rq

                D[p, p] = app - t * apq
                D[q, q] = aqq + t * apq
                D[p, q] = 0.0
                D[q, p] = 0.0

                for r in range(n):
                    vrp = V[r, p]
                    vrq = V[r, q]
                    V[r, p] = c * vrp - s * vrq
                    V[r, q] = s * vrp + c * vrq

                changed = True

        if not changed:
            break

    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        w[i] = D[i, i]

    order = np.argsort(w)
    ws = np.empty(n, dtype=np.float64)
    Vs = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        idx = order[j]
        ws[j] = w[idx]
        for i in range(n):
            Vs[i, j] = V[i, idx]

    return ws, Vs

@njit(cache=True)
def _small_solve_numba(A, n_components, sparsity_param):
    n = A.shape[0]
    X = np.zeros((n, n_components), dtype=np.float64)
    explained = np.zeros(n_components, dtype=np.float64)

    if n_components == 0:
        return X, explained

    eigvals, eigvecs = _jacobi_eigh_small(A)

    first_pos = 0
    while first_pos < n and eigvals[first_pos] <= 0.0:
        first_pos += 1

    pos_count = n - first_pos
    k = pos_count if pos_count < n_components else n_components
    half_lambda = 0.5 * sparsity_param

    for j in range(k):
        src = n - 1 - j
        scale = np.sqrt(eigvals[src])
        norm2 = 0.0

        for i in range(n):
            b = eigvecs[i, src] * scale
            if b > half_lambda:
                y = b - half_lambda
            elif b < -half_lambda:
                y = b + half_lambda
            else:
                y = 0.0
            X[i, j] = y
            norm2 += y * y

        if norm2 > 1.0:
            inv_norm = 1.0 / np.sqrt(norm2)
            for i in range(n):
                X[i, j] *= inv_norm

    for j in range(n_components):
        var = 0.0
        for r in range(n):
            xr = X[r, j]
            if xr != 0.0:
                row_dot = 0.0
                for c in range(n):
                    row_dot += A[r, c] * X[c, j]
                var += xr * row_dot
        explained[j] = var

    return X, explained

class Solver:
    def __init__(self):
        try:
            _small_solve_numba(np.eye(2, dtype=np.float64), 1, 0.1)
        except Exception:
            pass

    def _top_scaled_eigvecs(self, A: np.ndarray, n_components: int) -> np.ndarray:
        n = A.shape[0]
        if n_components <= 0 or n == 0:
            return np.empty((n, 0), dtype=np.float64)

        q = min(n, n_components)
        eigvals = None
        eigvecs = None

        if scipy_eigsh is not None and n >= 64 and q <= 8 and q * 5 <= n:
            try:
                eigvals, eigvecs = scipy_eigsh(
                    A,
                    k=q,
                    which="LA",
                    tol=1e-5,
                    ncv=min(n, max(4 * q + 1, 20)),
                )
                order = np.argsort(eigvals)
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
            except Exception:
                eigvals = None
                eigvecs = None

        if eigvals is None:
            if scipy_eigh is not None:
                try:
                    if n <= 40 or q * 3 >= n:
                        eigvals, eigvecs = scipy_eigh(
                            A,
                            check_finite=False,
                            driver="evd",
                        )
                    else:
                        eigvals, eigvecs = scipy_eigh(
                            A,
                            subset_by_index=[n - q, n - 1],
                            check_finite=False,
                            driver="evr",
                        )
                except Exception:
                    eigvals = None
                    eigvecs = None
                    eigvals = None
                    eigvecs = None

        if eigvals is None:
            eigvals, eigvecs = np.linalg.eigh(A)

        first_pos = np.searchsorted(eigvals, 0.0, side="right")
        if first_pos >= eigvals.shape[0]:
            return np.empty((n, 0), dtype=np.float64)

        eigvals = eigvals[first_pos:]
        eigvecs = eigvecs[:, first_pos:]
        k = min(n_components, eigvals.shape[0])
        vals = eigvals[-k:][::-1]
        vecs = eigvecs[:, -k:][:, ::-1]
        return vecs * np.sqrt(vals)

    def solve(self, problem, **kwargs):
        try:
            raw_A = problem["covariance"]

            n_components = int(problem["n_components"])
            if n_components < 0:
                return {"components": [], "explained_variance": []}

            if isinstance(raw_A, np.ndarray):
                n = raw_A.shape[0]
                if raw_A.ndim != 2 or raw_A.shape[1] != n:
                    return {"components": [], "explained_variance": []}
            else:
                n = len(raw_A)

            if n <= 16:
                A = raw_A
                if not isinstance(A, np.ndarray) or A.dtype != np.float64:
                    A = np.asarray(A, dtype=np.float64)
                if A.ndim != 2 or A.shape[1] != n:
                    return {"components": [], "explained_variance": []}

                sparsity_param = float(problem["sparsity_param"])
                X, explained = _small_solve_numba(A, n_components, sparsity_param)
                return {
                    "components": X,
                    "explained_variance": explained,
                }

            A = raw_A
            if not isinstance(A, np.ndarray) or A.dtype != np.float64:
                A = np.asarray(A, dtype=np.float64)

            if A.ndim != 2 or A.shape[1] != n:
                return {"components": [], "explained_variance": []}

            if n_components == 0:
                return {
                    "components": np.empty((n, 0), dtype=np.float64),
                    "explained_variance": np.empty(0, dtype=np.float64),
                }

            sparsity_param = float(problem["sparsity_param"])
            B = self._top_scaled_eigvecs(A, n_components)
            k = B.shape[1]

            if k:
                half_lambda = 0.5 * sparsity_param
                Xk = np.sign(B) * np.maximum(np.abs(B) - half_lambda, 0.0)
                norms = np.sqrt(np.einsum("ij,ij->j", Xk, Xk))
                big = norms > 1.0
                if np.any(big):
                    Xk[:, big] /= norms[big]

                if k == n_components:
                    X = Xk
                else:
                    X = np.zeros((n, n_components), dtype=np.float64)
                    X[:, :k] = Xk
            else:
                X = np.zeros((n, n_components), dtype=np.float64)

            AX = A @ X
            explained_variance = np.einsum("ij,ij->j", X, AX)

            return {
                "components": X,
                "explained_variance": explained_variance,
            }
        except Exception:
            return {"components": [], "explained_variance": []}