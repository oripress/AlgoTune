from __future__ import annotations

from typing import Any, Dict

import numpy as np

# Optional fast SPD solve (Cholesky). Import once at module load.
_HAS_SCIPY_CHO = False
try:  # pragma: no cover
    from scipy.linalg import cho_factor, cho_solve  # type: ignore

    _HAS_SCIPY_CHO = True
except Exception:  # pragma: no cover
    pass

class Solver:
    """
    Fast solver for the Kalman-filter QP via elimination + Riccati recursion.

    Eliminate v_t = y_t - C x_t:
        min_{x,w} Σ ||w_t||^2 + tau Σ ||y_t - C x_t||^2
        s.t. x_{t+1} = A x_t + B w_t, x_0 fixed.

    Solve using backward Riccati recursion with affine term (tracking LQR),
    then forward simulate. Complexity O(N) with small dense linear algebra.
    """

    def __init__(self) -> None:
        pass

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        try:
            A = np.asarray(problem["A"], dtype=np.float64)
            B = np.asarray(problem["B"], dtype=np.float64)
            C = np.asarray(problem["C"], dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64)
            x0 = np.asarray(problem["x_initial"], dtype=np.float64).reshape(-1)
            tau = float(problem["tau"])
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if A.ndim != 2 or B.ndim != 2 or C.ndim != 2 or y.ndim != 2:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        N = int(y.shape[0])
        m = int(y.shape[1])
        n = int(A.shape[1])
        p = int(B.shape[1])

        if x0.shape != (n,):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if N == 0:
            x_hat = x0.reshape(1, n)
            w_hat = np.empty((0, p), dtype=np.float64)
            v_hat = np.empty((0, m), dtype=np.float64)
            return {"x_hat": x_hat, "w_hat": w_hat, "v_hat": v_hat}

        At = A.T
        Bt = B.T
        Ct = C.T

        Q = tau * (Ct @ C)  # (n,n)
        b_all = tau * (y @ C)  # (N,n)

        # Backward recursion storage
        Pn = np.zeros((n, n), dtype=np.float64)  # P_{t+1}
        sn = np.zeros(n, dtype=np.float64)  # s_{t+1}

        K_all = np.empty((N, p, n), dtype=np.float64)
        f_all = np.empty((N, p), dtype=np.float64)

        # Reusable RHS buffer for linear solves (saves per-iteration allocations)
        rhs = np.empty((p, n + 1), dtype=np.float64)

        # Backward pass
        try:
            if _HAS_SCIPY_CHO:
                for t in range(N - 1, -1, -1):
                    PB = Pn @ B  # (n,p)

                    # H = I + B^T P B (SPD), build without allocating identity
                    H = Bt @ PB  # (p,p)
                    H.flat[:: p + 1] += 1.0

                    # G = B^T P A
                    G = PB.T @ A  # (p,n)
                    g0 = Bt @ sn  # (p,)

                    # One factorization, one solve with multiple RHS; overwrite rhs in-place
                    c, lower = cho_factor(H, lower=True, overwrite_a=False, check_finite=False)
                    rhs[:, :n] = G
                    rhs[:, n] = g0
                    sol = cho_solve((c, lower), rhs, overwrite_b=True, check_finite=False)

                    K = sol[:, :n]
                    f = sol[:, n]

                    K_all[t] = K
                    f_all[t] = f

                    PA = Pn @ A
                    Pn = Q + At @ PA - (G.T @ K)
                    sn = b_all[t] + At @ sn - (G.T @ f)
            else:
                for t in range(N - 1, -1, -1):
                    PB = Pn @ B

                    H = Bt @ PB
                    H.flat[:: p + 1] += 1.0

                    G = PB.T @ A
                    g0 = Bt @ sn

                    rhs[:, :n] = G
                    rhs[:, n] = g0
                    sol = np.linalg.solve(H, rhs)

                    K = sol[:, :n]
                    f = sol[:, n]

                    K_all[t] = K
                    f_all[t] = f

                    PA = Pn @ A
                    Pn = Q + At @ PA - (G.T @ K)
                    sn = b_all[t] + At @ sn - (G.T @ f)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Forward pass
        x_hat = np.empty((N + 1, n), dtype=np.float64)
        w_hat = np.empty((N, p), dtype=np.float64)
        x_hat[0] = x0

        try:
            for t in range(N):
                xt = x_hat[t]
                wt = -K_all[t] @ xt + f_all[t]
                w_hat[t] = wt
                x_hat[t + 1] = A @ xt + B @ wt
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        v_hat = y - (x_hat[:N] @ Ct)

        if not (np.isfinite(x_hat).all() and np.isfinite(w_hat).all() and np.isfinite(v_hat).all()):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {"x_hat": x_hat, "w_hat": w_hat, "v_hat": v_hat}