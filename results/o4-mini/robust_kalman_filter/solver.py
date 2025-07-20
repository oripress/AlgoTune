import numba

@numba.njit(cache=True)
def _block_tridiag_solve_numba(D, L, U, b):
    N, n, _ = D.shape
    P = np.empty((N, n, n))
    bp = np.empty((N, n))
    P[0] = np.linalg.inv(D[0])
    bp[0] = b[0]
    for i in range(1, N):
        S = L[i] @ P[i - 1]
        Dp = D[i] - S @ U[i - 1]
        P[i] = np.linalg.inv(Dp)
        bp[i] = b[i] - S @ bp[i - 1]
    X = np.empty((N, n))
    X[N - 1] = P[N - 1] @ bp[N - 1]
    for i in range(N - 2, -1, -1):
        X[i] = P[i] @ (bp[i] - U[i] @ X[i + 1])
    return X
import numpy as np
import cvxpy as cp
from typing import Any, Dict, List

class Solver:
    def __init__(self):
        import numpy as _np
        # Warm-up compile of the block-tridiagonal solver
        D_dummy = _np.ones((1, 1, 1))
        L_dummy = _np.zeros((1, 1, 1))
        U_dummy = _np.zeros((1, 1, 1))
        b_dummy = _np.zeros((1, 1))
        _block_tridiag_solve_numba(D_dummy, L_dummy, U_dummy, b_dummy)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Robust Kalman smoothing via IRLS and block-tridiagonal solve when B is square and invertible.
        Fallback to CVXPY if B not square/invertible.
        """
        # Parse inputs
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        C = np.array(problem["C"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        x0 = np.array(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Try optimized branch if B is square and invertible
        if B.shape[0] == B.shape[1]:
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                # fallback to CVXPY
                return self._cvxsolve(A, B, C, y, x0, tau, M)
            # Precompute for IRLS
            Q = B_inv.T @ B_inv  # dynamic penalty matrix
            CtC = C.T @ C
            S0 = Q + A.T @ (Q @ A)
            U_const = -A.T @ Q
            L_const = -Q @ A
            Cty = y @ C  # shape N x n
            # Initial guess: propagate dynamics
            x = np.zeros((N + 1, n))
            x[0] = x0
            for t in range(N):
                x[t + 1] = A @ x[t]
            # Preallocate block matrices
            D = np.empty((N, n, n))
            L_mat = np.empty((N, n, n))
            U_mat = np.empty((N, n, n))
            b = np.empty((N, n))
            # Build constant L and U
            L_mat[:] = L_const
            U_mat[:] = U_const
            L_mat[0] = np.zeros((n, n))
            U_mat[-1] = np.zeros((n, n))
            # IRLS iterations
            max_iter = 10
            tol = 1e-6
            b0 = Q @ (A @ x0)
            for _ in range(max_iter):
                x_prev = x.copy()
                # Residuals and weights
                v = y - x[:N] @ C.T  # N x m
                norms = np.linalg.norm(v, axis=1)
                beta = np.ones(N)
                mask = norms > M
                beta[mask] = M / norms[mask]
                alpha = tau * beta  # weights for squared residuals
                # Assemble RHS b
                b.fill(0)
                b[:-1] = alpha[1:, None] * Cty[1:]
                b[0] += b0
                # Assemble diagonal blocks D
                D[:-1] = S0
                D[:-1] += alpha[1:, None, None] * CtC
                D[-1] = Q
                # Solve block-tridiagonal system
                X = self._block_tridiag_solve(D, L_mat, U_mat, b)
                # Update states x
                x[1:] = X
                # Check convergence
                if np.max(np.linalg.norm(x - x_prev, axis=1)) < tol:
                    break
            # Recover w and v
            diff = x[1:] - x[:-1] @ A.T
            w = diff @ B_inv.T
            v_hat = y - x[:N] @ C.T
            return {
                "x_hat": x.tolist(),
                "w_hat": w.tolist(),
                "v_hat": v_hat.tolist()
            }
        else:
            # Fallback to CVXPY solver
            return self._cvxsolve(A, B, C, y, x0, tau, M)

    def _block_tridiag_solve(self, D: List[np.ndarray], L: List[np.ndarray],
                             U: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
        """
        Solve block-tridiagonal system with blocks D, L, U and RHS b.
        System: for i=0..N-1:
          L[i] x_{i-1} + D[i] x_i + U[i] x_{i+1} = b[i],
        with L[0]=0, U[N-1]=0.
        """
        N = len(D)
        # Forward elimination
        P: List[np.ndarray] = [None] * N
        bp: List[np.ndarray] = [None] * N
        P[0] = np.linalg.inv(D[0])
        bp[0] = b[0]
        for i in range(1, N):
            S = L[i] @ P[i - 1]
            Dp = D[i] - S @ U[i - 1]
            P[i] = np.linalg.inv(Dp)
            bp[i] = b[i] - S @ bp[i - 1]
        # Back substitution
        X: List[np.ndarray] = [None] * N
        X[N - 1] = P[N - 1] @ bp[N - 1]
        for i in range(N - 2, -1, -1):
            X[i] = P[i] @ (bp[i] - U[i] @ X[i + 1])
        return X

    def _cvxsolve(self, A, B, C, y, x0, tau, M) -> Dict[str, Any]:
        """
        Fallback solver using CVXPY ECOS.
        """
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        # Variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")
        # Objective
        process_cost = cp.sum_squares(w)
        meas_cost = tau * cp.sum([cp.huber(cp.norm(v[t, :]), M) for t in range(N)])
        obj = cp.Minimize(process_cost + meas_cost)
        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True, feastol=1e-4, reltol=1e-4, verbose=False)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        return {"x_hat": x.value.tolist(), "w_hat": w.value.tolist(), "v_hat": v.value.tolist()}