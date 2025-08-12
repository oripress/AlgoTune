from __future__ import annotations

from typing import Any, Dict

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fast solver via dynamic programming (finite-horizon LQR with tracking).

        Eliminating v yields the equivalent problem:
            minimize_{x,w} sum_{t=0}^{N-1} ( ||w_t||^2 + tau ||y_t - C x_t||^2 )
            subject to x_{t+1} = A x_t + B w_t,  x_0 given.

        This is a convex quadratic optimal control problem. Its exact solution is
        obtained by a Riccati recursion with an additional linear term due to tracking,
        followed by a forward rollout:
            w_t = -K_t x_t + g_t,
            x_{t+1} = A x_t + B w_t,
            v_t = y_t - C x_t.
        """
        # Parse inputs
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])

        # Dimensions
        N, m = y.shape
        n = A.shape[0]
        p = B.shape[1]

        # Trivial horizon case
        if N == 0:
            return {
                "x_hat": np.atleast_2d(x0).tolist(),
                "w_hat": np.zeros((0, p), dtype=float).tolist(),
                "v_hat": np.zeros((0, m), dtype=float).tolist(),
            }

        # If no control input (p == 0): dynamics are autonomous; w is empty.
        if p == 0:
            x_hat = np.empty((N + 1, n), dtype=float)
            x_hat[0] = x0
            for t in range(N):
                x_hat[t + 1] = A @ x_hat[t]
            v_hat = np.empty((N, m), dtype=float)
            for t in range(N):
                v_hat[t] = y[t] - C @ x_hat[t] if m else np.zeros(0, dtype=float)
            return {
                "x_hat": x_hat.tolist(),
                "w_hat": np.zeros((N, 0), dtype=float).tolist(),
                "v_hat": v_hat.tolist(),
            }

        # Precompute quadratic tracking terms
        # Q = tau * C^T C,  c_t = tau * C^T y_t  (we build all c_t at once)
        if m:
            Q = tau * (C.T @ C)
            c_all = tau * (y @ C)  # shape (N, n); each row is c_t^T
        else:
            Q = np.zeros((n, n), dtype=float)
            c_all = np.zeros((N, n), dtype=float)

        # Backward Riccati with linear term
        P = np.zeros((n, n), dtype=float)  # terminal P_N = 0
        s = np.zeros(n, dtype=float)       # terminal s_N = 0

        K_arr = np.empty((N, p, n), dtype=float)
        g_arr = np.empty((N, p), dtype=float)

        I_p = np.eye(p, dtype=float)

        for t in range(N - 1, -1, -1):
            Pn = P
            sn = s

            BtP = B.T @ Pn                 # (p, n)
            H = I_p + BtP @ B              # (p, p) = I + B^T P B

            # Solve for gains: K = H^{-1} B^T P A,  g = H^{-1} B^T s
            RHS_K = BtP @ A                # (p, n)
            K = np.linalg.solve(H, RHS_K)  # (p, n)
            g = np.linalg.solve(H, B.T @ sn)  # (p,)

            # Update P and s:
            PA = Pn @ A                    # (n, n)
            PB = Pn @ B                    # (n, p)
            P = Q + A.T @ PA - A.T @ PB @ K
            # Enforce symmetry to reduce numerical drift
            P = 0.5 * (P + P.T)

            s = c_all[t] + A.T @ sn - A.T @ (PB @ g)

            K_arr[t] = K
            g_arr[t] = g

        # Forward rollout to compute optimal trajectories
        x_hat = np.empty((N + 1, n), dtype=float)
        w_hat = np.empty((N, p), dtype=float)
        v_hat = np.empty((N, m), dtype=float)

        x_hat[0] = x0
        for t in range(N):
            # w_t = -K_t x_t + g_t
            w_hat[t] = -K_arr[t] @ x_hat[t] + g_arr[t]
            x_hat[t + 1] = A @ x_hat[t] + B @ w_hat[t]
            v_hat[t] = y[t] - C @ x_hat[t] if m else np.zeros(0, dtype=float)

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }