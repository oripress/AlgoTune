import logging
from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solve the MAP Kalman filtering problem using a direct linear‑least‑squares
        formulation. This eliminates the process‑ and measurement‑noise variables
        and solves for the state trajectory in one shot, which is far faster than
        invoking a generic QP solver.

        Parameters
        ----------
        problem : dict
            Dictionary with keys ``A``, ``B``, ``C``, ``y``, ``x_initial`` and ``tau``.

        Returns
        -------
        dict
            ``x_hat`` – list of N+1 state vectors (including the given initial state)  
            ``w_hat`` – list of N process‑noise vectors  
            ``v_hat`` – list of N measurement‑noise vectors
        """
        try:
            # ----- unpack problem -------------------------------------------------
            A = np.asarray(problem["A"], dtype=float)          # (n, n)
            B = np.asarray(problem["B"], dtype=float)          # (n, p)
            C = np.asarray(problem["C"], dtype=float)          # (m, n)
            y = np.asarray(problem["y"], dtype=float)          # (N, m)
            x0 = np.asarray(problem["x_initial"], dtype=float)  # (n,)
            tau = float(problem["tau"])

            N, m = y.shape                     # number of measurements, measurement dim
            n = A.shape[1]                     # state dimension
            p = B.shape[1]                     # process‑noise dimension

            # ----- pre‑compute matrices -------------------------------------------
            # Minimal‑norm process noise: w = pinv(B) (x_{t+1} - A x_t)
            R = np.linalg.pinv(B)               # shape (p, n)
            Q = R.T @ R                         # (n, n) weight for dynamics residuals
            sqrt_tau = np.sqrt(tau)

            # ----- build block‑tridiagonal quadratic system H z = -b  (z = [x1,…,xN]) -----
            # H is (N*n)×(N*n), b is (N*n,)
            H = np.zeros((N * n, N * n))
            b = np.zeros(N * n)

            # Dynamics contributions
            # t = 0 uses known x0
            H[0:n, 0:n] += Q
            b[0:n] += -Q @ (A @ x0)

            for t in range(1, N):
                # indices for x_t and x_{t+1}
                i = (t - 1) * n          # block for x_t
                j = t * n                # block for x_{t+1}

                # x_t term: A^T Q A
                H[i:i + n, i:i + n] += A.T @ Q @ A
                # x_{t+1} term: Q
                H[j:j + n, j:j + n] += Q
                # cross terms: -Q A and its transpose
                H[j:j + n, i:i + n] += -Q @ A
                H[i:i + n, j:j + n] += -(A.T @ Q)

            # Measurement contributions
            CtC = C.T @ C
            for t in range(1, N):
                i = (t - 1) * n
                H[i:i + n, i:i + n] += tau * CtC
                b[i:i + n] += -tau * (C.T @ y[t])

            # Solve the linear system (H is symmetric positive definite)
            try:
                z = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                z, *_ = np.linalg.lstsq(H, -b, rcond=None)

            # ----- reconstruct full state trajectory -------------------------------
            x_hat = np.vstack((x0.reshape(1, -1), z.reshape(N, n)))   # (N+1, n)

            # ----- recover optimal w and v ----------------------------------------
            w_hat: List[np.ndarray] = []
            v_hat: List[np.ndarray] = []
            for t in range(N):
                w_t = R @ (x_hat[t + 1] - A @ x_hat[t])
                w_hat.append(w_t)

                v_t = y[t] - C @ x_hat[t]
                v_hat.append(v_t)

            # Convert to plain Python lists for the required output format
            return {
                "x_hat": x_hat.tolist(),
                "w_hat": [w.tolist() for w in w_hat],
                "v_hat": [v.tolist() for v in v_hat],
            }

        except Exception as exc:
            logging.error("Kalman MAP solver failed: %s", exc)
            return {"x_hat": [], "w_hat": [], "v_hat": []}