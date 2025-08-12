import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the Kalman-filter QP via a direct KKT linear system.

        Variables z = [x1...xN, w0...w_{N-1}].
        Objective: sum_t ||w_t||^2 + tau * sum_t ||y_t - C x_t||^2
        Dynamics constraints: x_{t+1} - A x_t - B w_t = 0, x0 given.

        This constructs the quadratic terms (H, f) after eliminating v,
        builds equality constraints A_eq z = b, and solves the KKT system:
            [ H   A_eq^T ] [ z     ] = [ -f ]
            [ A_eq  0    ] [ lambda]   [  b ]
        Fallbacks use least-squares / pseudo-inverse for numerical robustness.
        """
        # Parse inputs
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float).reshape(-1)
        tau = float(problem["tau"])

        # Ensure y is 2-D: (N, m)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        N, m = y.shape

        # Dimensions
        # A should be (n, n)
        if A.size == 0:
            n = x0.size
        else:
            # prefer A.shape[1] (consistent with reference)
            n = int(A.shape[1])

        p = int(B.shape[1]) if B.size else 0

        # Quick return for zero timesteps
        if N == 0:
            return {"x_hat": [x0.tolist()], "w_hat": [], "v_hat": []}

        # Variable sizes
        nx = n * N  # x1..xN
        nw = p * N  # w0..w_{N-1}
        z_size = nx + nw

        # Build H and f for (1/2) z^T H z + f^T z + const
        H = np.zeros((z_size, z_size), dtype=float)
        f = np.zeros(z_size, dtype=float)

        # Measurement contributions: tau * sum_{t=0}^{N-1} ||y_t - C x_t||^2
        # x_t is a variable for t >= 1 (x1..xN). For t==0 x0 is fixed.
        if m > 0 and tau != 0.0:
            CTC = C.T @ C
            Ct = C.T
            # t from 1..N-1 correspond to x1..x_{N-1}
            for t in range(1, N):
                xi = (t - 1) * n
                H[xi:xi + n, xi:xi + n] += 2.0 * tau * CTC
                f[xi:xi + n] += -2.0 * tau * (Ct @ y[t])

        # w objective: sum ||w_t||^2 -> contributes 2*I in H (because of 1/2 factor)
        if nw > 0:
            H[nx:nx + nw, nx:nx + nw] = 2.0 * np.eye(nw, dtype=float)

        # Equality constraints from dynamics: x_{t+1} - A x_t - B w_t = 0, t=0..N-1
        K = n * N
        A_eq = np.zeros((K, z_size), dtype=float)
        b = np.zeros(K, dtype=float)

        for t in range(N):
            row = t * n
            # x_{t+1} variable (x1..xN) -> index t
            col_x_next = t * n
            A_eq[row:row + n, col_x_next:col_x_next + n] = np.eye(n)

            if t >= 1:
                # x_t is variable at index (t-1)
                col_x = (t - 1) * n
                A_eq[row:row + n, col_x:col_x + n] = -A
            else:
                # t == 0: move A x0 to RHS -> A_eq * z = b where b = A x0
                b[row:row + n] = A @ x0

            # w_t block
            if p > 0:
                col_w = nx + t * p
                A_eq[row:row + n, col_w:col_w + p] = -B

        # Assemble KKT matrix
        total_dim = z_size + K
        # If there are no optimization variables (z_size == 0), directly solve for lambda?
        if total_dim == 0:
            return {"x_hat": [x0.tolist()], "w_hat": [], "v_hat": []}

        KKT = np.zeros((total_dim, total_dim), dtype=float)
        if z_size > 0:
            KKT[:z_size, :z_size] = H
        if K > 0 and z_size > 0:
            KKT[:z_size, z_size:] = A_eq.T
            KKT[z_size:, :z_size] = A_eq
        elif K > 0 and z_size == 0:
            # degenerate: no z variables, but constraints exist (shouldn't normally happen)
            KKT[:, :] = 0.0

        # RHS
        rhs = np.concatenate([-f, b])

        # Solve the KKT system robustly
        sol = None
        try:
            sol = np.linalg.solve(KKT, rhs)
        except Exception:
            # Try least-squares (handles singular or ill-conditioned KKT)
            try:
                sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            except Exception:
                # Last resort: pseudo-inverse
                try:
                    sol = np.linalg.pinv(KKT) @ rhs
                except Exception:
                    sol = None

        if sol is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Extract z = [x1..xN, w0..w_{N-1}]
        if z_size > 0:
            z = sol[:z_size]
        else:
            z = np.zeros(0, dtype=float)

        # Reshape into per-timestep arrays
        if nx > 0:
            x_vars = z[:nx].reshape((N, n))
        else:
            x_vars = np.zeros((N, n), dtype=float)

        if nw > 0:
            w_vars = z[nx:nx + nw].reshape((N, p))
        else:
            w_vars = np.zeros((N, p), dtype=float)

        # Build outputs
        x_hat = [x0.tolist()]
        for i in range(N):
            x_hat.append(x_vars[i].tolist())

        if p == 0:
            w_hat = [[] for _ in range(N)]
        else:
            w_hat = [w_vars[i].tolist() for i in range(N)]

        # v_t = y_t - C x_t  (x_t is x0 for t=0, else x_{t})
        v_hat = []
        for t in range(N):
            if t == 0:
                x_t = x0
            else:
                x_t = x_vars[t - 1]
            if m == 0:
                v_hat.append([])
            else:
                v_t = (y[t] - (C @ x_t)).tolist()
                v_hat.append(v_t)

        return {"x_hat": x_hat, "w_hat": w_hat, "v_hat": v_hat}