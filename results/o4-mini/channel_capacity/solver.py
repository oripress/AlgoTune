import numpy as np
import numba
import math

@numba.njit(fastmath=True)
def _blahut_numba(P, x, tol, max_iter):
    m, n = P.shape
    # Precompute log(P)
    logP = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            val = P[i, j]
            if val > 0.0:
                logP[i, j] = math.log(val)
            else:
                logP[i, j] = 0.0
    C_prev = 0.0
    for _ in range(max_iter):
        # Compute output distribution y = P @ x
        y = np.zeros(m, dtype=np.float64)
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += P[i, j] * x[j]
            y[i] = s
        # Compute log(y)
        logy = np.empty(m, dtype=np.float64)
        for i in range(m):
            val = y[i]
            if val > 0.0:
                logy[i] = math.log(val)
            else:
                logy[i] = 0.0
        # Compute divergence D_j
        D = np.empty(n, dtype=np.float64)
        for j in range(n):
            s = 0.0
            for i in range(m):
                s += P[i, j] * (logP[i, j] - logy[i])
            D[j] = s
        # Update x
        total = 0.0
        for j in range(n):
            x[j] = x[j] * math.exp(D[j])
            total += x[j]
        if total <= 0.0:
            break
        inv_total = 1.0 / total
        for j in range(n):
            x[j] *= inv_total
        # Compute capacity in nats
        C = 0.0
        for j in range(n):
            C += x[j] * D[j]
        if C - C_prev < tol:
            C_prev = C
            break
        C_prev = C
    return x, C_prev

class Solver:
    def __init__(self):
        # JIT‐compile BA routine (not counted in solve runtime)
        dummy_P = np.ones((1, 1), dtype=np.float64)
        dummy_x = np.ones(1, dtype=np.float64)
        _blahut_numba(dummy_P, dummy_x, 1e-8, 1)

    def solve(self, problem, **kwargs):
        P_list = problem.get("P")
        if P_list is None:
            return None
        P = np.array(P_list, dtype=np.float64)
        if P.ndim != 2:
            return None
        m, n = P.shape
        if m <= 0 or n <= 0:
            return None
        if not np.allclose(P.sum(axis=0), 1.0, atol=1e-8):
            return None
        # Uniform start
        x0 = np.full(n, 1.0 / n, dtype=np.float64)
        # BA convergence parameters (nats)
        tol = 1e-12
        max_iter = 50000
        # Run Blahut–Arimoto algorithm and get capacity in nats
        x_opt, C_nats = _blahut_numba(P, x0, tol, max_iter)
        # Convert capacity to bits
        C_bits = float(C_nats / math.log(2))
        return {"x": x_opt.tolist(), "C": C_bits}