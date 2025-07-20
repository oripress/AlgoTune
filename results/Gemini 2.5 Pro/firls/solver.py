import numpy as np
from scipy import linalg
from numba import jit
# The linter might not know about prange, so we use a workaround
try:
    from numba import prange
except ImportError:
    prange = range

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def build_q_and_c(m, f1, f2):
    """
    Constructs the Q matrix and c vector for the least-squares FIR problem.
    This version is based on the hypothesis that the reference firls
    implementation omits the sin(pi*m) term from the analytical formula
    for the Q matrix calculation when m is not zero, possibly for
    numerical stability.
    """
    Q = np.zeros((m, m), dtype=np.float64)
    c = np.zeros(m, dtype=np.float64)
    pi = np.pi

    # c_k = integral D(w)cos(kw)dw over passband
    for k in prange(m):
        if k == 0:
            c[k] = f1
        else:
            c[k] = np.sin(pi * f1 * k) / (pi * k)

    # Q_ij = 0.5 * [I(i-j) + I(i+j)]
    # where I(m) = integral cos(mw)dw over bands
    for i in prange(m):
        for j in range(i, m):
            # Calculate I(i-j)
            m1 = float(i - j)
            if m1 == 0.0:
                I_m1 = f1 + 1.0 - f2
            else:
                pi_m1 = pi * m1
                I_m1 = (np.sin(pi_m1 * f1) - np.sin(pi_m1 * f2)) / pi_m1
            
            # Calculate I(i+j)
            m2 = float(i + j)
            if m2 == 0.0:
                I_m2 = f1 + 1.0 - f2
            else:
                pi_m2 = pi * m2
                I_m2 = (np.sin(pi_m2 * f1) - np.sin(pi_m2 * f2)) / pi_m2

            val = 0.5 * (I_m1 + I_m2)
            Q[i, j] = val
            Q[j, i] = val # Exploit symmetry
            
    return Q, c
    return Q, c

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        n_half, edges = problem
        edges = tuple(edges)
        
        N = 2 * n_half + 1
        m = n_half + 1
        f1, f2 = edges

        Q, c = build_q_and_c(m, f1, f2)

        try:
            # Use a specialized solver for symmetric matrices
            a = linalg.solve(Q, c, assume_a='sym')
        except linalg.LinAlgError:
            # Fallback to reference if matrix is singular
            from scipy import signal
            return signal.firls(N, (0.0, *edges, 1.0), [1, 1, 0, 0])

        # Convert ideal filter coefficients 'a' to FIR coefficients 'h'
        h = np.zeros(N, dtype=np.float64)
        h[n_half] = a[0]
        if m > 1:
            k_coeffs = np.arange(1, m)
            h[n_half - k_coeffs] = a[1:] / 2
            h[n_half + k_coeffs] = a[1:] / 2
            
        return h