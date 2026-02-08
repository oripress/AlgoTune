import numpy as np
from scipy.linalg import expm
import numba

@numba.njit(cache=True)
def _propagate(Ad, Bd1, Bd2, C_flat, D_val, u, n):
    nstates = Ad.shape[0]
    x = np.zeros(nstates)
    yout = np.empty(n)
    for k in range(n):
        y = D_val * u[k]
        for i in range(nstates):
            y += C_flat[i] * x[i]
        yout[k] = y
        if k < n - 1:
            x_new = np.empty(nstates)
            uk = u[k]
            uk1 = u[k + 1]
            for i in range(nstates):
                s = Bd1[i] * uk + Bd2[i] * uk1
                for j in range(nstates):
                    s += Ad[i, j] * x[j]
                x_new[i] = s
            x = x_new
    return yout

def _tf2ss_fast(num, den):
    """Minimal tf2ss: controllable canonical form."""
    den0 = den[0]
    if den0 != 1.0:
        num = num / den0
        den = den / den0
    
    nn = len(num)
    nd = len(den)
    M = max(nn, nd)
    
    if nn < M:
        num_pad = np.zeros(M)
        num_pad[M - nn:] = num
        num = num_pad
    
    n = M - 1
    if n == 0:
        return np.empty((0, 0)), np.empty((0, 1)), np.empty((1, 0)), num[0:1].reshape(1, 1)
    
    D_val = num[0]
    
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    for i in range(n):
        A[n - 1, i] = -den[n - i]
    
    B = np.zeros((n, 1))
    B[n - 1, 0] = 1.0
    
    C = np.zeros((1, n))
    for i in range(n):
        C[0, i] = num[n - i] - den[n - i] * D_val
    
    D = np.array([[D_val]])
    return A, B, C, D

class Solver:
    def __init__(self):
        _propagate(
            np.zeros((1, 1)), np.zeros(1), np.zeros(1),
            np.zeros(1), 0.0, np.zeros(2), 2
        )
        _propagate(
            np.zeros((2, 2)), np.zeros(2), np.zeros(2),
            np.zeros(2), 0.0, np.zeros(3), 3
        )

    def solve(self, problem, **kwargs):
        num = np.atleast_1d(np.asarray(problem["num"], dtype=np.float64))
        den = np.atleast_1d(np.asarray(problem["den"], dtype=np.float64))
        u = np.asarray(problem["u"], dtype=np.float64)
        t = np.asarray(problem["t"], dtype=np.float64)

        n = len(t)
        if n == 0:
            return {"yout": []}

        A, B, C, D = _tf2ss_fast(num, den)
        nstates = A.shape[0]
        D_val = float(D.flat[0])

        if nstates == 0:
            return {"yout": (D_val * u).tolist()}

        if n == 1:
            return {"yout": [D_val * u[0]]}

        C_flat = C[0, :]
        b = B[:, 0]

        # Check uniform spacing efficiently
        dt = t[1] - t[0]
        if n <= 2:
            uniform = True
        else:
            dt_arr = np.diff(t)
            tol = 1e-8 * max(abs(dt), 1.0)
            uniform = (np.max(np.abs(dt_arr - dt)) < tol)

        if uniform and dt > 0:
            aug = nstates + 2
            Em = np.zeros((aug, aug))
            Em[:nstates, :nstates] = A * dt
            Em[:nstates, nstates] = b * dt
            Em[nstates, nstates + 1] = dt

            ms = expm(Em)

            Ad = np.ascontiguousarray(ms[:nstates, :nstates])
            cd = ms[:nstates, nstates]
            dd = ms[:nstates, nstates + 1]

            Bd1 = cd - dd / dt
            Bd2 = dd / dt

            yout = _propagate(Ad, Bd1, Bd2, C_flat, D_val, u, n)
            return {"yout": yout.tolist()}
        else:
            from scipy.signal import lsim
            _, yout_arr, _ = lsim((A, B, C, D), u, t)
            return {"yout": yout_arr.tolist()}