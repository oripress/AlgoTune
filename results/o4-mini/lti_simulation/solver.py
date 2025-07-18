import numpy as np
from scipy.signal import tf2ss, lti, lsim
from scipy.linalg import expm
from numba import njit

@njit(cache=True)
def _simulate_foh(Ad, B0, B1, C, D, u):
    """
    Simulate discrete-time FOH response via state-space update.
    Ad: [nstates,nstates] state transition matrix
    B0, B1: [nstates] FOH input coefficient vectors
    C: [nstates] output mapping vector
    D: scalar direct feedthrough
    u: [n] input signal
    returns y: [n] output signal
    """
    nstates = Ad.shape[0]
    n = u.shape[0]
    y = np.empty(n, dtype=np.float64)
    x = np.zeros(nstates, dtype=np.float64)
    # initial output at t0
    y[0] = D * u[0]
    for k in range(n - 1):
        # state update: x[k+1] = Ad x[k] + B0 u[k] + B1 u[k+1]
        x_new = np.zeros(nstates, dtype=np.float64)
        for i in range(nstates):
            acc = 0.0
            for j in range(nstates):
                acc += Ad[i, j] * x[j]
            x_new[i] = acc + B0[i] * u[k] + B1[i] * u[k+1]
        x = x_new
        # output at next step
        acc2 = 0.0
        for i in range(nstates):
            acc2 += C[i] * x[i]
        y[k + 1] = acc2 + D * u[k + 1]
    return y

# Pre-compile (warm-up) JIT function with minimal data
_dummy = _simulate_foh(
    np.zeros((1, 1), dtype=np.float64),
    np.zeros((1,), dtype=np.float64),
    np.zeros((1,), dtype=np.float64),
    np.zeros((1,), dtype=np.float64),
    0.0,
    np.zeros((1,), dtype=np.float64),
)

class Solver:
    def solve(self, problem, **kwargs):
        # Parse inputs
        num = np.asarray(problem["num"], dtype=float)
        den = np.asarray(problem["den"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)
        n = t.size

        # Fast FOH branch if uniform time-steps
        if n > 1:
            dt = t[1] - t[0]
            if np.allclose(t[1:] - t[:-1], dt):
                try:
                    # Convert to continuous state-space
                    A, B_col, C_row, D = tf2ss(num, den)
                    A = np.asarray(A, dtype=np.float64)
                    B = np.asarray(B_col, dtype=np.float64).flatten()
                    C = np.asarray(C_row, dtype=np.float64).flatten()
                    D = float(D)
                    # Build augmented matrix for integrals
                    nstates = A.shape[0]
                    M = np.zeros((nstates + 2, nstates + 2), dtype=np.float64)
                    M[:nstates, :nstates] = A
                    M[:nstates, nstates] = B
                    M[nstates, nstates + 1] = 1.0
                    EM = expm(M * dt)
                    # Discrete state matrix
                    Ad = EM[:nstates, :nstates]
                    # FOH integrals
                    S0 = EM[:nstates, nstates]         # ∫0^dt exp(Aτ)B dτ
                    M1 = EM[:nstates, nstates + 1]     # ∫0^dt τ exp(Aτ)B dτ
                    B0 = S0 - (M1 / dt)
                    B1 = M1 / dt
                    # Simulate via JIT-compiled loop
                    y = _simulate_foh(Ad, B0, B1, C, D, u)
                    return {"yout": y.tolist()}
                except Exception:
                    pass

        # Fallback to continuous-time simulation
        system = lti(num, den)
        _, y, _ = lsim(system, u, t)
        return {"yout": y.tolist()}