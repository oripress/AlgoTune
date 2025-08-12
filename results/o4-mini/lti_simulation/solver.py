import numpy as np
from scipy.signal import tf2ss, lti, lsim
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        # Parse inputs
        num = np.array(problem["num"], dtype=float)
        den = np.array(problem["den"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)
        n = t.size

        # Fast path: uniform time steps with FOH discretization
        if n >= 2:
            dt = t[1] - t[0]
            tol = 1e-8 * (1.0 + abs(dt))
            if dt > 0 and np.all(np.abs(np.diff(t) - dt) <= tol):
                try:
                    # State-space conversion
                    A, B, C, D = tf2ss(num, den)
                    B = B.reshape(-1)
                    C = C.reshape(-1)
                    D = float(D)

                    # Matrix exponential
                    E = expm(A * dt)
                    I = np.eye(A.shape[0])

                    # Compute G = ∫0^dt exp(A s) ds * B
                    G = np.linalg.solve(A, (E - I).dot(B))
                    # Compute J = ∫0^dt s exp(A s) ds * B
                    M = E - I - A * dt
                    J = np.linalg.solve(A, np.linalg.solve(A, M.dot(B)))

                    # Discrete FOH input matrices
                    B_d1 = J / dt
                    B_d0 = G - B_d1

                    # Simulate discrete-time FOH response
                    x = np.zeros_like(B)
                    y = np.empty(n, dtype=float)
                    y[0] = C.dot(x) + D * u[0]
                    for k in range(n - 1):
                        x = E.dot(x) + B_d0 * u[k] + B_d1 * u[k + 1]
                        y[k + 1] = C.dot(x) + D * u[k + 1]
                    return {"yout": y.tolist()}
                except Exception:
                    pass

        # Fallback: continuous-time simulation
        system = lti(num, den)
        _, yout, _ = lsim(system, u, t)
        return {"yout": yout.tolist()}