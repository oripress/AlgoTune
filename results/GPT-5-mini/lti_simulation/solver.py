import numpy as np
from scipy import signal
from scipy.linalg import expm
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Simulate a continuous-time SISO LTI system given by transfer function num/den
        for an input u sampled at times t.

        Uses tf2ss + matrix exponential with an augmented matrix to handle
        piecewise-linear inputs exactly on each interval.
        """
        # Parse inputs
        num = np.asarray(problem.get("num", []), dtype=float).ravel()
        den = np.asarray(problem.get("den", []), dtype=float).ravel()
        t = np.asarray(problem.get("t", []), dtype=float).ravel()
        u = np.asarray(problem.get("u", []), dtype=float).ravel()

        # Basic checks / trivial cases
        if t.size == 0:
            return {"yout": []}
        if u.size == 0:
            u = np.zeros_like(t)
        if u.size == 1 and t.size > 1:
            u = np.full_like(t, float(u[0]))
        if u.size != t.size:
            raise ValueError("Length of 'u' must match length of 't'")

        # Convert transfer function to state-space; fallback to lsim if conversion fails
        try:
            A, B, C, D = signal.tf2ss(num, den)
        except Exception:
            system = signal.lti(num, den)
            _, yout, _ = signal.lsim(system, u, t)
            return {"yout": yout.tolist()}

        A = np.atleast_2d(np.asarray(A, dtype=float))
        n = A.shape[0]
        if n == 0:
            # Pure gain (no dynamics)
            D_scalar = float(np.asarray(D, dtype=float).squeeze())
            return {"yout": (D_scalar * u).astype(float).tolist()}

        def to_vec(arr, n):
            a = np.asarray(arr, dtype=float).ravel()
            if a.size >= n:
                return a[:n].copy().astype(float)
            return np.pad(a, (0, n - a.size), "constant").astype(float)

        B = to_vec(B, n)
        C = to_vec(C, n)
        D_scalar = float(np.asarray(D, dtype=float).squeeze())

        # State and output initialization
        x = np.zeros(n, dtype=float)
        nt = t.size
        yout = np.empty(nt, dtype=float)
        yout[0] = float(np.dot(C, x) + D_scalar * u[0])

        # Cache exponentials for repeated dt values
        exp_cache: Dict[float, tuple] = {}

        for k in range(nt - 1):
            dt = float(t[k + 1] - t[k])

            if dt <= 0.0:
                E11 = np.eye(n, dtype=float)
                R0 = np.zeros(n, dtype=float)
                R1p = np.zeros(n, dtype=float)
            else:
                key = dt
                cached = exp_cache.get(key)
                if cached is None:
                    # Augmented matrix trick:
                    # M = [[A, B, 0],
                    #      [0, 0, 1],
                    #      [0, 0, 0]]
                    # exp(M*dt) yields blocks containing:
                    # E11 = exp(A*dt)
                    # R0  = ∫_0^dt exp(A s) B ds
                    # R1p = ∫_0^dt (dt - s) exp(A s) B ds
                    M = np.zeros((n + 2, n + 2), dtype=float)
                    M[:n, :n] = A
                    M[:n, n] = B
                    M[n, n + 1] = 1.0
                    Em = expm(M * dt)
                    E11 = Em[:n, :n]
                    R0 = Em[:n, n].copy()
                    R1p = Em[:n, n + 1].copy()
                    exp_cache[key] = (E11, R0, R1p)
                else:
                    E11, R0, R1p = cached

            # slope of input on interval (u is piecewise-linear)
            r = (u[k + 1] - u[k]) / dt if dt != 0.0 else 0.0

            # state update for piecewise-linear input on [t_k, t_{k+1}]
            x = E11.dot(x) + u[k] * R0 + r * R1p

            # output at next time point
            yout[k + 1] = float(np.dot(C, x) + D_scalar * u[k + 1])

        # Ensure purely real output
        if np.iscomplexobj(yout):
            if np.allclose(yout.imag, 0.0, atol=1e-12):
                yout = yout.real
            else:
                yout = yout.real

        return {"yout": yout.tolist()}