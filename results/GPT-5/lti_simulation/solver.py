from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.linalg import expm

class Solver:
    def _tf_to_ss(self, num: np.ndarray, den: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Convert SISO transfer function (num/den) to controllable canonical form (A,B,C,D).
        Coefficients are given highest power first.
        Assumes a proper transfer function or static gain (n=0). If improper, caller should fallback.
        """
        # Normalize to make denominator monic
        den = np.asarray(den, dtype=float)
        num = np.asarray(num, dtype=float)
        if den[0] == 0.0:
            raise ValueError("Leading denominator coefficient is zero.")
        scale = den[0]
        den = den / scale
        num = num / scale

        n = len(den) - 1  # system order

        # Static gain (n == 0)
        if n == 0:
            if len(num) == 0:
                D = 0.0
            else:
                D = float(num[-1])
            return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), D

        # Companion (controllable canonical) form
        A = np.zeros((n, n), dtype=float)
        if n > 1:
            A[:-1, 1:] = np.eye(n - 1, dtype=float)
        A[-1, :] = -den[1:][::-1]

        B = np.zeros((n,), dtype=float)
        B[-1] = 1.0

        m = len(num) - 1
        if m > n:
            raise ValueError("Improper transfer function (deg(num) > deg(den)).")

        # Direct feedthrough
        D = float(num[0]) if m == n else 0.0

        # Compute bbar = num - D * den; pad if strictly proper
        if m == n:
            bbar = num - D * den
        else:
            bbar = np.zeros((n + 1,), dtype=float)
            if m >= 0:
                bbar[-(m + 1) :] = num

        # Drop the s^n term; b_remain is for s^{n-1}..s^0 (highest-first), reverse to ascending
        b_remain = bbar[1:]
        C = b_remain[::-1].copy()

        return A, B, C, D

    def _simulate_uniform_foh(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: float, t: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """
        Simulate continuous-time LTI with piecewise linear input over uniformly spaced t using a single
        matrix exponential of an augmented system to obtain an O(N) recurrence.

        x_{k+1} = e^{Ah} x_k + G0 u_k + (K1/h) (u_{k+1} - u_k)
        y_k = C x_k + D u_k
        where:
          G0 = ∫_0^h e^{A(h - τ)} B dτ = ∫_0^h e^{Aσ} B dσ
          K1 = ∫_0^h e^{A(h - τ)} B τ dτ
        """
        n = A.shape[0]
        N = t.size

        # Initial condition x0 = 0
        x = np.zeros((n,), dtype=float)
        y = np.empty((N,), dtype=float)

        if n == 0:
            y[:] = D * u
            return y

        y[0] = float(C @ x + D * u[0])
        if N == 1:
            return y

        h = t[1] - t[0]

        # Build augmented matrix for simultaneous computation of e^{Ah}, G0, and K1
        # M = [[A, B, 0],
        #      [0, 0, 1],
        #      [0, 0, 0]]
        M = np.zeros((n + 2, n + 2), dtype=float)
        M[:n, :n] = A
        M[:n, n] = B  # coupling to u
        M[n, n + 1] = 1.0  # du/dt = v, dv/dt = 0

        S = expm(M * h)

        eAh = S[:n, :n]
        G0 = S[:n, n]        # ∫_0^h e^{A(h - τ)} B dτ = ∫_0^h e^{Aσ} B dσ
        K1 = S[:n, n + 1]    # ∫_0^h e^{A(h - τ)} B τ dτ

        inv_h = 1.0 / h

        # Iterate
        for k in range(N - 1):
            uk = u[k]
            duk = u[k + 1] - uk
            # Update state
            x = eAh @ x + G0 * uk + (K1 * inv_h) * duk
            # Output at next time
            y[k + 1] = float(C @ x + D * u[k + 1])

        return y

    def solve(self, problem, **kwargs) -> Dict[str, List[float]]:
        # Extract and validate inputs
        num = np.asarray(problem["num"], dtype=float)
        den = np.asarray(problem["den"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)

        N = t.size
        if N == 0:
            return {"yout": []}
        if u.size != N:
            # Fall back to robust SciPy path
            tout, yout, _ = signal.lsim(signal.lti(num, den), U=u, T=t)
            return {"yout": yout.tolist()}

        # Handle static (order 0) quickly
        if len(den) == 1:
            gain = float(num[0] / den[0]) if len(num) > 0 else 0.0
            return {"yout": (gain * u).tolist()}

        # Try fast path: proper TF and uniform grid
        try:
            A, B, C, D = self._tf_to_ss(num, den)
        except Exception:
            # Fallback to SciPy robust solver
            system = signal.lti(num, den)
            tout, yout, _ = signal.lsim(system, U=u, T=t)
            return {"yout": yout.tolist()}

        # Detect uniform sampling (strict)
        if N >= 2:
            dt = np.diff(t)
            uniform = np.allclose(dt, dt[0], rtol=1e-12, atol=1e-15)
        else:
            uniform = True

        if uniform:
            y = self._simulate_uniform_foh(A, B, C, D, t, u)
            return {"yout": y.tolist()}

        # Non-uniform grid: fall back to SciPy's lsim using SS form
        n = A.shape[0]
        Bm = B.reshape(n, 1)
        Cm = C.reshape(1, n)
        Dm = np.array([[D]], dtype=float)
        tout, yout, _ = signal.lsim((A, Bm, Cm, Dm), U=u, T=t)
        if yout.ndim > 1:
            yout = yout[:, 0]
        return {"yout": yout.tolist()}