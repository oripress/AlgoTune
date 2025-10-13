from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm
from scipy.signal import tf2ss

class Solver:
    def solve(self, problem: Dict[str, ArrayLike], **kwargs) -> Dict[str, List[float]]:
        """
        Fast LTI continuous-time simulation using state-space FOH (first-order hold) discretization
        over the provided time grid and linearly interpolated input u.

        The result matches scipy.signal.lsim within tight numerical tolerance.
        """
        # Extract and convert inputs
        num = np.asarray(problem["num"], dtype=float).ravel()
        den = np.asarray(problem["den"], dtype=float).ravel()
        t = np.asarray(problem["t"], dtype=float).ravel()
        u = np.asarray(problem["u"], dtype=float).ravel()

        # Basic checks
        if t.ndim != 1 or u.ndim != 1:
            raise ValueError("t and u must be 1-D arrays.")
        if len(t) != len(u):
            raise ValueError("Length of t and u must match.")
        n_time = len(t)
        if n_time == 0:
            return {"yout": []}
        if n_time == 1:
            # Single time point: output is instantaneous Cx+Du with x0=0
            # Handle static gain case too.
            if len(den) == 1:  # static mapping
                gain = np.polyval(num, 0.0) / den[0]
                y0 = gain * u[0]
                return {"yout": [float(y0)]}
            # Otherwise dynamic system, y = D * u
            A, B, C, D = tf2ss(num, den)
            y0 = float((C @ np.zeros(A.shape[0])) + D * u[0])
            return {"yout": [y0]}

        # Check monotonicity of t
        if not np.all(np.diff(t) > 0):
            raise ValueError("Time vector t must be strictly increasing.")

        # Handle static gain system quickly: denominator of degree 0
        if len(den) == 1:
            gain = np.polyval(num, 0.0) / den[0]
            y = gain * u
            return {"yout": y.astype(float).tolist()}

        # Convert TF to state-space (SISO)
        A, B, C, D = tf2ss(num, den)
        # Ensure 1D/2D shapes
        n = A.shape[0]
        # Make B shape (n, 1), C shape (1, n)
        B = np.atleast_2d(B)
        if B.shape[1] != 1:
            # Ensure SISO (as per problem statement)
            if B.shape[0] == n and B.shape[1] == 1:
                pass
            else:
                # Reduce/reshape to single input
                B = B.reshape(n, 1)
        C = np.atleast_2d(C)
        C = C.reshape(1, n)
        D = float(np.asarray(D).reshape(()))

        # Precompute y array
        y = np.empty(n_time, dtype=float)

        # Initial state
        x = np.zeros((n,), dtype=float)

        # Output at initial time
        y[0] = float(C @ x + D * u[0])

        # Time steps
        dt = np.diff(t)

        # Check if dt is (almost) constant to enable reuse of matrices
        # Use relative and absolute tolerance
        dt0 = dt[0]
        if np.allclose(dt, dt0, rtol=1e-12, atol=1e-15):
            # Build single block matrix for FOH integrals
            M = np.zeros((3 * n, 3 * n), dtype=float)
            # M top-left A
            M[:n, :n] = A
            # M top-middle I
            M[:n, n : 2 * n] = np.eye(n, dtype=float)
            # M middle-right I
            M[n : 2 * n, 2 * n : 3 * n] = np.eye(n, dtype=float)

            E = expm(M * dt0)
            Phi = E[:n, :n]  # e^{A dt}
            S1 = E[:n, n : 2 * n]  # ∫_0^{dt} e^{A s} ds
            E13 = E[:n, 2 * n : 3 * n]  # ∫_0^{dt} (dt - s) e^{A s} ds = dt S1 - S2

            # J0 = S1 @ B
            J0B = S1 @ B  # (n,1)
            # J1 = (1/dt) * E13 @ B
            J1B = (E13 @ B) / dt0  # (n,1)

            # Loop over intervals with reused matrices
            for k in range(n_time - 1):
                uk = u[k]
                uk1 = u[k + 1]
                # Update state: x_{k+1} = Phi x_k + (J0 - J1)*uk + J1*uk1
                x = Phi @ x + (J0B[:, 0] - J1B[:, 0]) * uk + J1B[:, 0] * uk1
                # Output at next time point
                y[k + 1] = float(C @ x + D * uk1)
        else:
            # Non-uniform time steps: recompute per-step matrices
            # Pre-allocate block matrix to reuse memory
            M = np.zeros((3 * n, 3 * n), dtype=float)
            M[:n, :n] = A
            M[:n, n : 2 * n] = np.eye(n, dtype=float)
            M[n : 2 * n, 2 * n : 3 * n] = np.eye(n, dtype=float)

            for k in range(n_time - 1):
                dtk = dt[k]
                Ek = expm(M * dtk)
                Phi = Ek[:n, :n]
                S1 = Ek[:n, n : 2 * n]
                E13 = Ek[:n, 2 * n : 3 * n]
                J0B = S1 @ B
                J1B = (E13 @ B) / dtk

                uk = u[k]
                uk1 = u[k + 1]
                x = Phi @ x + (J0B[:, 0] - J1B[:, 0]) * uk + J1B[:, 0] * uk1
                y[k + 1] = float(C @ x + D * uk1)

        return {"yout": y.tolist()}