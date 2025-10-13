from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

try:
    # SciPy is available per task description; using it ensures close match to reference
    from scipy.integrate import solve_ivp
except Exception as e:  # pragma: no cover
    raise RuntimeError("scipy is required for this solver.") from e

def _accel_numpy(positions: np.ndarray, masses: np.ndarray, eps: float) -> np.ndarray:
    """
    Vectorized O(N^2) acceleration computation using NumPy broadcasting.
    positions: (N,3)
    masses: (N,)
    returns accelerations: (N,3)
    """
    # R[i, j, :] = r_j - r_i
    r = positions[None, :, :] - positions[:, None, :]
    dist2 = np.sum(r * r, axis=2) + eps * eps  # (N,N)
    inv = 1.0 / (dist2 * np.sqrt(dist2))  # (N,N)
    # Weight by masses of j
    w = inv * masses[None, :]  # (N,N)
    # Sum over j
    a = np.einsum("ij,ijk->ik", w, r, optimize=True)
    return a

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _accel_numba(positions: np.ndarray, masses: np.ndarray, eps: float) -> np.ndarray:
        """
        Pairwise symmetric O(N^2) acceleration with softening, compiled with numba.
        positions: (N,3), masses: (N,)
        returns accelerations: (N,3)
        """
        n = positions.shape[0]
        a = np.zeros((n, 3), dtype=positions.dtype)
        eps2 = eps * eps
        for i in range(n - 1):
            xi = positions[i, 0]
            yi = positions[i, 1]
            zi = positions[i, 2]
            mi = masses[i]
            for j in range(i + 1, n):
                rx = positions[j, 0] - xi
                ry = positions[j, 1] - yi
                rz = positions[j, 2] - zi
                d2 = rx * rx + ry * ry + rz * rz + eps2
                inv_d = 1.0 / np.sqrt(d2)
                inv_d3 = inv_d / d2  # 1 / r^3 with softening
                s_i = masses[j] * inv_d3
                s_j = mi * inv_d3
                a[i, 0] += s_i * rx
                a[i, 1] += s_i * ry
                a[i, 2] += s_i * rz
                a[j, 0] -= s_j * rx
                a[j, 1] -= s_j * ry
                a[j, 2] -= s_j * rz
        return a
else:
    _accel_numba = None  # type: ignore

class Solver:
    def solve(self, problem, **kwargs) -> Any:  # noqa: D401
        """
        Solve N-body gravitational system to final time using SciPy's RK45,
        with an accelerated RHS (Numba when available, otherwise vectorized NumPy),
        matching reference tolerances for accuracy.
        """
        # Extract and prepare inputs
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        masses = np.asarray(problem["masses"], dtype=np.float64)
        softening = float(problem["softening"])
        num_bodies = int(problem["num_bodies"])

        n3 = num_bodies * 3
        if y0.shape[0] != 2 * n3:
            raise ValueError("y0 length must be 6 * num_bodies.")

        # Choose acceleration function
        if _accel_numba is not None:
            accel_fn = _accel_numba
        else:
            accel_fn = _accel_numpy

        # Preallocate output buffer to reduce allocations per RHS call
        out = np.empty_like(y0)

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            # Views without copies
            pos = y[:n3].reshape(num_bodies, 3)
            vel = y[n3:].reshape(num_bodies, 3)

            # dp/dt = v
            out[:n3] = vel.reshape(-1)

            # dv/dt = a(positions)
            a = accel_fn(pos, masses, softening)
            out[n3:] = a.reshape(-1)
            return out

        # Solver parameters: match reference for accuracy compatibility
        rtol = 1e-8
        atol = 1e-8

        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()