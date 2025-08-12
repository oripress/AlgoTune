from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

def _compute_accelerations(positions: np.ndarray, masses: np.ndarray, softening: float) -> np.ndarray:
    """
    Vectorized accelerations for N bodies.
    positions: (N,3)
    masses: (N,)
    returns: (N,3)
    """
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    N = positions.shape[0]

    if N == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if N == 1:
        return np.zeros((1, 3), dtype=np.float64)

    # diff[i,j,k] = r_jk - r_ik
    diff = positions[None, :, :] - positions[:, None, :]  # shape (N, N, 3)

    soft2 = float(softening) * float(softening)

    # squared distances with softening
    dist2 = np.einsum("ijk,ijk->ij", diff, diff) + soft2  # shape (N, N)

    # inverse distance cubed: 1 / (dist2 * sqrt(dist2))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))

    # zero self-contributions and any non-finite entries
    if inv_dist3.shape[0] == inv_dist3.shape[1]:
        np.fill_diagonal(inv_dist3, 0.0)
    inv_dist3[~np.isfinite(inv_dist3)] = 0.0

    # factor[i,j] = m_j * inv_dist3[i,j]
    factor = inv_dist3 * masses[None, :]

    # acceleration on i: sum_j factor[i,j] * (r_j - r_i)
    acc = np.einsum("ij,ijk->ik", factor, diff)

    return acc

class Solver:
    """
    N-body gravitational solver.

    solve(self, problem: dict, **kwargs) -> list[float]
    Expects keys: 't0', 't1', 'y0', 'masses', 'softening', 'num_bodies'
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        required = ("t0", "t1", "y0", "masses", "softening", "num_bodies")
        if not all(k in problem for k in required):
            raise ValueError(f"Problem dictionary must contain keys: {required}")

        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=np.float64).ravel()
        masses = np.asarray(problem["masses"], dtype=np.float64).ravel()
        softening = float(problem["softening"])
        num_bodies = int(problem["num_bodies"])

        if y0.size != 6 * num_bodies:
            raise ValueError("y0 length does not match 6 * num_bodies")
        if masses.size != num_bodies:
            raise ValueError("masses length does not match num_bodies")

        N = num_bodies

        # Trivial cases
        if N == 0 or t1 == t0:
            return [float(x) for x in y0]

        def _rhs(t, y):
            pos = y[: 3 * N].reshape((N, 3))
            vel = y[3 * N :].reshape((N, 3))

            acc = _compute_accelerations(pos, masses, softening)

            dp_dt = vel.ravel()
            dv_dt = acc.ravel()
            return np.concatenate([dp_dt, dv_dt])

        # Solver parameters - match reference tolerances by default
        rtol = kwargs.get("rtol", 1e-8)
        atol = kwargs.get("atol", 1e-8)
        method = kwargs.get("method", "RK45")

        sol = solve_ivp(_rhs, (t0, t1), y0, method=method, rtol=rtol, atol=atol)

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return [float(x) for x in sol.y[:, -1]]