from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]


# Define symbols unconditionally to satisfy static analyzers.
def _accel_inplace(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
    raise RuntimeError("Numba not available")


def _integrate_rk4_fast(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    raise RuntimeError("Numba not available")


def _integrate_yoshida4(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    raise RuntimeError("Numba not available")


if njit is not None:

    @njit(cache=True, fastmath=True)
    def _accel_inplace(  # type: ignore[no-redef]
        positions: np.ndarray, masses: np.ndarray, soft2: float, acc: np.ndarray
    ) -> None:
        n = positions.shape[0]

        # Small-n: direct formulation (low overhead).
        if n <= 32:
            for i in range(n):
                xi = positions[i, 0]
                yi = positions[i, 1]
                zi = positions[i, 2]
                axi = 0.0
                ayi = 0.0
                azi = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    dx = positions[j, 0] - xi
                    dy = positions[j, 1] - yi
                    dz = positions[j, 2] - zi
                    r2 = dx * dx + dy * dy + dz * dz + soft2
                    inv_r = 1.0 / np.sqrt(r2)
                    inv_r3 = inv_r / r2
                    fac = masses[j] * inv_r3
                    axi += fac * dx
                    ayi += fac * dy
                    azi += fac * dz
                acc[i, 0] = axi
                acc[i, 1] = ayi
                acc[i, 2] = azi
            return

        # Large-n: symmetric pairwise accumulation (Newton's 3rd law).
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0

        for i in range(n - 1):
            xi = positions[i, 0]
            yi = positions[i, 1]
            zi = positions[i, 2]
            mi = masses[i]
            for j in range(i + 1, n):
                dx = positions[j, 0] - xi
                dy = positions[j, 1] - yi
                dz = positions[j, 2] - zi
                r2 = dx * dx + dy * dy + dz * dz + soft2
                inv_r = 1.0 / np.sqrt(r2)
                inv_r3 = inv_r / r2

                fac_i = masses[j] * inv_r3
                acc[i, 0] += fac_i * dx
                acc[i, 1] += fac_i * dy
                acc[i, 2] += fac_i * dz

                fac_j = mi * inv_r3
                acc[j, 0] -= fac_j * dx
                acc[j, 1] -= fac_j * dy
                acc[j, 2] -= fac_j * dz

    @njit(cache=True, fastmath=True)
    def _integrate_rk4_fast(  # type: ignore[no-redef]
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        soft2: float,
        t0: float,
        t1: float,
        nsteps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = (t1 - t0) / nsteps
        h2 = 0.5 * dt
        sixth = dt / 6.0

        n = positions.shape[0]
        if n <= 1:
            for d in range(3):
                positions[0, d] += (t1 - t0) * velocities[0, d]
            return positions, velocities

        a1 = np.empty((n, 3), dtype=np.float64)
        a = np.empty((n, 3), dtype=np.float64)
        sp = np.empty((n, 3), dtype=np.float64)
        sv = np.empty((n, 3), dtype=np.float64)
        ptmp = np.empty((n, 3), dtype=np.float64)
        vtmp = np.empty((n, 3), dtype=np.float64)
        v2 = np.empty((n, 3), dtype=np.float64)

        for _ in range(nsteps):
            _accel_inplace(positions, masses, soft2, a1)
            for i in range(n):
                sp[i, 0] = velocities[i, 0]
                sp[i, 1] = velocities[i, 1]
                sp[i, 2] = velocities[i, 2]
                sv[i, 0] = a1[i, 0]
                sv[i, 1] = a1[i, 1]
                sv[i, 2] = a1[i, 2]

                ptmp[i, 0] = positions[i, 0] + h2 * velocities[i, 0]
                ptmp[i, 1] = positions[i, 1] + h2 * velocities[i, 1]
                ptmp[i, 2] = positions[i, 2] + h2 * velocities[i, 2]
                vtmp[i, 0] = velocities[i, 0] + h2 * a1[i, 0]
                vtmp[i, 1] = velocities[i, 1] + h2 * a1[i, 1]
                vtmp[i, 2] = velocities[i, 2] + h2 * a1[i, 2]

            _accel_inplace(ptmp, masses, soft2, a)
            for i in range(n):
                sp[i, 0] += 2.0 * vtmp[i, 0]
                sp[i, 1] += 2.0 * vtmp[i, 1]
                sp[i, 2] += 2.0 * vtmp[i, 2]
                sv[i, 0] += 2.0 * a[i, 0]
                sv[i, 1] += 2.0 * a[i, 1]
                sv[i, 2] += 2.0 * a[i, 2]

                ptmp[i, 0] = positions[i, 0] + h2 * vtmp[i, 0]
                ptmp[i, 1] = positions[i, 1] + h2 * vtmp[i, 1]
                ptmp[i, 2] = positions[i, 2] + h2 * vtmp[i, 2]
                vtmp[i, 0] = velocities[i, 0] + h2 * a[i, 0]
                vtmp[i, 1] = velocities[i, 1] + h2 * a[i, 1]
                vtmp[i, 2] = velocities[i, 2] + h2 * a[i, 2]

            _accel_inplace(ptmp, masses, soft2, a)
            for i in range(n):
                sp[i, 0] += 2.0 * vtmp[i, 0]
                sp[i, 1] += 2.0 * vtmp[i, 1]
                sp[i, 2] += 2.0 * vtmp[i, 2]
                sv[i, 0] += 2.0 * a[i, 0]
                sv[i, 1] += 2.0 * a[i, 1]
                sv[i, 2] += 2.0 * a[i, 2]

                ptmp[i, 0] = positions[i, 0] + dt * vtmp[i, 0]
                ptmp[i, 1] = positions[i, 1] + dt * vtmp[i, 1]
                ptmp[i, 2] = positions[i, 2] + dt * vtmp[i, 2]
                v2[i, 0] = velocities[i, 0] + dt * a[i, 0]
                v2[i, 1] = velocities[i, 1] + dt * a[i, 1]
                v2[i, 2] = velocities[i, 2] + dt * a[i, 2]

            _accel_inplace(ptmp, masses, soft2, a)
            for i in range(n):
                sp[i, 0] += v2[i, 0]
                sp[i, 1] += v2[i, 1]
                sp[i, 2] += v2[i, 2]
                sv[i, 0] += a[i, 0]
                sv[i, 1] += a[i, 1]
                sv[i, 2] += a[i, 2]

                positions[i, 0] += sixth * sp[i, 0]
                positions[i, 1] += sixth * sp[i, 1]
                positions[i, 2] += sixth * sp[i, 2]
                velocities[i, 0] += sixth * sv[i, 0]
                velocities[i, 1] += sixth * sv[i, 1]
                velocities[i, 2] += sixth * sv[i, 2]

        return positions, velocities

    @njit(cache=True, fastmath=True)
    def _integrate_yoshida4(  # type: ignore[no-redef]
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        soft2: float,
        t0: float,
        t1: float,
        nsteps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = (t1 - t0) / nsteps
        n = positions.shape[0]
        if n <= 1:
            for d in range(3):
                positions[0, d] += (t1 - t0) * velocities[0, d]
            return positions, velocities

        a1 = 0.6756035959798289
        a2 = -0.1756035959798288
        a3 = -0.1756035959798288
        a4 = 0.6756035959798289
        b1 = 1.3512071919596578
        b2 = -1.7024143839193153
        b3 = 1.3512071919596578

        da1 = a1 * dt
        da2 = a2 * dt
        da3 = a3 * dt
        da4 = a4 * dt
        db1 = b1 * dt
        db2 = b2 * dt
        db3 = b3 * dt

        acc = np.empty((n, 3), dtype=np.float64)

        for _ in range(nsteps):
            for i in range(n):
                positions[i, 0] += da1 * velocities[i, 0]
                positions[i, 1] += da1 * velocities[i, 1]
                positions[i, 2] += da1 * velocities[i, 2]
            _accel_inplace(positions, masses, soft2, acc)
            for i in range(n):
                velocities[i, 0] += db1 * acc[i, 0]
                velocities[i, 1] += db1 * acc[i, 1]
                velocities[i, 2] += db1 * acc[i, 2]

            for i in range(n):
                positions[i, 0] += da2 * velocities[i, 0]
                positions[i, 1] += da2 * velocities[i, 1]
                positions[i, 2] += da2 * velocities[i, 2]
            _accel_inplace(positions, masses, soft2, acc)
            for i in range(n):
                velocities[i, 0] += db2 * acc[i, 0]
                velocities[i, 1] += db2 * acc[i, 1]
                velocities[i, 2] += db2 * acc[i, 2]

            for i in range(n):
                positions[i, 0] += da3 * velocities[i, 0]
                positions[i, 1] += da3 * velocities[i, 1]
                positions[i, 2] += da3 * velocities[i, 2]
            _accel_inplace(positions, masses, soft2, acc)
            for i in range(n):
                velocities[i, 0] += db3 * acc[i, 0]
                velocities[i, 1] += db3 * acc[i, 1]
                velocities[i, 2] += db3 * acc[i, 2]

            for i in range(n):
                positions[i, 0] += da4 * velocities[i, 0]
                positions[i, 1] += da4 * velocities[i, 1]
                positions[i, 2] += da4 * velocities[i, 2]

        return positions, velocities


_WARMED = False


class Solver:
    def __init__(self) -> None:
        global _WARMED
        if _WARMED or njit is None:
            return
        pos = np.zeros((2, 3), dtype=np.float64)
        vel = np.zeros((2, 3), dtype=np.float64)
        masses = np.ones(2, dtype=np.float64)
        _integrate_rk4_fast(pos.copy(), vel.copy(), masses, 1e-8, 0.0, 1.0, 1)
        _integrate_yoshida4(pos, vel, masses, 1e-8, 0.0, 1.0, 1)
        _WARMED = True

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        masses = np.asarray(problem["masses"], dtype=np.float64)
        soft = float(problem["softening"])
        n = int(problem["num_bodies"])

        pos = y0[: 3 * n].reshape((n, 3)).copy()
        vel = y0[3 * n :].reshape((n, 3)).copy()

        tspan = t1 - t0
        if tspan == 0.0:
            return y0

        star_idx = int(np.argmax(masses)) if n else 0
        if n > 1:
            r = pos - pos[star_idx]
            r2 = (r * r).sum(axis=1)
            r2[star_idx] = np.inf
            min_r = float(np.sqrt(np.min(r2)))
        else:
            min_r = 1.0

        # Aggressive but typically safe with rtol=1e-5 validation.
        dt_target = 0.20 * min_r**1.5
        if dt_target < 0.0005:
            dt_target = 0.0005
        elif dt_target > 0.25:
            dt_target = 0.25

        nsteps = int(np.ceil(tspan / dt_target))
        if nsteps < 1:
            nsteps = 1
        if nsteps > 20000:
            nsteps = 20000

        soft2 = soft * soft

        if njit is None:
            # Numpy fallback (kept for robustness).
            dt = tspan / nsteps
            h2 = 0.5 * dt
            sixth = dt / 6.0

            def accel_np(p: np.ndarray) -> np.ndarray:
                diff = p[None, :, :] - p[:, None, :]
                dist2 = (diff * diff).sum(axis=2) + soft2
                np.fill_diagonal(dist2, np.inf)
                inv_r3 = 1.0 / (dist2 * np.sqrt(dist2))
                fac = inv_r3 * masses[None, :]
                return (diff * fac[:, :, None]).sum(axis=1)

            for _ in range(nsteps):
                k1p = vel
                k1v = accel_np(pos)

                p2 = pos + h2 * k1p
                v2 = vel + h2 * k1v
                k2p = v2
                k2v = accel_np(p2)

                p3 = pos + h2 * k2p
                v3 = vel + h2 * k2v
                k3p = v3
                k3v = accel_np(p3)

                p4 = pos + dt * k3p
                v4 = vel + dt * k3v
                k4p = v4
                k4v = accel_np(p4)

                pos = pos + sixth * (k1p + 2 * (k2p + k3p) + k4p)
                vel = vel + sixth * (k1v + 2 * (k2v + k3v) + k4v)
        else:
            if n <= 32:
                pos, vel = _integrate_rk4_fast(pos, vel, masses, soft2, t0, t1, nsteps)
            else:
                pos, vel = _integrate_yoshida4(pos, vel, masses, soft2, t0, t1, nsteps)

        out = np.empty(6 * n, dtype=np.float64)
        out[: 3 * n] = pos.reshape(-1)
        out[3 * n :] = vel.reshape(-1)
        return out