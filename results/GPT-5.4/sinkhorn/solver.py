from __future__ import annotations

from typing import Any

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None  # type: ignore[assignment]

try:
    import ot
    from ot.bregman import sinkhorn_knopp
except Exception:  # pragma: no cover
    ot = None  # type: ignore[assignment]
    sinkhorn_knopp = None  # type: ignore[assignment]

_NUM_ITER_MAX = 1000
_STOP_THR = 1e-9
_CHECK_EVERY = 10
_NUMBA_THRESHOLD = 4096

if nb is not None:

    @nb.njit(cache=True)
    def _sinkhorn_numba(
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: float,
        num_iter_max: int,
        stop_thr: float,
        check_every: int,
    ) -> tuple[np.ndarray, int]:
        n, m = M.shape
        min_m = M[0, 0]
        for i in range(n):
            for j in range(m):
                if M[i, j] < min_m:
                    min_m = M[i, j]

        K = np.empty((n, m), dtype=np.float64)
        inv_reg = -1.0 / reg
        for i in range(n):
            for j in range(m):
                K[i, j] = np.exp((M[i, j] - min_m) * inv_reg)

        u = np.full(n, 1.0 / n, dtype=np.float64)
        v = np.full(m, 1.0 / m, dtype=np.float64)

        for it in range(num_iter_max):
            for j in range(m):
                s = 0.0
                for i in range(n):
                    s += K[i, j] * u[i]
                if s <= 0.0 or not np.isfinite(s):
                    return K, 0
                v[j] = b[j] / s
                if not np.isfinite(v[j]):
                    return K, 0

            for i in range(n):
                s = 0.0
                for j in range(m):
                    s += K[i, j] * v[j]
                if s <= 0.0 or not np.isfinite(s):
                    return K, 0
                u[i] = a[i] / s
                if not np.isfinite(u[i]):
                    return K, 0

            if it % check_every == 0:
                err = 0.0
                for j in range(m):
                    s = 0.0
                    for i in range(n):
                        s += K[i, j] * u[i]
                    d = v[j] * s - b[j]
                    err += d * d
                if err <= stop_thr * stop_thr:
                    break

        for i in range(n):
            ui = u[i]
            for j in range(m):
                K[i, j] *= ui * v[j]
        return K, 1

else:

    def _sinkhorn_numba(
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: float,
        num_iter_max: int,
        stop_thr: float,
        check_every: int,
    ) -> tuple[np.ndarray, int]:
        raise RuntimeError("numba unavailable")

def _solve_2x2(a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float) -> np.ndarray | None:
    a0 = float(a[0])
    b0 = float(b[0])
    lower = max(0.0, a0 - float(b[1]), b0 - float(a[1]))
    upper = min(a0, b0)

    delta = (M[0, 1] + M[1, 0] - M[0, 0] - M[1, 1]) / reg
    if not np.isfinite(delta):
        return None

    if delta >= 50.0:
        x = upper
    elif delta <= -50.0:
        x = lower
    else:
        r = float(np.exp(delta))
        A = 1.0 - r
        B = 1.0 - a0 - b0 + r * (a0 + b0)
        C = -r * a0 * b0
        if abs(A) < 1e-14:
            if abs(B) < 1e-14:
                return None
            x = -C / B
        else:
            disc = B * B - 4.0 * A * C
            if disc < 0.0:
                if disc > -1e-14:
                    disc = 0.0
                else:
                    return None
            s = float(np.sqrt(disc))
            x1 = (-B + s) / (2.0 * A)
            x2 = (-B - s) / (2.0 * A)

            ok1 = lower - 1e-12 <= x1 <= upper + 1e-12
            ok2 = lower - 1e-12 <= x2 <= upper + 1e-12
            if ok1 and ok2:
                def resid(z: float) -> float:
                    return abs(z * (1.0 - a0 - b0 + z) - r * (a0 - z) * (b0 - z))
                x = x1 if resid(x1) <= resid(x2) else x2
            elif ok1:
                x = x1
            elif ok2:
                x = x2
            else:
                return None

    if x < lower:
        x = lower
    elif x > upper:
        x = upper

    G = np.array(
        [
            [x, a0 - x],
            [b0 - x, 1.0 - a0 - b0 + x],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(G).all():
        return None
    G[G < 0.0] = 0.0
    return G

class Solver:
    def __init__(self) -> None:
        self._compute = (
            sinkhorn_knopp
            if sinkhorn_knopp is not None
            else getattr(ot, "sinkhorn", None)
        )
        self._numba_ok = nb is not None
        if self._numba_ok:
            try:
                a = np.array([0.5, 0.5], dtype=np.float64)
                b = np.array([0.5, 0.5], dtype=np.float64)
                M = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
                _sinkhorn_numba(a, b, M, 1.0, 2, _STOP_THR, 1)
            except Exception:
                self._numba_ok = False

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        try:
            a = np.asarray(problem["source_weights"], dtype=np.float64)
            b = np.asarray(problem["target_weights"], dtype=np.float64)
            M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
            reg = float(problem["reg"])

            if a.size == 1 or b.size == 1:
                G = np.outer(a, b)
            elif a.size == 2 and b.size == 2:
                G = _solve_2x2(a, b, M, reg)
                if G is None:
                    if self._numba_ok and M.size <= _NUMBA_THRESHOLD:
                        G, ok = _sinkhorn_numba(
                            a,
                            b,
                            M,
                            reg,
                            _NUM_ITER_MAX,
                            _STOP_THR,
                            _CHECK_EVERY,
                        )
                        if not ok:
                            compute = self._compute
                            if compute is None:  # pragma: no cover
                                raise ImportError("ot package unavailable")
                            G = compute(a, b, M, reg)
                    else:
                        compute = self._compute
                        if compute is None:  # pragma: no cover
                            raise ImportError("ot package unavailable")
                        G = compute(a, b, M, reg)
            elif self._numba_ok and M.size <= _NUMBA_THRESHOLD:
                G, ok = _sinkhorn_numba(
                    a,
                    b,
                    M,
                    reg,
                    _NUM_ITER_MAX,
                    _STOP_THR,
                    _CHECK_EVERY,
                )
                if not ok:
                    compute = self._compute
                    if compute is None:  # pragma: no cover
                        raise ImportError("ot package unavailable")
                    G = compute(a, b, M, reg)
            else:
                compute = self._compute
                if compute is None:  # pragma: no cover
                    raise ImportError("ot package unavailable")
                G = compute(a, b, M, reg)

            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}