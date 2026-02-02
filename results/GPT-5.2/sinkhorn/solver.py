from __future__ import annotations

from typing import Any

import numpy as np

class _Buffers:
    __slots__ = ("K", "u", "v", "Kv", "Ktu", "r")

    def __init__(self, n: int, m: int) -> None:
        self.K = np.empty((n, m), dtype=np.float64)  # reused for kernel AND final G
        self.u = np.ones(n, dtype=np.float64)
        self.v = np.ones(m, dtype=np.float64)
        self.Kv = np.empty(n, dtype=np.float64)
        self.Ktu = np.empty(m, dtype=np.float64)
        self.r = np.empty(n, dtype=np.float64)

def _sinkhorn_inplace(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float,
    buf: _Buffers,
    *,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-9,
    check_every: int = 10,
) -> np.ndarray:
    """
    Sinkhorn-Knopp where the kernel buffer `buf.K` is overwritten and returned as G.
    This avoids allocating (n,m) arrays per call.
    """
    n = a.shape[0]
    m = b.shape[0]

    if n == 1:
        return b.reshape(1, m).copy()
    if m == 1:
        return a.reshape(n, 1).copy()

    K = buf.K
    u = buf.u
    v = buf.v
    Kv = buf.Kv
    Ktu = buf.Ktu
    r = buf.r

    # Kernel matrix: K = exp(-M/reg) computed in-place with no temporary (n,m) allocations.
    inv_reg = -1.0 / reg
    np.multiply(M, inv_reg, out=K)
    np.exp(K, out=K)

    # reset scalings
    u.fill(1.0)
    # reset scalings
    u.fill(1.0)
    v.fill(1.0)

    eps = 1e-300
    Kt = K.T  # view, reused

    # Precompute Kv for current v (starts as all-ones)
    np.dot(K, v, out=Kv)

    for it in range(num_iter_max):
        # u = a / (K @ v)   (Kv already holds K@v)
        np.maximum(Kv, eps, out=Kv)
        np.divide(a, Kv, out=u)

        # v = b / (K.T @ u)
        np.dot(Kt, u, out=Ktu)
        np.maximum(Ktu, eps, out=Ktu)
        np.divide(b, Ktu, out=v)

        # Kv <- K @ v for next iteration AND for error checking (row marginal)
        np.dot(K, v, out=Kv)

        if (it % check_every) == 0:
            np.multiply(u, Kv, out=r)  # r = u * (K@v)
            np.subtract(r, a, out=r)
            np.abs(r, out=r)
            if r.sum() <= stop_thr:
                break

    # Overwrite K -> transport plan: G = diag(u) K diag(v)
    K *= v  # broadcast over rows
    K *= u[:, None]
    return K

class Solver:
    def __init__(self) -> None:
        self._ot = None
        self._bufs: dict[tuple[int, int], _Buffers] = {}

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)
        if not M.flags["C_CONTIGUOUS"]:
            M = np.ascontiguousarray(M)
        reg = float(problem["reg"])

        n = a.shape[0]
        m = b.shape[0]
        buf = self._bufs.get((n, m))
        if buf is None:
            buf = _Buffers(n, m)
            self._bufs[(n, m)] = buf

        try:
            G = _sinkhorn_inplace(a, b, M, reg, buf)
            # Note: we intentionally return the reusable buffer for speed.
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            # fallback to POT reference for robustness
            try:
                if self._ot is None:
                    import ot  # type: ignore

                    self._ot = ot
                G = self._ot.sinkhorn(a, b, M, reg)
                return {"transport_plan": G, "error_message": None}
            except Exception:
                return {"transport_plan": None, "error_message": str(exc)}
        except Exception as exc:
            # fallback to POT reference for robustness
            try:
                if self._ot is None:
                    import ot  # type: ignore

                    self._ot = ot
                G = self._ot.sinkhorn(a, b, M, reg)
                if not np.isfinite(G).all():
                    raise ValueError("Non-finite transport plan")
                return {"transport_plan": G, "error_message": None}
            except Exception:
                return {"transport_plan": None, "error_message": str(exc)}