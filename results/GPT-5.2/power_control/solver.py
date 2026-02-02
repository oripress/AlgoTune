from __future__ import annotations

from typing import Any

import numpy as np

try:
    import numba as nb

    @nb.njit(cache=True, fastmath=True)
    def _fp_iter_numba(
        G: np.ndarray,
        base: np.ndarray,
        inv: np.ndarray,
        P_min: np.ndarray,
        S_min: float,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        n = P_min.shape[0]
        P = P_min.copy()
        P_new = np.empty_like(P)

        for _ in range(max_iter):
            diff = 0.0
            pmax = 0.0
            for i in range(n):
                # acc = sum_j G_ij * P_j
                Gi = G[i]
                acc = 0.0
                for j in range(n):
                    acc += Gi[j] * P[j]

                # req = inv[i]*acc + base[i] - S_min*P[i]
                req = inv[i] * acc + base[i] - S_min * P[i]
                v = P_min[i] if req < P_min[i] else req
                P_new[i] = v

                d = v - P[i]  # monotone => >= 0
                if d > diff:
                    diff = d
                if v > pmax:
                    pmax = v

            if diff <= tol * (1.0 + pmax):
                return P_new
            P, P_new = P_new, P

        return P

except Exception:  # pragma: no cover
    _fp_iter_numba = None  # type: ignore

class Solver:
    # Keep __init__ trivial; compilation/import time isn't counted.
    def __init__(
        self,
        fp_max_iter: int = 2000,
        fp_tol: float = 1e-11,
        numba_nmax: int = 140,
    ):
        self.fp_max_iter = int(fp_max_iter)
        self.fp_tol = float(fp_tol)
        self.numba_nmax = int(numba_nmax)

        self._cache_n = -1
        self._tmp: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._pnew: np.ndarray | None = None

        # Warm up compilation (not counted) to remove first-call JIT latency.
        if _fp_iter_numba is not None:
            G0 = np.zeros((1, 1), dtype=np.float64)
            base0 = np.zeros(1, dtype=np.float64)
            inv0 = np.ones(1, dtype=np.float64)
            p0 = np.zeros(1, dtype=np.float64)
            _fp_iter_numba(G0, base0, inv0, p0, 1.0, 1, 0.0)

    def _get_bufs(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n != self._cache_n:
            self._cache_n = n
            self._tmp = np.empty(n, dtype=np.float64)
            self._buf = np.empty(n, dtype=np.float64)
            self._pnew = np.empty(n, dtype=np.float64)
        return self._tmp, self._buf, self._pnew  # type: ignore

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        G = np.asarray(problem["G"], dtype=np.float64)
        sigma = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        P_max = np.asarray(problem["P_max"], dtype=np.float64)
        S_min = float(problem["S_min"])

        n = int(P_min.size)
        if n == 0:
            return {"P": np.empty(0, dtype=np.float64)}

        diagG = G.diagonal()
        # Assume instances are valid (diagG > 0) in benchmark; keep minimal guard.
        if diagG.min() <= 0.0:
            raise ValueError("Invalid instance: nonpositive direct path gains on diagonal.")

        inv = S_min / diagG  # (n,)
        base = sigma * inv    # (n,)

        tol = self.fp_tol
        max_iter = self.fp_max_iter

        tmp, buf, P_new = self._get_bufs(n)

        # Fast path for small/medium n: fused numba kernel (often beats BLAS overhead).
        if _fp_iter_numba is not None and n <= self.numba_nmax:
            P = _fp_iter_numba(G, base, inv, P_min, S_min, max_iter, tol)
        else:
            # BLAS path: iterate using G@P then elementwise transforms (allocation-free).
            P = np.array(P_min, copy=True, dtype=np.float64)

            for _ in range(max_iter):
                # tmp := G @ P
                np.dot(G, P, out=tmp)

                # tmp := inv*(G@P) + base - S_min*P
                tmp *= inv
                tmp += base
                np.multiply(P, S_min, out=buf)
                tmp -= buf

                # P_new := max(P_min, tmp)
                np.maximum(P_min, tmp, out=P_new)

                # Since the sequence is nondecreasing, diff = max(P_new - P).
                np.subtract(P_new, P, out=tmp)
                diff = float(tmp.max())

                # Powers are nonnegative, so scale = 1 + max(P_new).
                scale = 1.0 + float(P_new.max())

                P, P_new = P_new, P
                if diff <= tol * scale:
                    break
            else:
                return self._solve_lp_highs(G, sigma, P_min, P_max, S_min)

        # Bounds guard.
        np.subtract(P, P_max, out=buf)
        if float(buf.max()) > 5e-8:
            return self._solve_lp_highs(G, sigma, P_min, P_max, S_min)

        return {"P": P}

    @staticmethod
    def _solve_lp_highs(
        G: np.ndarray,
        sigma: np.ndarray,
        P_min: np.ndarray,
        P_max: np.ndarray,
        S_min: float,
    ) -> dict[str, Any]:
        # Solve LP:
        #   min 1^T P
        #   s.t.  -G_ii P_i + S_min * sum_{k!=i} G_ik P_k <= -S_min*sigma_i
        #         P_min <= P <= P_max
        from scipy.optimize import linprog  # lazy import

        n = int(P_min.size)
        diagG = G.diagonal().astype(np.float64, copy=False)

        A_ub = (S_min * G).astype(np.float64, copy=True)
        A_ub.flat[:: n + 1] = -diagG
        b_ub = (-S_min * sigma).astype(np.float64, copy=False)

        c = np.ones(n, dtype=np.float64)
        bounds = list(zip(P_min.tolist(), P_max.tolist()))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            raise ValueError(f"Solver failed (HiGHS status: {res.status}, msg={res.message})")

        P = res.x.astype(np.float64, copy=False)
        return {"P": P}