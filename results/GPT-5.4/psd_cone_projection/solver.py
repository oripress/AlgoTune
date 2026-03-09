from math import hypot
from typing import Any

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _jacobi_psd_project(A):
    n = A.shape[0]
    D = A.copy()
    V = np.eye(n, dtype=np.float64)

    for _ in range(12):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = D[p, q]
                if abs(apq) <= 1e-11 * (abs(D[p, p]) + abs(D[q, q]) + 1.0):
                    continue

                changed = True
                app = D[p, p]
                aqq = D[q, q]
                tau = (aqq - app) / (2.0 * apq)

                if tau >= 0.0:
                    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))

                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c

                D[p, p] = app - t * apq
                D[q, q] = aqq + t * apq
                D[p, q] = 0.0
                D[q, p] = 0.0

                for r in range(n):
                    if r != p and r != q:
                        drp = D[r, p]
                        drq = D[r, q]
                        nrp = c * drp - s * drq
                        nrq = s * drp + c * drq
                        D[r, p] = nrp
                        D[p, r] = nrp
                        D[r, q] = nrq
                        D[q, r] = nrq

                for r in range(n):
                    vrp = V[r, p]
                    vrq = V[r, q]
                    V[r, p] = c * vrp - s * vrq
                    V[r, q] = s * vrp + c * vrq

        if not changed:
            break

    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w = D[i, i]
        if w > 0.0:
            for r in range(n):
                xir = w * V[r, i]
                for c in range(r, n):
                    X[r, c] += xir * V[c, i]

    for r in range(n):
        for c in range(r):
            X[r, c] = X[c, r]

    return X

class Solver:
    def __init__(self) -> None:
        self._asarray = np.asarray
        self._eigh = np.linalg.eigh
        self._zeros = np.zeros
        self._jacobi = _jacobi_psd_project
        self._jacobi(np.zeros((3, 3), dtype=np.float64))

    def solve(self, problem, **kwargs) -> Any:
        raw = problem["A"]
        n = raw.shape[0] if isinstance(raw, np.ndarray) else len(raw)

        if n == 0:
            return {"X": raw}

        if n == 1:
            x = raw[0, 0] if isinstance(raw, np.ndarray) else raw[0][0]
            if x > 0.0:
                return {"X": raw}
            return {"X": [[0.0]]}

        if n == 2:
            if isinstance(raw, np.ndarray):
                a = float(raw[0, 0])
                b = float(raw[0, 1])
                d = float(raw[1, 1])
            else:
                r0 = raw[0]
                r1 = raw[1]
                a = float(r0[0])
                b = float(r0[1])
                d = float(r1[1])

            t = 0.5 * (a + d)
            s = hypot(0.5 * (a - d), b)
            lmax = t + s

            if lmax <= 0.0:
                return {"X": [[0.0, 0.0], [0.0, 0.0]]}

            lmin = t - s
            if lmin >= 0.0:
                return {"X": raw}

            if s == 0.0:
                v = lmax if lmax > 0.0 else 0.0
                return {"X": [[v, 0.0], [0.0, v]]}

            scale = lmax / (2.0 * s)
            x00 = scale * (a - lmin)
            x01 = scale * b
            x11 = scale * (d - lmin)
            return {"X": [[x00, x01], [x01, x11]]}

        if n <= 7:
            A = self._asarray(raw, dtype=np.float64)
            return {"X": self._jacobi(A)}

        A = self._asarray(raw, dtype=np.float64)
        w, v = self._eigh(A)

        if w[0] > 0.0:
            return {"X": raw}

        if w[-1] <= 0.0:
            return {"X": self._zeros(A.shape, dtype=np.float64)}

        k = np.searchsorted(w, 0.0, side="right")
        vp = v[:, k:]
        X = (vp * w[k:]) @ vp.T
        return {"X": X}