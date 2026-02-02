from __future__ import annotations

import math
from typing import Any

import numpy as np

_LN2 = math.log(2.0)
_INV_LN2 = 1.0 / _LN2
_TINY = float(np.finfo(np.float64).tiny)

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        if not isinstance(problem, dict):
            return None
        P_in = problem.get("P")
        if P_in is None:
            return None

        P = np.asarray(P_in, dtype=np.float64)
        if P.ndim != 2:
            return None
        m, n = P.shape
        if m <= 0 or n <= 0:
            return None
        if n == 1:
            return {"x": [1.0], "C": 0.0}

        # Precompute s1_j = sum_i P_ij * ln(P_ij), with convention 0*ln(0)=0.
        # Using ln(P+tiny) avoids mask/where overhead; when P_ij==0, term is exactly 0.
        logP = np.log(P + _TINY)
        s1 = np.sum(P * logP, axis=0)  # shape (n,), nats

        # Blahut-Arimoto with in-place buffers (min allocations)
        x = np.full(n, 1.0 / n, dtype=np.float64)
        q = np.empty(m, dtype=np.float64)
        logq = np.empty(m, dtype=np.float64)
        D = np.empty(n, dtype=np.float64)

        PT = P.T  # view

        tol_bits = 2e-10
        tol_nats = tol_bits * _LN2
        max_iters = 20000

        for _ in range(max_iters):
            np.dot(P, x, out=q)
            np.add(q, _TINY, out=logq)
            np.log(logq, out=logq)

            np.dot(PT, logq, out=D)      # D := PT @ logq
            np.subtract(s1, D, out=D)    # D := s1 - PT@logq

            Dmax = float(D.max())
            lower = float(np.dot(x, D))
            if Dmax - lower <= tol_nats:
                break

            # x <- x * exp(D - Dmax); normalize
            np.subtract(D, Dmax, out=D)
            np.exp(D, out=D)
            x *= D
            x /= float(x.sum())

        # Final capacity C = sum_j x_j D_j / ln2 with D recomputed for final x
        np.dot(P, x, out=q)
        np.add(q, _TINY, out=logq)
        np.log(logq, out=logq)

        np.dot(PT, logq, out=D)
        np.subtract(s1, D, out=D)
        C = float(np.dot(x, D)) * _INV_LN2

        return {"x": x.tolist(), "C": float(C)}