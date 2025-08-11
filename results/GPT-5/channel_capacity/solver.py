from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        # Parse and validate input
        if problem is None or "P" not in problem:
            return None
        try:
            P = np.asarray(problem.get("P", None), dtype=float)
        except Exception:
            return None

        if P is None or P.ndim != 2:
            return None

        m, n = P.shape
        if m == 0 or n == 0:
            return None

        # Basic validity checks
        if np.any(~np.isfinite(P)) or np.any(P < -1e-12):
            return None

        # Columns must sum to 1 (probabilities for Y|X=j)
        col_sums = P.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6, rtol=0.0):
            return None

        # Remove output symbols (rows) that are never produced (all zeros).
        row_mask = (P.sum(axis=1) > 0)
        if not np.all(row_mask):
            P = P[row_mask]
            m = P.shape[0]
            if m == 0:
                return None

        # Trivial case: single output symbol => zero capacity, any x on simplex works.
        if m == 1:
            x = np.full(n, 1.0 / n)
            return {"x": x.tolist(), "C": 0.0}

        # Parameters (tolerance in bits)
        tol_bits = float(kwargs.get("tol", 2e-7))
        max_iters = int(kwargs.get("max_iters", 20000))
        x0 = kwargs.get("x0", None)

        return self._solve_blahut_arimoto(P, tol_bits=tol_bits, max_iters=max_iters, x0=x0)

    @staticmethod
    def _solve_blahut_arimoto(
        P: np.ndarray, tol_bits: float = 2e-7, max_iters: int = 20000, x0: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fast Blahut-Arimoto solver using multiplicative updates and the
        duality gap (max D_j - x·D) for termination. Works in nats internally.
        """
        m, n = P.shape
        ln2 = math.log(2.0)

        # Precompute c_j = sum_i P_ij * log(P_ij) (in nats)
        # Use np.where to avoid expensive SciPy import.
        c = np.sum(np.where(P > 0.0, P * np.log(P), 0.0), axis=0)  # shape (n,)

        # Initialize x (strictly positive to avoid zero q entries)
        if x0 is None:
            x = np.full(n, 1.0 / n, dtype=float)
        else:
            x = np.asarray(x0, dtype=float)
            if x.shape != (n,):
                x = np.full(n, 1.0 / n, dtype=float)
            else:
                np.maximum(x, 0.0, out=x)
                s = float(x.sum())
                x = (x / s) if s > 0 else np.full(n, 1.0 / n, dtype=float)
        # Ensure strictly positive entries
        if np.any(x <= 0):
            x = x + 1e-15
            x /= x.sum()

        PT = P.T

        # Main BA loop (multiplicative update)
        for _ in range(max_iters):
            # Output distribution q = P @ x
            q = P @ x  # shape (m,)
            # Clamp tiny values to avoid -inf in logs
            q[q < 1e-300] = 1e-300
            logq = np.log(q)

            # D_j = sum_i P_ij * log(P_ij / q_i) = c_j - sum_i P_ij * log(q_i)
            t = PT @ logq  # shape (n,)
            D = c - t  # shape (n,), in nats

            # Lower bound: I(x) = x·D (nats)
            I_lower_nats = x.dot(D)
            # Valid upper bound on capacity: max_j D_j (Arimoto dual) (nats)
            I_upper_nats = D.max()

            gap_bits = (I_upper_nats - I_lower_nats) / ln2
            if gap_bits <= tol_bits:
                # Return current iterate's mutual information
                C_bits = I_lower_nats / ln2
                # Normalize and ensure non-negativity
                np.maximum(x, 0.0, out=x)
                s2 = x.sum()
                x = (x / s2) if s2 > 0 else np.full(n, 1.0 / n)
                return {"x": x.tolist(), "C": float(C_bits)}

            # Next iterate: x_new ∝ x * exp(D_j - max(D))  (stabilized)
            I_up = float(I_upper_nats)
            D -= I_up  # subtract max(D) in-place for stability: D := D - max(D)
            w = x * np.exp(D)
            s = w.sum()
            x = (w / s) if s > 0 else np.full(n, 1.0 / n, dtype=float)

        # Reached max_iters; return best current iterate's mutual information
        q = P @ x
        q[q < 1e-300] = 1e-300
        logq = np.log(q)
        D = c - (PT @ logq)
        I_lower_nats = x.dot(D)
        C_bits = I_lower_nats / ln2

        np.maximum(x, 0.0, out=x)
        s = x.sum()
        x = (x / s) if s > 0 else np.full(n, 1.0 / n)

        return {"x": x.tolist(), "C": float(C_bits)}