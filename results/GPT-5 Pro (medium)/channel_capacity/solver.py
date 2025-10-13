from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from scipy.special import xlogy

class Solver:
    def _mutual_information_bits(self, P: np.ndarray, x: np.ndarray, ln2: float) -> float:
        # I(X;Y) = H(Y) - H(Y|X); with natural logs then divide by ln2 to convert to bits
        y = P @ x
        # H(Y) = -sum y log y
        H_Y = -np.sum(xlogy(y, y))
        # H(Y|X) = - sum_j x_j sum_i P_ij log P_ij
        H_Y_given_X = -np.dot(x, np.sum(xlogy(P, P), axis=0))
        return (H_Y - H_Y_given_X) / ln2

    def _blahut_arimoto(
        self,
        P: np.ndarray,
        *,
        tol_bits: float = 5e-8,
        max_iter: int = 20000,
    ) -> tuple[np.ndarray, float, float, int]:
        """
        Run Blahut-Arimoto to compute capacity and optimal input distribution.

        Returns:
            x: optimal input distribution over columns (n,)
            C_bits: achieved mutual information (bits)
            gap_bits: final duality gap (bits)
            iters: iterations performed
        """
        m, n = P.shape
        ln2 = math.log(2.0)

        # Remove rows that are identically zero to avoid log(0) issues; they contribute nothing
        row_mask = np.sum(P, axis=1) > 0.0
        if not np.all(row_mask):
            P_eff = P[row_mask, :]
        else:
            P_eff = P

        # Precompute c in nats: c_j = sum_i P_ij log P_ij
        # xlogy handles 0 * log(0) = 0
        c_nats = np.sum(xlogy(P_eff, P_eff), axis=0)

        # Initialize x strictly positive; uniform over inputs
        n_eff = P_eff.shape[1]
        x = np.full(n_eff, 1.0 / n_eff, dtype=float)

        tol_nats = tol_bits * ln2

        # Iterations
        LB = -np.inf
        UB = np.inf
        gap = np.inf

        # Pre-allocate vectors
        for it in range(1, max_iter + 1):
            y = P_eff @ x  # (m_eff,)
            # y should be strictly positive since each kept row has some positive mass and x > 0
            # Still, guard against underflow
            # Note: no need to clip if y>0; but tiny values are okay for log
            log_y = np.log(y)

            # t_j = sum_i P_ij log P_ij - sum_i P_ij log y_i = c_j - (P^T log y)_j
            t = c_nats - (P_eff.T @ log_y)  # (n,)

            # Lower bound: I(x) = x^T t (nats)
            LB_new = float(np.dot(x, t))

            # Upper bound: log sum_j x_j * exp(t_j) (nats)
            t_max = float(np.max(t))
            s = np.exp(t - t_max)  # rescaled to avoid overflow
            Z = float(np.dot(x, s))
            # In theory Z > 0
            UB_new = t_max + math.log(Z)

            gap_new = UB_new - LB_new

            # Check convergence
            if gap_new <= tol_nats:
                LB = LB_new
                gap = gap_new
                break

            # Update x: x_new_j = x_j * exp(t_j) / sum_k x_k * exp(t_k)
            x = (x * s) / Z

            LB = LB_new
            UB = UB_new
            gap = gap_new

        C_bits = LB / ln2
        gap_bits = gap / ln2
        return x, C_bits, gap_bits, it

    def solve(self, problem: dict, **kwargs) -> Optional[dict]:
        # Extract and validate P
        P_raw = problem.get("P", None)
        if P_raw is None:
            return None
        try:
            P = np.asarray(P_raw, dtype=float)
        except Exception:
            return None

        if P.ndim != 2:
            return None
        m, n = P.shape
        if m == 0 or n == 0:
            return None

        # BA parameters
        tol_bits = kwargs.get("tol_bits", 5e-8)
        max_iter = kwargs.get("max_iter", 20000)

        # Run Blahut-Arimoto
        x_ba, C_bits_ba, gap_bits, _ = self._blahut_arimoto(P, tol_bits=tol_bits, max_iter=max_iter)

        # Compute capacity using the original P (not row-filtered) to match validator
        ln2 = math.log(2.0)
        C_bits = self._mutual_information_bits(P, x_ba, ln2)

        # If BA result is sufficiently accurate, return
        # Use a safe margin vs. 1e-6 tolerance of the validator
        if gap_bits <= max(tol_bits, 2e-7):
            return {"x": x_ba.tolist(), "C": C_bits}

        # Fallback to CVX (if needed) for rare cases of slow BA convergence
        try:
            import cvxpy as cp

            x = cp.Variable(n)
            y_expr = P @ x
            c_bits = np.sum(xlogy(P, P), axis=0) / ln2  # c in bits
            objective = cp.Maximize(cp.sum(cp.entr(y_expr) / ln2) + c_bits @ x)
            constraints = [cp.sum(x) == 1, x >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=5000)

            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and prob.value is not None:
                x_val = np.asarray(x.value, dtype=float).flatten()
                # Clip tiny negatives due to solver tolerance and renormalize
                x_val = np.maximum(x_val, 0.0)
                s = x_val.sum()
                if s <= 0:
                    # Degenerate; fall back to BA result
                    return {"x": x_ba.tolist(), "C": C_bits}
                x_val /= s
                C_bits_val = float(prob.value)
                # Use mutual information computed from x/P to ensure consistency with validator
                C_bits_val = self._mutual_information_bits(P, x_val, ln2)
                return {"x": x_val.tolist(), "C": C_bits_val}
        except Exception:
            pass

        # Fall back to BA result if CVX path unavailable
        return {"x": x_ba.tolist(), "C": C_bits}