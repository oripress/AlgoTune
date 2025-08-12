from __future__ import annotations
from typing import Any, Dict
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Parse inputs
        G = np.asarray(problem["G"], float)
        sigma = np.asarray(problem["σ"], float)
        P_min = np.asarray(problem["P_min"], float)
        P_max = np.asarray(problem["P_max"], float)
        S_min = float(problem["S_min"])

        n = G.shape[0]
        if G.shape != (n, n):
            raise ValueError("G must be a square matrix")
        if sigma.shape != (n,) or P_min.shape != (n,) or P_max.shape != (n,):
            raise ValueError("σ, P_min, and P_max must be vectors of length n")

        # Objective: minimize sum(P)
        c = np.ones(n, dtype=float)

        # Bounds: max(0, P_min[i]) <= P[i] <= P_max[i]
        lb = np.maximum(P_min, 0.0)
        ub = P_max
        bounds = list(zip(lb.tolist(), ub.tolist()))
        diag = np.diag(G)
        idx = np.arange(n)

        # Fast check 1: lower bounds already feasible -> optimal
        P0 = lb
        GP0 = G @ P0
        denom0 = sigma + GP0 - diag * P0
        if np.all(diag * P0 >= S_min * denom0 - 1e-12):
            return {"P": P0.tolist(), "objective": float(P0.sum())}

        # Fast check 2: closed-form equality solution when diag > 0 and within bounds
        if np.all(diag > 0):
            # A P = b with A = I - S_min * (G / diag[:, None]), b = S_min * sigma / diag
            A = np.eye(n, dtype=float) - S_min * (G / diag[:, None])
            b = S_min * sigma / diag
            try:
                Peq = np.linalg.solve(A, b)
                # Check bounds (with small tolerance) and feasibility
                if (Peq >= lb - 1e-12).all() and (Peq <= ub + 1e-12).all() and (Peq >= -1e-12).all():
                    GPeq = G @ Peq
                    deneq = sigma + GPeq - diag * Peq
                    if np.all(diag * Peq >= S_min * deneq - 1e-9):
                        # Clip tiny numerical deviations and return
                        P_opt = np.minimum(np.maximum(Peq, lb), ub).astype(float, copy=False)
                        return {"P": P_opt.tolist(), "objective": float(P_opt.sum())}
            except np.linalg.LinAlgError:
                pass  # Fall back to LP

        # LP formulation: S_min * (G @ P) - (1 + S_min) * diag(G) * P <= -S_min * sigma
        A_ub = S_min * G.copy()
        A_ub[idx, idx] -= (1.0 + S_min) * diag
        b_ub = -S_min * sigma

        # Solve LP using HiGHS
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            raise ValueError(f"Solver failed (status={res.status}, message={res.message})")

        P_opt = res.x.astype(float, copy=False)
        return {"P": P_opt.tolist(), "objective": float(P_opt.sum())}