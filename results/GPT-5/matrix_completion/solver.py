from __future__ import annotations

from typing import Any

import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the Perron-Frobenius matrix completion using CVXPY with the pf_eigenvalue objective,
        while computing index sets efficiently.

        Args:
            problem: Dict containing "inds" (list of [i, j]), "a" (list of observed values), and "n" (int).

        Returns:
            Dict with keys:
              - "B": list of lists representing the matrix B
              - "optimal_value": float optimal Perron-Frobenius eigenvalue
        """
        # Parse inputs
        inds = np.array(problem["inds"], dtype=int)
        a = np.array(problem["a"], dtype=float)
        n = int(problem["n"])

        # Build observed mask and extract indices
        obs_mask = np.zeros((n, n), dtype=bool)
        if inds.size > 0:
            obs_mask[inds[:, 0], inds[:, 1]] = True
            obs_rows = inds[:, 0]
            obs_cols = inds[:, 1]
        else:
            obs_rows = np.array([], dtype=int)
            obs_cols = np.array([], dtype=int)

        # Unknown entries are the complement of observed
        unk_mask = ~obs_mask
        unk_rows, unk_cols = np.nonzero(unk_mask)
        num_unknown = unk_rows.size

        # Fast paths: no unknowns or exactly one unknown -> B is fixed
        if num_unknown == 0:
            B_fixed = np.empty((n, n), dtype=float)
            B_fixed[obs_rows, obs_cols] = a
            vals = np.linalg.eigvals(B_fixed)
            optimal_value = float(np.max(np.abs(vals)).real)
            return {"B": B_fixed.tolist(), "optimal_value": optimal_value}
        elif num_unknown == 1:
            B_fixed = np.empty((n, n), dtype=float)
            if obs_rows.size > 0:
                B_fixed[obs_rows, obs_cols] = a
            B_fixed[unk_rows[0], unk_cols[0]] = 1.0
            vals = np.linalg.eigvals(B_fixed)
            optimal_value = float(np.max(np.abs(vals)).real)
            return {"B": B_fixed.tolist(), "optimal_value": optimal_value}

        # Define variables (positive for GP)
        B = cp.Variable((n, n), pos=True)

        # Objective: minimize Perron-Frobenius eigenvalue
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints: product over unknown entries equals 1, and match observed entries
        constraints = []
        if num_unknown > 0:
            constraints.append(cp.prod(B[unk_rows, unk_cols]) == 1.0)
        if obs_rows.size > 0:
            constraints.append(B[obs_rows, obs_cols] == a)

        # Solve problem in GP mode to match reference behavior
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except Exception:
            return {
                "B": (np.full((n, n), np.nan)).tolist(),
                "optimal_value": float("nan"),
            }

        # If solver failed to produce a value, return NaNs
        if B.value is None or result is None or not np.isfinite(result):
            return {
                "B": (np.full((n, n), np.nan)).tolist(),
                "optimal_value": float("nan"),
            }

        return {
            "B": B.value.tolist(),
            "optimal_value": float(result),
        }