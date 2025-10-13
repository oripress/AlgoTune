from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, List[List[int]] | List[float] | int], **kwargs) -> Dict[str, List[List[float]] | float]:
        """
        Solves the Perron-Frobenius matrix completion using CVXPY in GP form,
        matching the reference formulation to ensure identical outputs.

        minimize     λ_pf(B)
        subject to   prod_{(i,j) ∉ Ω} B_ij = 1
                     B_ij = A_ij, (i,j) ∈ Ω
                     B > 0
        """
        inds = np.array(problem["inds"], dtype=int)
        a = np.array(problem["a"], dtype=float)
        n = int(problem["n"])

        # Efficient construction of unobserved index set via mask
        obs_mask = np.zeros((n, n), dtype=bool)
        if inds.size > 0:
            obs_mask[inds[:, 0], inds[:, 1]] = True
        have_unobserved = (~obs_mask).any()

        # Define CVXPY Variables
        B = cp.Variable((n, n), pos=True)

        # Objective: λ_pf(B)
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints
        constraints = []
        if have_unobserved:
            # Product over all unobserved entries equals 1
            constraints.append(cp.prod(B[~obs_mask]) == 1.0)
        if inds.size > 0:
            constraints.append(B[inds[:, 0], inds[:, 1]] == a)

        # Solve Problem
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True, warm_start=True)
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if B.value is None or not np.all(np.isfinite(B.value)):
            return None
        if result is None or not np.isfinite(result):
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": float(result),
        }