from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the optimal power control problem using a fixed‑point iteration.
        The SINR constraints can be written as:
            P_i >= (S_min / G_ii) * (σ_i + Σ_{k≠i} G_ik P_k)

        Starting from the minimum power vector, we repeatedly apply the
        update until convergence. The resulting vector is the minimal feasible
        one (and thus optimal for minimizing the sum of powers). Bounds
        are enforced by early termination if any component exceeds its
        maximum allowed power.
        """
        # Extract data
        G = np.asarray(problem["G"], dtype=float)          # n x n gain matrix
        sigma = np.asarray(problem["σ"], dtype=float)    # noise vector
        P_min = np.asarray(problem["P_min"], dtype=float)  # lower bounds
        P_max = np.asarray(problem["P_max"], dtype=float)  # upper bounds
        S_min = float(problem["S_min"])

        n = G.shape[0]
        if G.shape != (n, n):
            raise ValueError("G must be a square matrix")

        diag = np.diag(G)
        if np.any(diag <= 0):
            raise ValueError("Diagonal entries of G must be positive")

        # Initialise with the lower‑bound vector.
        P = np.copy(P_min)

        # Fixed‑point iteration.
        max_iter = 1000
        tol = 1e-9
        for _ in range(max_iter):
            # interference = σ_i + Σ_{k} G_ik P_k - G_ii P_i
            interference = sigma + G @ P - diag * P
            # Required power from SINR constraints.
            P_new = np.maximum(P_min, (S_min * interference) / diag)

            # Convergence check.
            if np.all(np.abs(P_new - P) <= tol * np.maximum(1.0, np.abs(P_new))):
                P = P_new
                break
            P = P_new

            # Early infeasibility detection.
            if np.any(P > P_max + 1e-8):
                raise ValueError("Problem infeasible: power exceeds P_max during iteration")

        else:
            raise RuntimeError("Fixed‑point iteration failed to converge")

        # Final feasibility check.
        if np.any(P > P_max + 1e-8):
            raise ValueError("Problem infeasible: final power exceeds P_max")

        # Return solution.
        return {"P": P.tolist(), "objective": float(P.sum())}