import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the power control problem using a fast iterative algorithm.

        This problem is a standard form of power control in wireless networks,
        which can be formulated as a Linear Program (LP). The optimal solution
        (minimizing total power) is the component-wise smallest power vector P
        that satisfies the Signal-to-Interference-plus-Noise Ratio (SINR)
        and power limit constraints.

        This minimal vector can be found efficiently using a fixed-point iteration
        based on the standard interference function. This approach is typically
        much faster than using a general-purpose LP solver, as it consists of
        simple, vectorizable NumPy operations.

        The iteration is:
        P(t+1) = clip(f(P(t)), P_min, P_max)
        where f_i(P) = S_min/G_ii * (sigma_i + sum_{k!=i} G_ik * P_k)

        The algorithm starts with the lowest possible power P(0) = P_min and
        iteratively increases powers until all SINR constraints are met.
        Convergence is guaranteed because the update function is monotonic and
        the power values are bounded.
        """
        G = np.asarray(problem["G"], dtype=np.float64)
        sigma = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        P_max = np.asarray(problem["P_max"], dtype=np.float64)
        S_min = float(problem["S_min"])
        n = G.shape[0]

        # Pre-calculate constants for the iterative update.
        g_diag = np.diag(G)
        
        # If a diagonal element of G is zero or negative, the SINR for that user
        # is ill-defined or zero. If S_min > 0, the problem is infeasible.
        if np.any(g_diag <= 1e-12): # Use tolerance for float comparison
            if S_min > 0:
                 raise ValueError("Problem is infeasible due to non-positive diagonal elements in G.")
            # If S_min is 0, the SINR constraint is trivial. We can set the inverse
            # to 0 as it will be multiplied by S_min anyway.
            g_diag[g_diag <= 1e-12] = 1.0
        
        inv_g_diag_scaled = S_min / g_diag

        # Create the off-diagonal gain matrix.
        G_off_diag = G - np.diag(g_diag)

        # Initialize power vector to the minimum possible values.
        P = P_min.copy()

        # Iterate until convergence. A fixed number of iterations provides a
        # safeguard against pathological cases. Convergence is typically fast.
        for _ in range(500):
            # Calculate the target power T(P) required to meet SINR for each user,
            # given the interference from other users' current power P.
            interference = G_off_diag @ P
            P_target = (sigma + interference) * inv_g_diag_scaled

            # The new power is the target power, but clipped to the allowed range.
            P_new = np.maximum(P_min, np.minimum(P_target, P_max))

            # Check for convergence using a tight tolerance.
            if np.allclose(P, P_new, rtol=1e-12, atol=1e-14):
                P = P_new
                break
            
            P = P_new
        
        # After convergence, verify if the solution is feasible. Infeasibility
        # arises if the required power to meet SINR for some user exceeds its P_max.
        # The SINR constraint is P >= P_target.
        final_interference = G_off_diag @ P
        final_P_target = (sigma + final_interference) * inv_g_diag_scaled

        # Check feasibility with a small tolerance for floating point errors.
        if not np.all(P >= final_P_target - 1e-9):
            raise ValueError("Solver failed to find a feasible solution.")

        return {"P": P.tolist(), "objective": float(np.sum(P))}