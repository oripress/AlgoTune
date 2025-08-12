import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Parse problem data
        G = np.array(problem["G"], dtype=float)
        sigma = np.array(problem["σ"], dtype=float)
        P_min = np.array(problem["P_min"], dtype=float)
        P_max = np.array(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])
        n = G.shape[0]

        # Precompute diagonal of G
        diagG = G.diagonal()

        # Initialize power at minimum levels
        P = P_min.copy()
        tol = 1e-9
        # Maximum iterations for fixed-point
        max_iter = 1000

        # Fixed-point iteration: P_i = clamp( S_min*(σ_i + sum_j≠i G[i,j]*P_j)/G_ii, P_min_i, P_max_i )
        for _ in range(max_iter):
            # Compute total received power including self
            sum_all = G.dot(P)
            # Interference excludes self-contribution
            interf = sigma + sum_all - diagG * P
            # Required power to meet SINR
            P_req = S_min * interf / diagG
            # Apply bounds
            P_new = np.minimum(np.maximum(P_req, P_min), P_max)
            # Check convergence
            if np.all(np.abs(P_new - P) <= tol):
                P = P_new
                break
            P = P_new

        # Final adjustment to ensure constraints
        sum_all = G.dot(P)
        interf = sigma + sum_all - diagG * P
        P_req = S_min * interf / diagG
        P = np.minimum(np.maximum(P_req, P_min), P_max)

        return {"P": P.tolist(), "objective": float(P.sum())}