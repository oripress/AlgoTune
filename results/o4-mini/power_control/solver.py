import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Parse inputs
        G = np.asarray(problem["G"], dtype=np.float64)
        sigma = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        P_max = np.asarray(problem["P_max"], dtype=np.float64)
        S_min = float(problem["S_min"])
        n = G.shape[0]

        # Build matrix F and vector u for fixed-point update: P = clamp(F P + u)
        diagG = G.diagonal()
        F = (S_min * G) / diagG[:, None]
        np.fill_diagonal(F, 0.0)
        u = S_min * sigma / diagG

        # Initialize P at minimum power
        P = P_min.copy()

        # Fixed-point iterations
        tol = 1e-10
        max_iter = 10000
        for _ in range(max_iter):
            P_next = F.dot(P) + u
            # clamp to bounds in-place
            np.maximum(P_next, P_min, out=P_next)
            np.minimum(P_next, P_max, out=P_next)
            # check convergence
            diff = np.max(np.abs(P_next - P))
            P = P_next
            if diff < tol:
                break

        return {"P": P.tolist(), "objective": float(P.sum())}