import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        observed = np.array(problem["tensor"], dtype=float)
        mask = np.array(problem["mask"], dtype=bool)
        d1, d2, d3 = observed.shape
        p, q, r = d2 * d3, d1 * d3, d1 * d2

        # ADMM parameters
        max_iter = 10
        tol = 1e-3
        mu = 1.0
        tau = 1.0

        # Initialize X
        X = observed.copy()
        if mask.sum() > 0:
            X[~mask] = observed[mask].mean()
        else:
            X[~mask] = 0.0

        # Dual and auxiliary variables
        Y1 = X.copy()
        Y2 = X.copy()
        Y3 = X.copy()
        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)
        Z3 = np.zeros_like(X)

        svd = np.linalg.svd
        maxi = np.maximum

        for _ in range(max_iter):
            X_prev = X

            # mode-1 unfolding and SVT
            M1 = (X + Z1 / mu).reshape(d1, p)
            U1, S1, Vt1 = svd(M1, full_matrices=False)
            S1t = maxi(S1 - tau, 0)
            M1_hat = (U1 * S1t) @ Vt1
            Y1 = M1_hat.reshape(d1, d2, d3)

            # mode-2 unfolding and SVT
            M2 = (X + Z2 / mu).transpose(1, 0, 2).reshape(d2, q)
            U2, S2, Vt2 = svd(M2, full_matrices=False)
            S2t = maxi(S2 - tau, 0)
            M2_hat = (U2 * S2t) @ Vt2
            Y2 = M2_hat.reshape(d2, d1, d3).transpose(1, 0, 2)

            # mode-3 unfolding and SVT
            M3 = (X + Z3 / mu).transpose(2, 0, 1).reshape(d3, r)
            U3, S3, Vt3 = svd(M3, full_matrices=False)
            S3t = maxi(S3 - tau, 0)
            M3_hat = (U3 * S3t) @ Vt3
            Y3 = M3_hat.reshape(d3, d1, d2).transpose(1, 2, 0)

            # primal update and enforce observed entries
            X = (Y1 - Z1 + Y2 - Z2 + Y3 - Z3) / 3.0
            X[mask] = observed[mask]

            # dual update
            Z1 += X - Y1
            Z2 += X - Y2
            Z3 += X - Y3

            # convergence check
            if np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-8) < tol:
                break

        return {"completed_tensor": X.tolist()}