import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        dim1, dim2, dim3 = tensor_dims

        # Mode 1 unfolding
        M = observed_tensor.reshape(dim1, dim2 * dim3)
        mask1 = mask.reshape(dim1, dim2 * dim3)
        obs_vals = M[mask1]

        # ADMM for: minimize ||Z||_* s.t. X = Z, X[mask1] = obs_vals
        m, n = dim1, dim2 * dim3

        X = np.zeros((m, n))
        X[mask1] = obs_vals
        Z = X.copy()
        Y = np.zeros((m, n))

        rho = 1.0
        max_iter = 2000
        atol = 1e-6
        rtol = 1e-5

        for iteration in range(max_iter):
            # X-update
            X = Z - Y / rho
            X[mask1] = obs_vals

            # Z-update: SVT
            W = X + Y / rho
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            s_thresh = np.maximum(s - 1.0 / rho, 0)
            Z_new = (U * s_thresh) @ Vt

            # Residuals
            R = X - Z_new
            S = rho * (Z_new - Z)

            Y = Y + rho * R

            primal_res = np.linalg.norm(R, 'fro')
            dual_res = np.linalg.norm(S, 'fro')

            Z = Z_new

            eps_pri = atol * np.sqrt(m * n) + rtol * max(np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro'))
            eps_dual = atol * np.sqrt(m * n) + rtol * np.linalg.norm(Y, 'fro')

            if primal_res < eps_pri and dual_res < eps_dual:
                break

            if primal_res > 10 * dual_res:
                rho *= 2.0
                Y /= 2.0
            elif dual_res > 10 * primal_res:
                rho /= 2.0
                Y *= 2.0

        result = Z.copy()
        result[mask1] = obs_vals
        completed_tensor = result.reshape(tensor_dims)
        return {"completed_tensor": completed_tensor.tolist()}