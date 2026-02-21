import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        dim1, dim2, dim3 = tensor_dims

        M = observed_tensor.reshape(dim1, dim2 * dim3)
        Omega = mask.reshape(dim1, dim2 * dim3)
        M_Omega = M[Omega]

        rho = 1.0
        max_iter = 1000
        tol = 1e-3

        X = np.zeros_like(M)
        X[Omega] = M_Omega
        Z = np.zeros_like(M)
        U = np.zeros_like(M)
        A = np.zeros_like(M)

        for _ in range(max_iter):
            np.subtract(Z, U, out=X)
            X[Omega] = M_Omega

            np.add(X, U, out=A)
            try:
                U_svd, S_svd, Vh_svd = np.linalg.svd(A, full_matrices=False)
            except np.linalg.LinAlgError:
                return {"completed_tensor": []}
            
            S_thresh = np.maximum(S_svd - 1.0 / rho, 0)
            U_svd *= S_thresh
            Z_new = np.matmul(U_svd, Vh_svd)

            if _ % 5 == 0:
                if np.linalg.norm(Z_new - Z, 'fro') / (np.linalg.norm(Z, 'fro') + 1e-8) < tol:
                    X = Z_new - U
                    X[Omega] = M_Omega
                    break
            
            Z = Z_new
            U += X
            U -= Z

        completed_tensor = X.reshape(tensor_dims)
        return {"completed_tensor": completed_tensor.tolist()}