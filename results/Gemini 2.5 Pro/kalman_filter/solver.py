import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solves the Kalman filtering problem by eliminating v_t and building
        the KKT system matrices directly in COO format using vectorized
        numpy operations, avoiding all intermediate matrix objects.
        """
        A_np = np.array(problem["A"])
        B_np = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x_initial = np.array(problem["x_initial"])
        tau = float(problem["tau"])

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        N, m = y.shape
        n = A_np.shape[1]
        p = B_np.shape[1]

        nx = (N + 1) * n
        nw = N * p
        n_vars = nx + nw

        # --- 1. Build P and q (Objective) using vectorized COO ---
        # P matrix (quadratic term)
        CTC = tau * (C.T @ C)
        CTC_coo = sp.coo_matrix(CTC)
        
        # P_x part: N blocks of CTC for x_0, ..., x_{N-1}
        data_Px = np.tile(CTC_coo.data, N)
        rows_Px = np.repeat(np.arange(N) * n, CTC_coo.nnz) + np.tile(CTC_coo.row, N)
        cols_Px = np.repeat(np.arange(N) * n, CTC_coo.nnz) + np.tile(CTC_coo.col, N)

        # P_w part: N blocks of Identity for w_0, ..., w_{N-1}
        I_p_indices = np.arange(p)
        data_Pw = np.tile(np.ones(p), N)
        rows_Pw = nx + np.repeat(np.arange(N) * p, p) + np.tile(I_p_indices, N)
        cols_Pw = nx + np.repeat(np.arange(N) * p, p) + np.tile(I_p_indices, N)

        P = sp.csc_matrix(
            (np.concatenate([data_Px, data_Pw]), (np.concatenate([rows_Px, rows_Pw]), np.concatenate([cols_Px, cols_Pw]))),
            shape=(n_vars, n_vars)
        )

        # q vector (linear term)
        q_x = (-tau * (C.T @ y.T)).flatten(order='F')
        q = np.concatenate([q_x, np.zeros(n + nw)])

        # --- 2. Build A_eq and b_eq (Constraints) using vectorized COO ---
        n_cons = n + N * n
        b_eq = np.zeros(n_cons)
        b_eq[:n] = x_initial.flatten()

        A_coo = sp.coo_matrix(A_np)
        B_coo = sp.coo_matrix(B_np)

        # Constraint 1: x_0 = I
        d_x0, r_x0, c_x0 = np.ones(n), np.arange(n), np.arange(n)

        # Constraint 2: x_{t+1} - A*x_t - B*w_t = 0
        # Term: I * x_{t+1} for t=0..N-1 -> affects x_1..x_N
        d_I = np.ones(N * n)
        r_I = n + np.arange(N * n)
        c_I = n + np.arange(N * n)

        # Term: -A * x_t for t=0..N-1
        d_A = np.tile(-A_coo.data, N)
        r_A = n + np.repeat(np.arange(N) * n, A_coo.nnz) + np.tile(A_coo.row, N)
        c_A = np.repeat(np.arange(N) * n, A_coo.nnz) + np.tile(A_coo.col, N)

        # Term: -B * w_t for t=0..N-1
        d_B = np.tile(-B_coo.data, N)
        r_B = n + np.repeat(np.arange(N) * n, B_coo.nnz) + np.tile(B_coo.row, N)
        c_B = nx + np.repeat(np.arange(N) * p, B_coo.nnz) + np.tile(B_coo.col, N)

        A_eq = sp.csc_matrix(
            (np.concatenate([d_x0, d_I, d_A, d_B]), (np.concatenate([r_x0, r_I, r_A, r_B]), np.concatenate([c_x0, c_I, c_A, c_B]))),
            shape=(n_cons, n_vars)
        )

        # --- 3. Assemble and solve KKT ---
        KKT = sp.bmat([[P, A_eq.T], [A_eq, None]], format='csc')
        rhs = np.concatenate([-q, b_eq])

        try:
            sol = spsolve(KKT, rhs, use_umfpack=True)
            if np.any(np.isnan(sol)):
                return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # --- 4. Extract results and compute v_hat ---
        x_hat = sol[:nx].reshape((N + 1, n))
        w_hat = sol[nx:nx + nw].reshape((N, p))
        v_hat = y - (C @ x_hat[:-1].T).T

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }