import numpy as np
from scipy.sparse import lil_matrix, diags, bmat, eye as speye
from scipy.sparse.linalg import spsolve

class Solver:
    def solve(self, problem: dict) -> dict:
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        # Use bmat for efficient block matrix construction
        # Variables: [x_0, ..., x_N, w_0, ..., w_{N-1}, v_0, ..., v_{N-1}]
        n_x = (N + 1) * n
        n_w = N * p
        n_v = N * m
        
        # Hessian: diag([0, ..., 0, 2, ..., 2, 2*tau, ..., 2*tau])
        H = diags(np.concatenate([np.zeros(n_x), 2.0*np.ones(n_w), 2.0*tau*np.ones(n_v)]), format='csr')
        
        # Build constraint matrix using block structure
        # Constraint blocks:
        # [I_n, 0, ...] for x_0 = x_initial
        # [A, -I_n, B, 0, ...] for dynamics
        # [C, 0, 0, I_m, ...] for measurements
        
        blocks_A = []
        b_parts = []
        
        # Initial constraint: x_0 = x_initial
        row = lil_matrix((n, n_x + n_w + n_v))
        row[:, :n] = np.eye(n)
        blocks_A.append(row)
        b_parts.append(x0)
        
        # Dynamics: x_{t+1} = A*x_t + B*w_t (rewritten as x_{t+1} - A*x_t - B*w_t = 0)
        for t in range(N):
            row = lil_matrix((n, n_x + n_w + n_v))
            row[:, (t+1)*n:(t+2)*n] = np.eye(n)  # x_{t+1}
            row[:, t*n:(t+1)*n] = -A  # -A*x_t
            row[:, n_x + t*p:n_x + (t+1)*p] = -B  # -B*w_t
            blocks_A.append(row)
            b_parts.append(np.zeros(n))
        
        # Measurements: C*x_t + v_t = y_t
        for t in range(N):
            row = lil_matrix((m, n_x + n_w + n_v))
            row[:, t*n:(t+1)*n] = C  # C*x_t
            row[:, n_x + n_w + t*m:n_x + n_w + (t+1)*m] = np.eye(m)  # v_t
            blocks_A.append(row)
            b_parts.append(y[t])
        
        # Stack all constraints
        from scipy.sparse import vstack as sp_vstack
        A_mat = sp_vstack(blocks_A).tocsr()
        b = np.concatenate(b_parts)
        
        # KKT system
        n_cons = A_mat.shape[0]
        n_vars = n_x + n_w + n_v
        KKT = bmat([[H, A_mat.T], [A_mat, None]], format='csr')
        
        rhs = np.concatenate([np.zeros(n_vars), b])
        
        # Solve
        sol = spsolve(KKT, rhs)
        
        # Extract
        x_hat = sol[:n_x].reshape(N+1, n)
        w_hat = sol[n_x:n_x+n_w].reshape(N, p)
        v_hat = sol[n_x+n_w:n_x+n_w+n_v].reshape(N, m)
        
        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }