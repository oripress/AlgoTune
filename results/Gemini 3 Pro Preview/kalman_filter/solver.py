import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def solve_block_tridiag(N, n, E, F, G, E_end, R, C_prime, D_prime, Z):
    # Forward elimination
    
    # Precompute inverse of E for the first step
    # E is 2n x 2n
    E_inv = np.linalg.inv(E)
    C_prime[0] = E_inv @ F
    D_prime[0] = E_inv @ R[0]
    
    # Variables for convergence check
    converged = False
    M_inv = np.zeros_like(E)
    
    # Loop i = 1 to N-2
    for i in range(1, N - 1):
        if not converged:
            # M = E - G @ C_prime[i-1]
            # G has structure [[-A, 0], [0, 0]]
            # G @ C_prime[i-1] = [[-A @ C_prime[i-1][:n, :], 0], [0, 0]]
            # But doing full matmul is fine for small n, or we can optimize if needed.
            # Let's stick to full matmul for simplicity unless n is large.
            
            M = E - G @ C_prime[i-1]
            M_inv = np.linalg.inv(M)
            
            C_prime[i] = M_inv @ F
            
            # Check convergence
            # Heuristic: if change is small enough.
            # For float64, 1e-9 is safe.
            if i > 1:
                diff = np.sum(np.abs(C_prime[i] - C_prime[i-1]))
                if diff < 1e-9:
                    converged = True
        else:
            # If converged, C_prime[i] is same as C_prime[i-1]
            C_prime[i] = C_prime[i-1]
            # M_inv is already computed and constant
        
        # D_prime calculation always needs to be done as R[i] changes
        # rhs_temp = R[i] - G @ D_prime[i-1]
        # G @ D_prime[i-1] = [-A @ D_prime[i-1][:n], 0]
        
        # Using full matmul for G @ D_prime[i-1]
        rhs_temp = R[i] - G @ D_prime[i-1]
        D_prime[i] = M_inv @ rhs_temp
        
    # Last step i = N-1
    if N > 1:
        M = E_end - G @ C_prime[N-2]
        rhs_temp = R[N-1] - G @ D_prime[N-2]
        D_prime[N-1] = np.linalg.solve(M, rhs_temp)
    else:
        # N=1 case
        # System is E_end Z_0 = R[0] ?
        # No, for N=1, we have just one block.
        # The loop range(1, 0) is empty.
        # We need to handle N=1 separately or adjust logic.
        pass

    # Back substitution
    Z[N-1] = D_prime[N-1]
    for i in range(N-2, -1, -1):
        Z[i] = D_prime[i] - C_prime[i] @ Z[i+1]

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        if N == 0:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Precompute matrices
        D = B @ B.T
        Q = tau * (C.T @ C)
        AT = A.T
        
        # Block matrices (2n x 2n)
        # E = [[I, D], [-Q, I]]
        E = np.zeros((2*n, 2*n))
        E[:n, :n] = np.eye(n)
        E[:n, n:] = D
        E[n:, :n] = -Q
        E[n:, n:] = np.eye(n)
        
        # F = [[0, 0], [0, -AT]]
        F = np.zeros((2*n, 2*n))
        F[n:, n:] = -AT
        
        # G = [[-A, 0], [0, 0]]
        G = np.zeros((2*n, 2*n))
        G[:n, :n] = -A
        
        # E_end = [[I, D], [0, I]]
        E_end = np.zeros((2*n, 2*n))
        E_end[:n, :n] = np.eye(n)
        E_end[:n, n:] = D
        E_end[n:, n:] = np.eye(n)
        
        # RHS
        # R[0] = [A x0, -tau C.T y_1]
        # R[t] = [0, -tau C.T y_{t+1}]
        # R[N-1] = [0, 0]
        
        R = np.zeros((N, 2*n))
        
        # t=0 (index 0)
        R[0, :n] = A @ x0
        if N > 1:
            R[0, n:] = -tau * (C.T @ y[1])
            
            # t=1..N-2
            if N > 2:
                # y indices 2 to N-1
                Y_part = y[2:]
                R[1:N-1, n:] = -tau * (Y_part @ C)
        
        # For N=1, R[0] is just [A x0, 0] because y_1 doesn't exist?
        # If N=1, we have x_1. y has y_0.
        # My derivation assumed y_1...y_{N-1}.
        # If N=1, loop for measurements is empty?
        # Problem: minimize sum_{t=0}^{N-1} ...
        # If N=1, sum is t=0.
        # x_1 = A x_0 + B w_0.
        # y_0 = C x_0 + v_0.
        # As discussed, y_0 term is constant.
        # So for N=1, we just minimize ||w_0||^2 subject to x_1 = A x_0 + B w_0.
        # Solution: w_0 = 0 (if possible).
        # My code sets R[0, n:] = 0 if N=1.
        # E_end is used for N=1 block.
        # E_end = [[I, D], [0, I]].
        # System: E_end Z_0 = R[0].
        # [I D; 0 I] [x_1; lambda_1] = [A x0; 0].
        # lambda_1 = 0.
        # x_1 + D*0 = A x0 -> x_1 = A x0.
        # w_0 = -B.T * 0 = 0.
        # This is correct for minimizing ||w_0||^2.
        
        # Allocate arrays for Numba
        C_prime = np.zeros((N, 2*n, 2*n))
        D_prime = np.zeros((N, 2*n))
        Z = np.zeros((N, 2*n))
        
        if N == 1:
            # Special handling for N=1 to avoid Numba issues or complex logic inside
            # E_end Z_0 = R[0]
            Z[0] = np.linalg.solve(E_end, R[0])
        else:
            solve_block_tridiag(N, n, E, F, G, E_end, R, C_prime, D_prime, Z)
        
        # Extract results
        x_hat = Z[:, :n]
        l_hat = Z[:, n:]
        
        x_hat_full = np.vstack([x0, x_hat])
        w_hat = (-B.T @ l_hat.T).T
        v_hat = y - (C @ x_hat_full[:-1].T).T
        
        return {
            "x_hat": x_hat_full.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist()
        }