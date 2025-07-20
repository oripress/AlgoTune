import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Convert inputs to numpy arrays
        A = np.array(problem['A'], dtype=np.float64, order='C')
        B = np.array(problem['B'], dtype=np.float64, order='C')
        Q = np.array(problem['Q'], dtype=np.float64, order='C')
        R = np.array(problem['R'], dtype=np.float64, order='C')
        P = np.array(problem['P'], dtype=np.float64, order='C')
        T = problem['T']
        x0 = np.array(problem['x0'], dtype=np.float64).flatten()
        
        n, m = B.shape
        
        # Precompute transposes
        A_T = A.T
        B_T = B.T
        
        # Initialize buffers
        S = P.copy()
        K = np.zeros((T, m, n))
        
        # Backward pass - preallocate intermediate buffers
        temp_SA = np.zeros((n, n))
        temp_SB = np.zeros((n, m))
        M1 = np.zeros((m, m))
        M2 = np.zeros((m, n))
        term = np.zeros((n, n))
        S_next = np.zeros((n, n))
        
        for t in range(T-1, -1, -1):
            # Compute intermediate products with BLAS
            np.dot(S, A, temp_SA)
            np.dot(S, B, temp_SB)
            
            # Compute M1 and M2
            np.dot(B_T, temp_SB, M1)
            M1 += R
            np.dot(B_T, temp_SA, M2)
            
            # Solve for Kt using efficient solver
            try:
                # Use Cholesky for positive definite systems
                Kt = np.linalg.solve(M1, M2)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if Cholesky fails
                Kt = np.linalg.pinv(M1) @ M2
            
            # Compute the term for S update
            np.dot(A_T, temp_SA, term)
            
            # Update S for next iteration
            S_next = Q + term - M2.T @ Kt
            
            # Ensure symmetry efficiently
            S_next = (S_next + S_next.T) * 0.5
            
            # Update S for next iteration
            S = S_next
            K[t] = Kt
        
        # Forward simulation - preallocate output
        x = x0.copy()
        U = np.zeros((T, m))
        
        for t in range(T):
            u = -K[t] @ x
            U[t] = u
            x = A @ x + B @ u

        return {"U": U.tolist()}