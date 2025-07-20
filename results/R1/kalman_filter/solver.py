import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Extract problem data
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])
        
        N = y.shape[0]  # Number of measurement time steps
        n = A.shape[0]   # State dimension
        m = C.shape[0]   # Measurement dimension
        p = B.shape[1]   # Process noise dimension
        
        # Handle case with no measurements
        if N == 0:
            return {
                "x_hat": [x0.tolist()],
                "w_hat": [],
                "v_hat": []
            }
        
        # Precompute covariance matrices
        Q = B @ B.T
        R = (1/tau) * np.eye(m)
        
        # Initialize arrays
        x_f = np.zeros((N+1, n))    # Filtered states
        P_f = np.zeros((N+1, n, n)) # Filtered covariances
        x_f[0] = x0
        P_f[0] = np.zeros((n, n))
        
        # Forward pass (Kalman filter)
        for t in range(N):
            # Predict next state
            x_pred = A @ x_f[t]
            P_pred = A @ P_f[t] @ A.T + Q
            
            # Update with measurement if available
            if t < N-1:
                residual = y[t+1] - C @ x_pred
                S = C @ P_pred @ C.T + R
                try:
                    K = P_pred @ C.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    K = P_pred @ C.T @ np.linalg.pinv(S)
                x_f[t+1] = x_pred + K @ residual
                P_f[t+1] = (np.eye(n) - K @ C) @ P_pred
            else:  # Final prediction
                x_f[t+1] = x_pred
                P_f[t+1] = P_pred
        
        # Backward pass (RTS smoother)
        x_s = np.zeros((N+1, n))
        x_s[0] = x0
        x_s[N] = x_f[N]
        
        for t in range(N-1, 0, -1):
            P_pred = A @ P_f[t] @ A.T + Q
            try:
                J = P_f[t] @ A.T @ np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                J = P_f[t] @ A.T @ np.linalg.pinv(P_pred)
            x_s[t] = x_f[t] + J @ (x_s[t+1] - A @ x_f[t])
        
        # Compute noise estimates
        w_hat = []
        v_hat = [y[0] - C @ x0]  # v0
        
        for t in range(N):
            # Process noise
            diff = x_s[t+1] - A @ x_s[t]
            w_t, _, _, _ = np.linalg.lstsq(B, diff, rcond=None)
            w_hat.append(w_t.tolist())
            
            # Measurement noise (except last)
            if t < N-1:
                v_hat.append(y[t+1] - C @ x_s[t+1])
        
        return {
            "x_hat": [x.tolist() for x in x_s],
            "w_hat": w_hat,
            "v_hat": [v.tolist() for v in v_hat]
        }