import numpy as np
from numba import njit

@njit
def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = -1
    for j in range(n):
        if u[j] + (1.0 - cssv[j]) / (j + 1) > 0:
            rho = j
    
    theta = (1.0 - cssv[rho]) / (rho + 1)
    w = np.maximum(v + theta, 0.0)
    return w

@njit
def solve_pgd(mu, Sigma, gamma, max_iter=5000, tol=1e-7):
    n = len(mu)
    Q = 2 * gamma * Sigma
    
    # Power iteration to estimate Lipschitz constant L (max eigenvalue of Q)
    # Start with a random vector or ones
    v = np.ones(n)
    for _ in range(10):
        v = Q @ v
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            break
        v = v / norm_v
    
    L = v @ (Q @ v)
    
    # If L is too small (e.g. zero matrix), use a default step size
    if L < 1e-8:
        step_size = 1.0
    else:
        step_size = 1.0 / L
        
    # Initialize w
    w = np.ones(n) / n
    
    # FISTA
    y = w.copy()
    t = 1.0
    
    for k in range(max_iter):
        w_old = w.copy()
        
        # Gradient of objective: 0.5 * w^T Q w - mu^T w
        # Grad = Q w - mu
        # Evaluate gradient at y
        grad = Q @ y - mu
        
        # Step
        z = y - step_size * grad
        
        # Project
        w = project_simplex(z)
        
        # Check convergence
        if np.linalg.norm(w - w_old) < tol:
            break
            
        # Update t and y
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w + ((t - 1.0) / t_new) * (w - w_old)
        t = t_new
        
    return w

class Solver:
    def __init__(self):
        # Trigger numba compilation
        dummy_mu = np.array([1.0, 2.0], dtype=float)
        dummy_Sigma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        dummy_gamma = 1.0
        solve_pgd(dummy_mu, dummy_Sigma, dummy_gamma, max_iter=10)

    def solve(self, problem, **kwargs):
        mu = np.array(problem["μ"], dtype=float)
        Sigma = np.array(problem["Σ"], dtype=float)
        gamma = float(problem["γ"])
        
        w = solve_pgd(mu, Sigma, gamma)
        
        return {"w": w.tolist()}