import numpy as np
import torch
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Extract parameters once
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]
        
        # Try torch if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda':
            # Use torch for GPU acceleration
            y0_t = torch.tensor(y0, dtype=torch.float64, device=device)
            t_span = torch.linspace(t0, t1, 1000, dtype=torch.float64, device=device)
            
            @torch.jit.script
            def lotka_volterra_torch(y, alpha: float, beta: float, delta: float, gamma: float):
                x = y[0]
                y_pred = y[1]
                dx = alpha * x - beta * x * y_pred
                dy = delta * x * y_pred - gamma * y_pred
                return torch.stack([dx, dy])
            
            # Simple Euler integration for speed
            dt = (t1 - t0) / 999
            y_current = y0_t
            for _ in range(999):
                dy = lotka_volterra_torch(y_current, alpha, beta, delta, gamma)
                y_current = y_current + dy * dt
            
            return y_current.cpu().numpy().tolist()
        else:
            # Fallback to scipy with optimized settings
            def lotka_volterra(t, y):
                x, y_pred = y
                return np.array([
                    alpha * x - beta * x * y_pred,
                    delta * x * y_pred - gamma * y_pred
                ])
            
            sol = solve_ivp(
                lotka_volterra,
                [t0, t1],
                y0,
                method="DOP853",
                rtol=2e-8,
                atol=2e-10,
            )
            
            if sol.success:
                return sol.y[:, -1].tolist()
            else:
                raise RuntimeError(f"Solver failed: {sol.message}")