import numpy as np
from numba import njit

@njit(fastmath=True)
def solve_rk4_fixed(x, y, t0, t1, alpha, beta, delta, gamma):
    # Fixed step size
    # Trying larger step size for more speed
    h_target = 2.1e-2
    
    duration = t1 - t0
    if duration <= 0:
        return [x, y]
        
    num_steps = int(np.ceil(duration / h_target))
    h = duration / num_steps
    
    h_half = 0.5 * h
    h_sixth = h / 6.0
    
    for _ in range(num_steps):
        # k1
        k1x = x * (alpha - beta * y)
        k1y = y * (delta * x - gamma)
        
        # k2
        x2 = x + h_half * k1x
        y2 = y + h_half * k1y
        k2x = x2 * (alpha - beta * y2)
        k2y = y2 * (delta * x2 - gamma)
        
        # k3
        x3 = x + h_half * k2x
        y3 = y + h_half * k2y
        k3x = x3 * (alpha - beta * y3)
        k3y = y3 * (delta * x3 - gamma)
        
        # k4
        x4 = x + h * k3x
        y4 = y + h * k3y
        k4x = x4 * (alpha - beta * y4)
        k4y = y4 * (delta * x4 - gamma)
        
        # Update
        x += h_sixth * (k1x + 2*(k2x + k3x) + k4x)
        y += h_sixth * (k1y + 2*(k2y + k3y) + k4y)
        
    return [x, y]

class Solver:
    def __init__(self):
        # Trigger compilation
        solve_rk4_fixed(10.0, 5.0, 0.0, 0.1, 1.1, 0.4, 0.1, 0.4)

    def solve(self, problem, **kwargs):
        y0 = problem["y0"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        
        res = solve_rk4_fixed(y0[0], y0[1], t0, t1, params["alpha"], params["beta"], params["delta"], params["gamma"])
        return list(res)