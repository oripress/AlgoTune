import numpy as np
from numba import njit
from scipy.integrate._ivp.rk import RK45

RK45_A = np.array(RK45.A, dtype=np.float64)
RK45_B = np.array(RK45.B, dtype=np.float64)
RK45_C = np.array(RK45.C, dtype=np.float64)
RK45_E = np.array(RK45.E, dtype=np.float64)

@njit(fastmath=True)
def burgers_equation_numba(t, u, nu, dx, du_dt):
    n = len(u)
    dx2 = dx * dx
    
    if n == 0:
        return
        
    # i = 0
    u_curr = u[0]
    u_prev = 0.0
    u_next = u[1] if n > 1 else 0.0
    diffusion = (u_next - 2.0 * u_curr + u_prev) / dx2
    du_dx = (u_curr - u_prev) / dx if u_curr >= 0 else (u_next - u_curr) / dx
    du_dt[0] = -u_curr * du_dx + nu * diffusion
    
    # i = 1 to n-2
    for i in range(1, n - 1):
        u_curr = u[i]
        u_prev = u[i-1]
        u_next = u[i+1]
        
        diffusion = (u_next - 2.0 * u_curr + u_prev) / dx2
        du_dx = (u_curr - u_prev) / dx if u_curr >= 0 else (u_next - u_curr) / dx
        du_dt[i] = -u_curr * du_dx + nu * diffusion
        
    # i = n - 1
    if n > 1:
        u_curr = u[n-1]
        u_prev = u[n-2]
        u_next = 0.0
        diffusion = (u_next - 2.0 * u_curr + u_prev) / dx2
        du_dx = (u_curr - u_prev) / dx if u_curr >= 0 else (u_next - u_curr) / dx
        du_dt[n-1] = -u_curr * du_dx + nu * diffusion

@njit(fastmath=True)
def rk45_integrate(t0, t1, y0, nu, dx, rtol, atol, A, B, C, E):
    n = len(y0)
    t = t0
    y = y0.copy()
    
    f0 = np.empty(n, dtype=np.float64)
    burgers_equation_numba(t, y, nu, dx, f0)
    
    # Initial step size selection
    d0_sq = 0.0
    d1_sq = 0.0
    for k in range(n):
        sk = atol + abs(y[k]) * rtol
        d0_sq += (y[k] / sk) ** 2
        d1_sq += (f0[k] / sk) ** 2
    d0 = np.sqrt(d0_sq / n)
    d1 = np.sqrt(d1_sq / n)
    
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
        
    y1 = np.empty(n, dtype=np.float64)
    for k in range(n):
        y1[k] = y[k] + h0 * f0[k]
        
    f1 = np.empty(n, dtype=np.float64)
    burgers_equation_numba(t + h0, y1, nu, dx, f1)
    
    d2_sq = 0.0
    for k in range(n):
        sk = atol + abs(y[k]) * rtol
        d2_sq += ((f1[k] - f0[k]) / sk) ** 2
    d2 = np.sqrt(d2_sq / n) / h0
    
    max_d1_d2 = max(d1, d2)
    if max_d1_d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max_d1_d2) ** 0.2
        
    h_abs = min(100.0 * h0, h1)
    
    max_factor = 10.0
    min_factor = 0.2
    
    K = np.zeros((7, n), dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    err = np.empty(n, dtype=np.float64)
    y_new = np.empty(n, dtype=np.float64)
    y_tmp = np.empty(n, dtype=np.float64)
    
    while t < t1:
        step_accepted = False
        step_rejected = False
        
        while not step_accepted:
            t_new = t + h_abs
            if t_new > t1:
                t_new = t1
            h = t_new - t
            
            for k in range(n):
                K[0, k] = f0[k]
            
            for i in range(1, 6):
                dy.fill(0.0)
                for j in range(i):
                    A_ij = A[i, j]
                    if A_ij != 0:
                        for k in range(n):
                            dy[k] += A_ij * K[j, k]
                            
                for k in range(n):
                    y_tmp[k] = y[k] + h * dy[k]
                    
                burgers_equation_numba(t + C[i] * h, y_tmp, nu, dx, K[i])
                
            for k in range(n):
                y_new[k] = y[k]
                
            for i in range(6):
                B_i = B[i]
                if B_i != 0:
                    for k in range(n):
                        y_new[k] += h * B_i * K[i, k]
                        
            burgers_equation_numba(t + h, y_new, nu, dx, K[6])
            
            err.fill(0.0)
            for i in range(7):
                E_i = E[i]
                if E_i != 0:
                    for k in range(n):
                        err[k] += h * E_i * K[i, k]
                        
            err_norm_sq = 0.0
            for k in range(n):
                sk = atol + max(abs(y[k]), abs(y_new[k])) * rtol
                err_norm_sq += (err[k] / sk) ** 2
            err_norm = np.sqrt(err_norm_sq / n)
            
            if err_norm < 1.0:
                if err_norm == 0.0:
                    factor = max_factor
                else:
                    factor = min(max_factor, 0.9 * err_norm ** -0.2)
                    
                if step_rejected:
                    factor = min(1.0, factor)
                    
                h_abs *= factor
                step_accepted = True
            else:
                factor = max(min_factor, 0.9 * err_norm ** -0.2)
                h_abs *= factor
                step_rejected = True
                
        t = t_new
        for k in range(n):
            y[k] = y_new[k]
            f0[k] = K[6, k]

    return y

class Solver:
    def solve(self, problem: dict, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]
        
        y_final = rk45_integrate(
            t0, t1, y0, nu, dx, 1e-6, 1e-6,
            RK45_A, RK45_B, RK45_C, RK45_E
        )
        
        return y_final.tolist()