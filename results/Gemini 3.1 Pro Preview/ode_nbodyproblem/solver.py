import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit(fastmath=True)
def nbody_derivatives(t, y, num_bodies, masses, softening_sq, out):
    for i in range(3 * num_bodies):
        out[i] = y[3 * num_bodies + i]
        out[3 * num_bodies + i] = 0.0
        
    for i in range(num_bodies):
        i3 = i * 3
        xi = y[i3]
        yi = y[i3 + 1]
        zi = y[i3 + 2]
        mi = masses[i]
        
        for j in range(i + 1, num_bodies):
            j3 = j * 3
            dx = y[j3] - xi
            dy = y[j3 + 1] - yi
            dz = y[j3 + 2] - zi
            
            dist_sq = dx*dx + dy*dy + dz*dz + softening_sq
            factor = dist_sq ** -1.5
            
            f_j = masses[j] * factor
            f_i = mi * factor
            
            out[3 * num_bodies + i3] += f_j * dx
            out[3 * num_bodies + i3 + 1] += f_j * dy
            out[3 * num_bodies + i3 + 2] += f_j * dz
            
            out[3 * num_bodies + j3] -= f_i * dx
            out[3 * num_bodies + j3 + 1] -= f_i * dy
            out[3 * num_bodies + j3 + 2] -= f_i * dz

@njit(fastmath=True)
def rk45_solve(t0, t1, y0, num_bodies, masses, softening_sq, rtol, atol):
    t = t0
    y = y0.copy()
    h = 1e-4
    
    n = len(y0)
    k1 = np.empty(n)
    k2 = np.empty(n)
    k3 = np.empty(n)
    k4 = np.empty(n)
    k5 = np.empty(n)
    k6 = np.empty(n)
    k7 = np.empty(n)
    y_temp = np.empty(n)
    y_new = np.empty(n)
    
    nbody_derivatives(t, y, num_bodies, masses, softening_sq, k1)
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        for i in range(n):
            y_temp[i] = y[i] + h * (1/5 * k1[i])
        nbody_derivatives(t + 1/5 * h, y_temp, num_bodies, masses, softening_sq, k2)
        
        for i in range(n):
            y_temp[i] = y[i] + h * (3/40 * k1[i] + 9/40 * k2[i])
        nbody_derivatives(t + 3/10 * h, y_temp, num_bodies, masses, softening_sq, k3)
        
        for i in range(n):
            y_temp[i] = y[i] + h * (44/45 * k1[i] - 56/15 * k2[i] + 32/9 * k3[i])
        nbody_derivatives(t + 4/5 * h, y_temp, num_bodies, masses, softening_sq, k4)
        
        for i in range(n):
            y_temp[i] = y[i] + h * (19372/6561 * k1[i] - 25360/2187 * k2[i] + 64448/6561 * k3[i] - 212/729 * k4[i])
        nbody_derivatives(t + 8/9 * h, y_temp, num_bodies, masses, softening_sq, k5)
        
        for i in range(n):
            y_temp[i] = y[i] + h * (9017/3168 * k1[i] - 355/33 * k2[i] + 46732/5247 * k3[i] + 49/176 * k4[i] - 5103/18656 * k5[i])
        nbody_derivatives(t + h, y_temp, num_bodies, masses, softening_sq, k6)
        
        for i in range(n):
            y_new[i] = y[i] + h * (35/384 * k1[i] + 500/1113 * k3[i] + 125/192 * k4[i] - 2187/6784 * k5[i] + 11/84 * k6[i])
        nbody_derivatives(t + h, y_new, num_bodies, masses, softening_sq, k7)
        
        err_sq_sum = 0.0
        for i in range(n):
            err = h * (-71/57600 * k1[i] + 71/16695 * k3[i] - 71/1920 * k4[i] + 17253/339200 * k5[i] - 22/525 * k6[i] + 1/40 * k7[i])
            scale = atol + max(abs(y[i]), abs(y_new[i])) * rtol
            err_sq_sum += (err / scale)**2
            
        err_norm = np.sqrt(err_sq_sum / n)
        
        if err_norm < 1.0:
            t += h
            for i in range(n):
                y[i] = y_new[i]
                k1[i] = k7[i]
            
            if err_norm == 0.0:
                factor = 10.0
            else:
                factor = 0.9 * err_norm**(-0.2)
            factor = min(10.0, max(0.2, factor))
            h *= factor
        else:
            factor = 0.9 * err_norm**(-0.2)
            factor = min(10.0, max(0.2, factor))
            h *= factor

    return y

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"])
        softening = problem["softening"]
        num_bodies = problem["num_bodies"]
        
        softening_sq = softening**2
        
        y_final = rk45_solve(t0, t1, y0, num_bodies, masses, softening_sq, 1e-8, 1e-8)
        
        return y_final.tolist()