import numpy as np
from numba import njit
import math

# RK45 Coefficients (Dormand-Prince)
C2, C3, C4, C5, C6 = 1/5, 3/10, 4/5, 8/9, 1.0
A21 = 1/5
A31, A32 = 3/40, 9/40
A41, A42, A43 = 44/45, -56/15, 32/9
A51, A52, A53, A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
A61, A62, A63, A64, A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
A71, A73, A74, A75, A76 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
E1, E3, E4, E5, E6, E7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

# Precomputed constants
INV_10 = 0.1
INV_18 = 1.0 / 18.0
INV_20 = 0.05
INV_80 = 0.0125

@njit(fastmath=True)
def compute_rates(V):
    vp40 = V + 40.0
    if abs(vp40) < 1e-6:
        alpha_m = 1.0
    else:
        alpha_m = (0.1 * vp40) / (1.0 - math.exp(-vp40 * INV_10))
    
    vp65 = V + 65.0
    beta_m = 4.0 * math.exp(-vp65 * INV_18)

    alpha_h = 0.07 * math.exp(-vp65 * INV_20)
    beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) * INV_10))

    vp55 = V + 55.0
    if abs(vp55) < 1e-6:
        alpha_n = 0.1
    else:
        alpha_n = (0.01 * vp55) / (1.0 - math.exp(-vp55 * INV_10))

    beta_n = 0.125 * math.exp(-vp65 * INV_80)
    
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

@njit(fastmath=True)
def rhs_scalar(V, m, h, n, params):
    C_m = params[0]
    g_Na = params[1]
    g_K = params[2]
    g_L = params[3]
    E_Na = params[4]
    E_K = params[5]
    E_L = params[6]
    I_app = params[7]
    
    m_c = 0.0 if m < 0.0 else (1.0 if m > 1.0 else m)
    h_c = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
    n_c = 0.0 if n < 0.0 else (1.0 if n > 1.0 else n)
    
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = compute_rates(V)
    
    m3 = m_c * m_c * m_c
    n4 = n_c * n_c
    n4 = n4 * n4
    
    I_Na = g_Na * m3 * h_c * (V - E_Na)
    I_K = g_K * n4 * (V - E_K)
    I_L = g_L * (V - E_L)
    
    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m_c) - beta_m * m_c
    dhdt = alpha_h * (1.0 - h_c) - beta_h * h_c
    dndt = alpha_n * (1.0 - n_c) - beta_n * n_c
    
    return dVdt, dmdt, dhdt, dndt

@njit(fastmath=True)
def solve_adaptive_scalar(y0, t0, t1, params):
    V, m, h_var, n = y0[0], y0[1], y0[2], y0[3]
    t = t0
    h = 0.05 
    
    rtol = 1e-8
    atol = 1e-8
    
    fac = 0.9
    fac_max = 10.0
    
    k1_0, k1_1, k1_2, k1_3 = rhs_scalar(V, m, h_var, n, params)
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
            if h < 1e-14:
                break
        
        # k2
        k2_0, k2_1, k2_2, k2_3 = rhs_scalar(
            V + h * (A21 * k1_0),
            m + h * (A21 * k1_1),
            h_var + h * (A21 * k1_2),
            n + h * (A21 * k1_3),
            params
        )
        
        # k3
        k3_0, k3_1, k3_2, k3_3 = rhs_scalar(
            V + h * (A31 * k1_0 + A32 * k2_0),
            m + h * (A31 * k1_1 + A32 * k2_1),
            h_var + h * (A31 * k1_2 + A32 * k2_2),
            n + h * (A31 * k1_3 + A32 * k2_3),
            params
        )
        
        # k4
        k4_0, k4_1, k4_2, k4_3 = rhs_scalar(
            V + h * (A41 * k1_0 + A42 * k2_0 + A43 * k3_0),
            m + h * (A41 * k1_1 + A42 * k2_1 + A43 * k3_1),
            h_var + h * (A41 * k1_2 + A42 * k2_2 + A43 * k3_2),
            n + h * (A41 * k1_3 + A42 * k2_3 + A43 * k3_3),
            params
        )
        
        # k5
        k5_0, k5_1, k5_2, k5_3 = rhs_scalar(
            V + h * (A51 * k1_0 + A52 * k2_0 + A53 * k3_0 + A54 * k4_0),
            m + h * (A51 * k1_1 + A52 * k2_1 + A53 * k3_1 + A54 * k4_1),
            h_var + h * (A51 * k1_2 + A52 * k2_2 + A53 * k3_2 + A54 * k4_2),
            n + h * (A51 * k1_3 + A52 * k2_3 + A53 * k3_3 + A54 * k4_3),
            params
        )
        
        # k6
        k6_0, k6_1, k6_2, k6_3 = rhs_scalar(
            V + h * (A61 * k1_0 + A62 * k2_0 + A63 * k3_0 + A64 * k4_0 + A65 * k5_0),
            m + h * (A61 * k1_1 + A62 * k2_1 + A63 * k3_1 + A64 * k4_1 + A65 * k5_1),
            h_var + h * (A61 * k1_2 + A62 * k2_2 + A63 * k3_2 + A64 * k4_2 + A65 * k5_2),
            n + h * (A61 * k1_3 + A62 * k2_3 + A63 * k3_3 + A64 * k4_3 + A65 * k5_3),
            params
        )
        
        # y_new
        V_new = V + h * (A71 * k1_0 + A73 * k3_0 + A74 * k4_0 + A75 * k5_0 + A76 * k6_0)
        m_new = m + h * (A71 * k1_1 + A73 * k3_1 + A74 * k4_1 + A75 * k5_1 + A76 * k6_1)
        h_new = h_var + h * (A71 * k1_2 + A73 * k3_2 + A74 * k4_2 + A75 * k5_2 + A76 * k6_2)
        n_new = n + h * (A71 * k1_3 + A73 * k3_3 + A74 * k4_3 + A75 * k5_3 + A76 * k6_3)
        
        # k7 (FSAL)
        k7_0, k7_1, k7_2, k7_3 = rhs_scalar(V_new, m_new, h_new, n_new, params)
        
        # Error
        e_0 = h * (E1 * k1_0 + E3 * k3_0 + E4 * k4_0 + E5 * k5_0 + E6 * k6_0 + E7 * k7_0)
        e_1 = h * (E1 * k1_1 + E3 * k3_1 + E4 * k4_1 + E5 * k5_1 + E6 * k6_1 + E7 * k7_1)
        e_2 = h * (E1 * k1_2 + E3 * k3_2 + E4 * k4_2 + E5 * k5_2 + E6 * k6_2 + E7 * k7_2)
        e_3 = h * (E1 * k1_3 + E3 * k3_3 + E4 * k4_3 + E5 * k5_3 + E6 * k6_3 + E7 * k7_3)
        
        # Error ratio
        sc_0 = atol + max(abs(V), abs(V_new)) * rtol
        sc_1 = atol + max(abs(m), abs(m_new)) * rtol
        sc_2 = atol + max(abs(h_var), abs(h_new)) * rtol
        sc_3 = atol + max(abs(n), abs(n_new)) * rtol
        
        err_ratio = max(abs(e_0)/sc_0, abs(e_1)/sc_1)
        err_ratio = max(err_ratio, abs(e_2)/sc_2)
        err_ratio = max(err_ratio, abs(e_3)/sc_3)
        
        if err_ratio < 1.0:
            t += h
            V, m, h_var, n = V_new, m_new, h_new, n_new
            k1_0, k1_1, k1_2, k1_3 = k7_0, k7_1, k7_2, k7_3
            
            if err_ratio == 0.0:
                h *= fac_max
            else:
                h *= fac * (1.0 / err_ratio) ** 0.2
                if h > 20.0: h = 20.0
        else:
            h *= fac * (1.0 / err_ratio) ** 0.2
            if h < 1e-15:
                pass
                
    return np.array([V, m, h_var, n])

class Solver:
    def solve(self, problem, **kwargs):
        p = problem["params"]
        params_tuple = (
            float(p["C_m"]), float(p["g_Na"]), float(p["g_K"]), float(p["g_L"]),
            float(p["E_Na"]), float(p["E_K"]), float(p["E_L"]), float(p["I_app"])
        )
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        
        y_final = solve_adaptive_scalar(y0, t0, t1, params_tuple)
        return list(y_final)