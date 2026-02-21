import math
from numba import njit

@njit(fastmath=True)
def hodgkin_huxley_numba(V, m, h, n, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    V_plus_65 = V + 65.0
    e_n = math.exp(-V_plus_65 * 0.0125)
    e_n2 = e_n * e_n
    e_n4 = e_n2 * e_n2
    e_n8 = e_n4 * e_n4

    if V == -40.0:
        alpha_m = 1.0
    else:
        alpha_m = (0.1 * (V + 40.0)) / (1.0 - e_n8 * 12.182493960703473)

    beta_m = 4.0 * math.exp(-V_plus_65 * 0.05555555555555555)

    alpha_h = 0.07 * e_n4
    beta_h = 1.0 / (1.0 + e_n8 * 20.085536923187668)

    if V == -55.0:
        alpha_n = 0.1
    else:
        alpha_n = (0.01 * (V + 55.0)) / (1.0 - e_n8 * 2.718281828459045)

    beta_n = 0.125 * e_n

    m = min(max(m, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    n = min(max(n, 0.0), 1.0)

    m3 = m * m * m
    n2 = n * n
    n4 = n2 * n2

    I_Na = g_Na * m3 * h * (V - E_Na)
    I_K = g_K * n4 * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_app - I_Na - I_K - I_L) * C_m_inv
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n

    return dVdt, dmdt, dhdt, dndt

@njit(fastmath=True)
def rk45_step(V, m, h, n, fV, fm, fh, fn, dt, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    k1V, k1m, k1h, k1n = fV, fm, fh, fn
    
    dt_5 = dt * 0.2
    V2 = V + dt_5 * k1V
    m2 = m + dt_5 * k1m
    h2 = h + dt_5 * k1h
    n2 = n + dt_5 * k1n
    k2V, k2m, k2h, k2n = hodgkin_huxley_numba(V2, m2, h2, n2, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    V3 = V + dt * (0.075 * k1V + 0.225 * k2V)
    m3 = m + dt * (0.075 * k1m + 0.225 * k2m)
    h3 = h + dt * (0.075 * k1h + 0.225 * k2h)
    n3 = n + dt * (0.075 * k1n + 0.225 * k2n)
    k3V, k3m, k3h, k3n = hodgkin_huxley_numba(V3, m3, h3, n3, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    V4 = V + dt * (44.0/45.0 * k1V - 56.0/15.0 * k2V + 32.0/9.0 * k3V)
    m4 = m + dt * (44.0/45.0 * k1m - 56.0/15.0 * k2m + 32.0/9.0 * k3m)
    h4 = h + dt * (44.0/45.0 * k1h - 56.0/15.0 * k2h + 32.0/9.0 * k3h)
    n4 = n + dt * (44.0/45.0 * k1n - 56.0/15.0 * k2n + 32.0/9.0 * k3n)
    k4V, k4m, k4h, k4n = hodgkin_huxley_numba(V4, m4, h4, n4, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    V5 = V + dt * (19372.0/6561.0 * k1V - 25360.0/2187.0 * k2V + 64448.0/6561.0 * k3V - 212.0/729.0 * k4V)
    m5 = m + dt * (19372.0/6561.0 * k1m - 25360.0/2187.0 * k2m + 64448.0/6561.0 * k3m - 212.0/729.0 * k4m)
    h5 = h + dt * (19372.0/6561.0 * k1h - 25360.0/2187.0 * k2h + 64448.0/6561.0 * k3h - 212.0/729.0 * k4h)
    n5 = n + dt * (19372.0/6561.0 * k1n - 25360.0/2187.0 * k2n + 64448.0/6561.0 * k3n - 212.0/729.0 * k4n)
    k5V, k5m, k5h, k5n = hodgkin_huxley_numba(V5, m5, h5, n5, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    V6 = V + dt * (9017.0/3168.0 * k1V - 355.0/33.0 * k2V + 46732.0/5247.0 * k3V + 49.0/176.0 * k4V - 5103.0/18656.0 * k5V)
    m6 = m + dt * (9017.0/3168.0 * k1m - 355.0/33.0 * k2m + 46732.0/5247.0 * k3m + 49.0/176.0 * k4m - 5103.0/18656.0 * k5m)
    h6 = h + dt * (9017.0/3168.0 * k1h - 355.0/33.0 * k2h + 46732.0/5247.0 * k3h + 49.0/176.0 * k4h - 5103.0/18656.0 * k5h)
    n6 = n + dt * (9017.0/3168.0 * k1n - 355.0/33.0 * k2n + 46732.0/5247.0 * k3n + 49.0/176.0 * k4n - 5103.0/18656.0 * k5n)
    k6V, k6m, k6h, k6n = hodgkin_huxley_numba(V6, m6, h6, n6, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    V_new = V + dt * (35.0/384.0 * k1V + 500.0/1113.0 * k3V + 125.0/192.0 * k4V - 2187.0/6784.0 * k5V + 11.0/84.0 * k6V)
    m_new = m + dt * (35.0/384.0 * k1m + 500.0/1113.0 * k3m + 125.0/192.0 * k4m - 2187.0/6784.0 * k5m + 11.0/84.0 * k6m)
    h_new = h + dt * (35.0/384.0 * k1h + 500.0/1113.0 * k3h + 125.0/192.0 * k4h - 2187.0/6784.0 * k5h + 11.0/84.0 * k6h)
    n_new = n + dt * (35.0/384.0 * k1n + 500.0/1113.0 * k3n + 125.0/192.0 * k4n - 2187.0/6784.0 * k5n + 11.0/84.0 * k6n)
    
    fV_new, fm_new, fh_new, fn_new = hodgkin_huxley_numba(V_new, m_new, h_new, n_new, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    errV = dt * (-71.0/57600.0 * k1V + 71.0/16695.0 * k3V - 71.0/1920.0 * k4V + 17253.0/339200.0 * k5V - 22.0/525.0 * k6V + 0.025 * fV_new)
    errm = dt * (-71.0/57600.0 * k1m + 71.0/16695.0 * k3m - 71.0/1920.0 * k4m + 17253.0/339200.0 * k5m - 22.0/525.0 * k6m + 0.025 * fm_new)
    errh = dt * (-71.0/57600.0 * k1h + 71.0/16695.0 * k3h - 71.0/1920.0 * k4h + 17253.0/339200.0 * k5h - 22.0/525.0 * k6h + 0.025 * fh_new)
    errn = dt * (-71.0/57600.0 * k1n + 71.0/16695.0 * k3n - 71.0/1920.0 * k4n + 17253.0/339200.0 * k5n - 22.0/525.0 * k6n + 0.025 * fn_new)
    
    return V_new, m_new, h_new, n_new, fV_new, fm_new, fh_new, fn_new, errV, errm, errh, errn

@njit(fastmath=True)
def rms_norm_tuple(e1, e2, e3, e4):
    return math.sqrt((e1*e1 + e2*e2 + e3*e3 + e4*e4) * 0.25)

@njit(fastmath=True)
def select_initial_step(V, m, h, n, fV, fm, fh, fn, rtol, atol, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    scaleV = atol + abs(V) * rtol
    scalem = atol + abs(m) * rtol
    scaleh = atol + abs(h) * rtol
    scalen = atol + abs(n) * rtol
    
    d0 = rms_norm_tuple(V/scaleV, m/scalem, h/scaleh, n/scalen)
    d1 = rms_norm_tuple(fV/scaleV, fm/scalem, fh/scaleh, fn/scalen)
    
    if d0 < 1e-5 or d1 < 1e-5:
        dt0 = 1e-6
    else:
        dt0 = 0.01 * d0 / d1

    V1 = V + dt0 * fV
    m1 = m + dt0 * fm
    h1 = h + dt0 * fh
    n1 = n + dt0 * fn
    
    fV1, fm1, fh1, fn1 = hodgkin_huxley_numba(V1, m1, h1, n1, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    d2 = rms_norm_tuple((fV1 - fV)/scaleV, (fm1 - fm)/scalem, (fh1 - fh)/scaleh, (fn1 - fn)/scalen) / dt0

    max_d1_d2 = max(d1, d2)
    if max_d1_d2 <= 1e-15:
        dt1 = max(1e-6, dt0 * 1e-3)
    else:
        dt1 = (0.01 / max_d1_d2) ** (1.0 / 6.0)

    return min(100.0 * dt0, dt1)

@njit(fastmath=True)
def solve_rk45_numba(t0, t1, V, m, h, n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, rtol, atol):
    t = t0
    C_m_inv = 1.0 / C_m
    
    fV, fm, fh, fn = hodgkin_huxley_numba(V, m, h, n, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    dt = select_initial_step(V, m, h, n, fV, fm, fh, fn, rtol, atol, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    max_factor = 10.0
    min_factor = 0.2
    error_exponent = -0.2
    
    while t < t1:
        if t + dt > t1:
            dt = t1 - t
            
        step_accepted = False
        step_rejected = False
        
        while not step_accepted:
            V_new, m_new, h_new, n_new, fV_new, fm_new, fh_new, fn_new, errV, errm, errh, errn = rk45_step(
                V, m, h, n, fV, fm, fh, fn, dt, C_m_inv, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
            )
            
            scaleV = atol + max(abs(V), abs(V_new)) * rtol
            scalem = atol + max(abs(m), abs(m_new)) * rtol
            scaleh = atol + max(abs(h), abs(h_new)) * rtol
            scalen = atol + max(abs(n), abs(n_new)) * rtol
            
            error_norm = rms_norm_tuple(errV/scaleV, errm/scalem, errh/scaleh, errn/scalen)
            
            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = max_factor
                else:
                    factor = min(max_factor, 0.9 * (error_norm ** error_exponent))
                
                if step_rejected:
                    factor = min(1.0, factor)
                
                t += dt
                V = V_new
                m = m_new
                h = h_new
                n = n_new
                fV = fV_new
                fm = fm_new
                fh = fh_new
                fn = fn_new
                dt *= factor
                step_accepted = True
            else:
                factor = max(min_factor, 0.9 * (error_norm ** error_exponent))
                dt *= factor
                step_rejected = True
                
    return V, m, h, n

class Solver:
    def solve(self, problem, **kwargs):
        y0 = problem["y0"]
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        return list(solve_rk45_numba(
            t0, t1, y0[0], y0[1], y0[2], y0[3], 
            params["C_m"], params["g_Na"], params["g_K"], params["g_L"], 
            params["E_Na"], params["E_K"], params["E_L"], params["I_app"], 
            1e-8, 1e-8
        ))