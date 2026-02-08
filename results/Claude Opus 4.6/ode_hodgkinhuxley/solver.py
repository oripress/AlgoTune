import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def hh_rhs(y, out, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    V = y[0]
    m_ = y[1]
    h_ = y[2]
    n_ = y[3]
    
    v40 = V + 40.0
    if abs(v40) < 1e-7:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * v40 / (1.0 - np.exp(-v40 / 10.0))
    
    vp65 = V + 65.0
    beta_m = 4.0 * np.exp(-vp65 / 18.0)
    alpha_h = 0.07 * np.exp(-vp65 / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    v55 = V + 55.0
    if abs(v55) < 1e-7:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * v55 / (1.0 - np.exp(-v55 / 10.0))
    
    beta_n = 0.125 * np.exp(-vp65 / 80.0)
    
    mc = min(max(m_, 0.0), 1.0)
    hc = min(max(h_, 0.0), 1.0)
    nc = min(max(n_, 0.0), 1.0)
    
    out[0] = (I_app - g_Na * mc*mc*mc * hc * (V - E_Na) - g_K * nc*nc*nc*nc * (V - E_K) - g_L * (V - E_L)) / C_m
    out[1] = alpha_m * (1.0 - mc) - beta_m * mc
    out[2] = alpha_h * (1.0 - hc) - beta_h * hc
    out[3] = alpha_n * (1.0 - nc) - beta_n * nc

@njit(cache=True, fastmath=True)
def rk45_integrate(t0, t1, y0, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, rtol, atol):
    b21 = 1.0/5.0
    b31 = 3.0/40.0; b32 = 9.0/40.0
    b41 = 44.0/45.0; b42 = -56.0/15.0; b43 = 32.0/9.0
    b51 = 19372.0/6561.0; b52 = -25360.0/2187.0; b53 = 64448.0/6561.0; b54 = -212.0/729.0
    b61 = 9017.0/3168.0; b62 = -355.0/33.0; b63 = 46732.0/5247.0; b64 = 49.0/176.0; b65 = -5103.0/18656.0
    
    c1 = 35.0/384.0; c3 = 500.0/1113.0; c4 = 125.0/192.0; c5 = -2187.0/6784.0; c6 = 11.0/84.0
    e1 = 71.0/57600.0; e3 = -71.0/16695.0; e4 = 71.0/1920.0; e5 = -17253.0/339200.0; e6 = 22.0/525.0; e7 = -1.0/40.0
    
    t = t0
    y = y0.copy()
    
    f1 = np.empty(4)
    ytmp = np.empty(4)
    k1 = np.empty(4)
    k2 = np.empty(4)
    k3 = np.empty(4)
    k4 = np.empty(4)
    k5 = np.empty(4)
    k6 = np.empty(4)
    k7 = np.empty(4)
    
    hh_rhs(y, f1, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    s0 = atol + rtol * abs(y[0])
    s1 = atol + rtol * abs(y[1])
    s2 = atol + rtol * abs(y[2])
    s3 = atol + rtol * abs(y[3])
    d0 = np.sqrt(((y[0]/s0)**2 + (y[1]/s1)**2 + (y[2]/s2)**2 + (y[3]/s3)**2) * 0.25)
    d1v = np.sqrt(((f1[0]/s0)**2 + (f1[1]/s1)**2 + (f1[2]/s2)**2 + (f1[3]/s3)**2) * 0.25)
    
    if d0 < 1e-5 or d1v < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1v
    
    h = min(h0, t1 - t0)
    fsal_valid = True
    
    for step in range(2000000):
        if t >= t1 - 1e-14:
            break
        h = min(h, t1 - t)
        if h < 1e-15:
            break
        
        if not fsal_valid:
            hh_rhs(y, f1, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        
        for i in range(4):
            k1[i] = h * f1[i]
        
        for i in range(4):
            ytmp[i] = y[i] + b21*k1[i]
        hh_rhs(ytmp, k2, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        for i in range(4):
            k2[i] *= h
        
        for i in range(4):
            ytmp[i] = y[i] + b31*k1[i] + b32*k2[i]
        hh_rhs(ytmp, k3, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        for i in range(4):
            k3[i] *= h
        
        for i in range(4):
            ytmp[i] = y[i] + b41*k1[i] + b42*k2[i] + b43*k3[i]
        hh_rhs(ytmp, k4, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        for i in range(4):
            k4[i] *= h
        
        for i in range(4):
            ytmp[i] = y[i] + b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]
        hh_rhs(ytmp, k5, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        for i in range(4):
            k5[i] *= h
        
        for i in range(4):
            ytmp[i] = y[i] + b61*k1[i] + b62*k2[i] + b63*k3[i] + b64*k4[i] + b65*k5[i]
        hh_rhs(ytmp, k6, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        for i in range(4):
            k6[i] *= h
        
        yn0 = y[0] + c1*k1[0] + c3*k3[0] + c4*k4[0] + c5*k5[0] + c6*k6[0]
        yn1 = y[1] + c1*k1[1] + c3*k3[1] + c4*k4[1] + c5*k5[1] + c6*k6[1]
        yn2 = y[2] + c1*k1[2] + c3*k3[2] + c4*k4[2] + c5*k5[2] + c6*k6[2]
        yn3 = y[3] + c1*k1[3] + c3*k3[3] + c4*k4[3] + c5*k5[3] + c6*k6[3]
        
        ytmp[0] = yn0; ytmp[1] = yn1; ytmp[2] = yn2; ytmp[3] = yn3
        hh_rhs(ytmp, k7, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        
        ev0 = e1*k1[0] + e3*k3[0] + e4*k4[0] + e5*k5[0] + e6*k6[0] + e7*h*k7[0]
        ev1 = e1*k1[1] + e3*k3[1] + e4*k4[1] + e5*k5[1] + e6*k6[1] + e7*h*k7[1]
        ev2 = e1*k1[2] + e3*k3[2] + e4*k4[2] + e5*k5[2] + e6*k6[2] + e7*h*k7[2]
        ev3 = e1*k1[3] + e3*k3[3] + e4*k4[3] + e5*k5[3] + e6*k6[3] + e7*h*k7[3]
        
        sc0 = atol + rtol * max(abs(y[0]), abs(yn0))
        sc1 = atol + rtol * max(abs(y[1]), abs(yn1))
        sc2 = atol + rtol * max(abs(y[2]), abs(yn2))
        sc3 = atol + rtol * max(abs(y[3]), abs(yn3))
        
        err = np.sqrt(((ev0/sc0)**2 + (ev1/sc1)**2 + (ev2/sc2)**2 + (ev3/sc3)**2) * 0.25)
        
        if err <= 1.0:
            t = t + h
            y[0] = yn0; y[1] = yn1; y[2] = yn2; y[3] = yn3
            f1[0] = k7[0]; f1[1] = k7[1]; f1[2] = k7[2]; f1[3] = k7[3]
            fsal_valid = True
            
            if err < 1e-10:
                factor = 5.0
            else:
                factor = min(5.0, max(0.2, 0.9 * err**(-0.2)))
            h = h * factor
        else:
            fsal_valid = False
            factor = max(0.2, 0.9 * err**(-0.2))
            h = h * factor
    
    return y

_dummy_y = np.array([-65.0, 0.053, 0.596, 0.318])
_dummy_out = np.empty(4)
hh_rhs(_dummy_y, _dummy_out, 1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 10.0)
_ = rk45_integrate(0.0, 0.1, _dummy_y, 1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 10.0, 1e-8, 1e-8)

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        
        C_m = float(params["C_m"])
        g_Na = float(params["g_Na"])
        g_K = float(params["g_K"])
        g_L = float(params["g_L"])
        E_Na = float(params["E_Na"])
        E_K = float(params["E_K"])
        E_L = float(params["E_L"])
        I_app = float(params["I_app"])
        
        result = rk45_integrate(t0, t1, y0, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, 1e-8, 1e-8)
        
        return result.tolist()