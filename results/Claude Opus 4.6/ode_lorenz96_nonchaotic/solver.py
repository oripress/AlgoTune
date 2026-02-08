import numpy as np

try:
    from lorenz96_solver import solve_dopri5 as solve_dopri5_cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

if not USE_CYTHON:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def lorenz96_rhs_inplace(dxdt, x, F, N):
        dxdt[0] = (x[1] - x[N-2]) * x[N-1] - x[0] + F
        dxdt[1] = (x[2] - x[N-1]) * x[0] - x[1] + F
        for i in range(2, N-1):
            dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        dxdt[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1] + F

    @njit(cache=True, fastmath=True)
    def solve_dopri5_numba(y0, t0, t1, F, rtol, atol):
        N = len(y0)
        y = y0.copy()
        t = t0
        
        a21 = 1.0/5.0
        a31 = 3.0/40.0; a32 = 9.0/40.0
        a41 = 44.0/45.0; a42 = -56.0/15.0; a43 = 32.0/9.0
        a51 = 19372.0/6561.0; a52 = -25360.0/2187.0; a53 = 64448.0/6561.0; a54 = -212.0/729.0
        a61 = 9017.0/3168.0; a62 = -355.0/33.0; a63 = 46732.0/5247.0; a64 = 49.0/176.0; a65 = -5103.0/18656.0
        a71 = 35.0/384.0; a73 = 500.0/1113.0; a74 = 125.0/192.0; a75 = -2187.0/6784.0; a76 = 11.0/84.0
        
        e1 = 71.0/57600.0; e3 = -71.0/16695.0; e4 = 71.0/1920.0
        e5 = -17253.0/339200.0; e6 = 22.0/525.0; e7 = -1.0/40.0
        
        k1 = np.empty(N)
        k2 = np.empty(N)
        k3 = np.empty(N)
        k4 = np.empty(N)
        k5 = np.empty(N)
        k6 = np.empty(N)
        k7 = np.empty(N)
        ytmp = np.empty(N)
        y_new = np.empty(N)
        
        lorenz96_rhs_inplace(k1, y0, F, N)
        d0 = 0.0
        d1 = 0.0
        for i in range(N):
            sc = atol + rtol * abs(y0[i])
            d0 += (y0[i] / sc) ** 2
            d1 += (k1[i] / sc) ** 2
        d0 = np.sqrt(d0 / N)
        d1 = np.sqrt(d1 / N)
        
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        
        for i in range(N):
            ytmp[i] = y0[i] + h0 * k1[i]
        lorenz96_rhs_inplace(k2, ytmp, F, N)
        d2 = 0.0
        for i in range(N):
            sc = atol + rtol * abs(y0[i])
            d2 += ((k2[i] - k1[i]) / sc) ** 2
        d2 = np.sqrt(d2 / N) / h0
        
        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** 0.2
        
        h = min(100.0 * h0, h1, t1 - t0)
        
        safety = 0.9
        min_factor = 0.2
        max_factor = 10.0
        beta1 = 0.7 / 5.0
        beta2 = 0.4 / 5.0
        prev_err_norm = 1.0
        
        for step in range(1000000):
            if t >= t1:
                break
            
            h = min(h, t1 - t)
            if h < 1e-14:
                break
            
            for i in range(N):
                ytmp[i] = y[i] + h * a21 * k1[i]
            lorenz96_rhs_inplace(k2, ytmp, F, N)
            
            for i in range(N):
                ytmp[i] = y[i] + h * (a31*k1[i] + a32*k2[i])
            lorenz96_rhs_inplace(k3, ytmp, F, N)
            
            for i in range(N):
                ytmp[i] = y[i] + h * (a41*k1[i] + a42*k2[i] + a43*k3[i])
            lorenz96_rhs_inplace(k4, ytmp, F, N)
            
            for i in range(N):
                ytmp[i] = y[i] + h * (a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i])
            lorenz96_rhs_inplace(k5, ytmp, F, N)
            
            for i in range(N):
                ytmp[i] = y[i] + h * (a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i] + a65*k5[i])
            lorenz96_rhs_inplace(k6, ytmp, F, N)
            
            for i in range(N):
                y_new[i] = y[i] + h * (a71*k1[i] + a73*k3[i] + a74*k4[i] + a75*k5[i] + a76*k6[i])
            
            lorenz96_rhs_inplace(k7, y_new, F, N)
            
            err_norm = 0.0
            for i in range(N):
                sc = atol + rtol * max(abs(y[i]), abs(y_new[i]))
                err_i = h * (e1*k1[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] + e6*k6[i] + e7*k7[i])
                err_norm += (err_i / sc) ** 2
            err_norm = np.sqrt(err_norm / N)
            
            if err_norm <= 1.0:
                t = t + h
                for i in range(N):
                    y[i] = y_new[i]
                    k1[i] = k7[i]
                
                if err_norm == 0.0:
                    factor = max_factor
                else:
                    factor = safety * err_norm**(-beta1) * prev_err_norm**beta2
                    factor = min(max_factor, max(min_factor, factor))
                prev_err_norm = max(err_norm, 1e-8)
                h = h * factor
            else:
                factor = max(min_factor, safety * err_norm**(-0.2))
                h = h * factor
        
        return y

    # Warm up
    for _n in [4, 7, 10, 20, 30, 40, 50]:
        _dummy_y0 = np.ones(_n) * 2.0
        solve_dopri5_numba(_dummy_y0, 0.0, 0.01, 2.0, 1e-8, 1e-8)


class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])
        
        if USE_CYTHON:
            result = solve_dopri5_cython(y0, t0, t1, F, 1e-8, 1e-8)
        else:
            result = solve_dopri5_numba(y0, t0, t1, F, 1e-8, 1e-8)
        return result.tolist()