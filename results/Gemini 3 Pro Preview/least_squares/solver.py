import numpy as np
from scipy.optimize import leastsq
from numba import jit

@jit(nopython=True)
def _resid_exponential(p, x, y):
    a, b, c = p
    res = np.empty_like(y)
    for i in range(len(y)):
        val = b * x[i]
        if val > 50.0: val = 50.0
        elif val < -50.0: val = -50.0
        res[i] = y[i] - (a * np.exp(val) + c)
    return res

@jit(nopython=True)
def _jac_exponential(p, x, y):
    a, b, c = p
    n = len(x)
    jac = np.empty((n, 3))
    for i in range(n):
        val_raw = b * x[i]
        if val_raw > 50.0:
            val = 50.0
            dval_db = 0.0
        elif val_raw < -50.0:
            val = -50.0
            dval_db = 0.0
        else:
            val = val_raw
            dval_db = x[i]
            
        exp_val = np.exp(val)
        
        jac[i, 0] = -exp_val
        jac[i, 1] = -a * exp_val * dval_db
        jac[i, 2] = -1.0
    return jac

@jit(nopython=True)
def _resid_logarithmic(p, x, y):
    a, b, c, d = p
    res = np.empty_like(y)
    for i in range(len(y)):
        res[i] = y[i] - (a * np.log(b * x[i] + c) + d)
    return res

@jit(nopython=True)
def _jac_logarithmic(p, x, y):
    a, b, c, d = p
    n = len(x)
    jac = np.empty((n, 4))
    for i in range(n):
        arg = b * x[i] + c
        jac[i, 0] = -np.log(arg)
        jac[i, 1] = -a * x[i] / arg
        jac[i, 2] = -a / arg
        jac[i, 3] = -1.0
    return jac

@jit(nopython=True)
def _resid_sigmoid(p, x, y):
    a, b, c, d = p
    res = np.empty_like(y)
    for i in range(len(y)):
        val = -b * (x[i] - c)
        if val > 50.0: val = 50.0
        elif val < -50.0: val = -50.0
        res[i] = y[i] - (a / (1.0 + np.exp(val)) + d)
    return res

@jit(nopython=True)
def _jac_sigmoid(p, x, y):
    a, b, c, d = p
    n = len(x)
    jac = np.empty((n, 4))
    for i in range(n):
        val_raw = -b * (x[i] - c)
        if val_raw > 50.0:
            val = 50.0
            dval_db = 0.0
            dval_dc = 0.0
        elif val_raw < -50.0:
            val = -50.0
            dval_db = 0.0
            dval_dc = 0.0
        else:
            val = val_raw
            dval_db = -(x[i] - c)
            dval_dc = b
            
        exp_val = np.exp(val)
        denom = 1.0 + exp_val
        denom2 = denom * denom
        
        jac[i, 0] = -1.0 / denom
        jac[i, 1] = a * exp_val * dval_db / denom2
        jac[i, 2] = a * exp_val * dval_dc / denom2
        jac[i, 3] = -1.0
    return jac

@jit(nopython=True)
def _resid_sinusoidal(p, x, y):
    a, b, c, d = p
    res = np.empty_like(y)
    for i in range(len(y)):
        res[i] = y[i] - (a * np.sin(b * x[i] + c) + d)
    return res

@jit(nopython=True)
def _jac_sinusoidal(p, x, y):
    a, b, c, d = p
    n = len(x)
    jac = np.empty((n, 4))
    for i in range(n):
        arg = b * x[i] + c
        sin_val = np.sin(arg)
        cos_val = np.cos(arg)
        
        jac[i, 0] = -sin_val
        jac[i, 1] = -a * x[i] * cos_val
        jac[i, 2] = -a * cos_val
        jac[i, 3] = -1.0
    return jac

class Solver:
    def __init__(self):
        # Warmup JIT functions
        x = np.array([0.0, 1.0], dtype=np.float64)
        y = np.array([0.0, 1.0], dtype=np.float64)
        
        p_exp = np.array([1.0, 0.05, 0.0], dtype=np.float64)
        _resid_exponential(p_exp, x, y)
        _jac_exponential(p_exp, x, y)
        
        p_log = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        _resid_logarithmic(p_log, x, y)
        _jac_logarithmic(p_log, x, y)
        
        p_sig = np.array([3.0, 0.5, 0.0, 0.0], dtype=np.float64)
        _resid_sigmoid(p_sig, x, y)
        _jac_sigmoid(p_sig, x, y)
        
        p_sin = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float64)
        _resid_sinusoidal(p_sin, x, y)
        _jac_sinusoidal(p_sin, x, y)

    def solve(self, problem, **kwargs):
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        if model_type == "polynomial":
            degree = problem["degree"]
            params = np.polyfit(x_data, y_data, degree)
            return {"params": params.tolist()}

        if model_type == "exponential":
            guess = np.array([1.0, 0.05, 0.0])
            func = _resid_exponential
            jac = _jac_exponential
        elif model_type == "logarithmic":
            guess = np.array([1.0, 1.0, 1.0, 0.0])
            func = _resid_logarithmic
            jac = _jac_logarithmic
        elif model_type == "sigmoid":
            guess = np.array([3.0, 0.5, np.median(x_data), 0.0])
            func = _resid_sigmoid
            jac = _jac_sigmoid
        elif model_type == "sinusoidal":
            guess = np.array([2.0, 1.0, 0.0, 0.0])
            func = _resid_sinusoidal
            jac = _jac_sinusoidal
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        params_opt, _ = leastsq(func, guess, Dfun=jac, args=(x_data, y_data), maxfev=10000)
        return {"params": params_opt.tolist()}