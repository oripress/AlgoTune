import numpy as np
from scipy.optimize import leastsq
from typing import Any
from numba import njit

@njit("float64[:](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def r_exponential(p, x_data, y_data):
    a, b, c = p[0], p[1], p[2]
    res = np.empty(x_data.size, dtype=np.float64)
    for i in range(x_data.size):
        val = b * x_data[i]
        if val < -50.0:
            val = -50.0
        elif val > 50.0:
            val = 50.0
        res[i] = y_data[i] - (a * np.exp(val) + c)
    return res

@njit("float64[:](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def r_logarithmic(p, x_data, y_data):
    a, b, c, d = p[0], p[1], p[2], p[3]
    res = np.empty(x_data.size, dtype=np.float64)
    for i in range(x_data.size):
        res[i] = y_data[i] - (a * np.log(b * x_data[i] + c) + d)
    return res

@njit("float64[:](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def r_sigmoid(p, x_data, y_data):
    a, b, c, d = p[0], p[1], p[2], p[3]
    res = np.empty(x_data.size, dtype=np.float64)
    for i in range(x_data.size):
        val = -b * (x_data[i] - c)
        if val < -50.0:
            val = -50.0
        elif val > 50.0:
            val = 50.0
        res[i] = y_data[i] - (a / (1.0 + np.exp(val)) + d)
    return res

@njit("float64[:](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def r_sinusoidal(p, x_data, y_data):
    a, b, c, d = p[0], p[1], p[2], p[3]
    res = np.empty(x_data.size, dtype=np.float64)
    for i in range(x_data.size):
        res[i] = y_data[i] - (a * np.sin(b * x_data[i] + c) + d)
    return res
@njit("float64[:, :](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def jac_exponential(p, x_data, y_data):
    a, b, c = p[0], p[1], p[2]
    n = x_data.size
    jac = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        val = b * x_data[i]
        if val < -50.0:
            jac[i, 0] = -np.exp(-50.0)
            jac[i, 1] = 0.0
            jac[i, 2] = -1.0
        elif val > 50.0:
            jac[i, 0] = -np.exp(50.0)
            jac[i, 1] = 0.0
            jac[i, 2] = -1.0
        else:
            E = np.exp(val)
            jac[i, 0] = -E
            jac[i, 1] = -a * x_data[i] * E
            jac[i, 2] = -1.0
    return jac

@njit("float64[:, :](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def jac_logarithmic(p, x_data, y_data):
    a, b, c, d = p[0], p[1], p[2], p[3]
    n = x_data.size
    jac = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        denom = b * x_data[i] + c
        jac[i, 0] = -np.log(denom)
        jac[i, 1] = -a * x_data[i] / denom
        jac[i, 2] = -a / denom
        jac[i, 3] = -1.0
    return jac

@njit("float64[:, :](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def jac_sigmoid(p, x_data, y_data):
    a, b, c, d = p[0], p[1], p[2], p[3]
    n = x_data.size
    jac = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        val = -b * (x_data[i] - c)
        if val < -50.0:
            denom = 1.0 + np.exp(-50.0)
            jac[i, 0] = -1.0 / denom
            jac[i, 1] = 0.0
            jac[i, 2] = 0.0
            jac[i, 3] = -1.0
        elif val > 50.0:
            denom = 1.0 + np.exp(50.0)
            jac[i, 0] = -1.0 / denom
            jac[i, 1] = 0.0
            jac[i, 2] = 0.0
            jac[i, 3] = -1.0
        else:
            E = np.exp(val)
            denom = 1.0 + E
            denom2 = denom * denom
            jac[i, 0] = -1.0 / denom
            jac[i, 1] = -a * E * (x_data[i] - c) / denom2
            jac[i, 2] = a * b * E / denom2
            jac[i, 3] = -1.0
    return jac

@njit("float64[:, :](float64[:], float64[:], float64[:])", cache=True, fastmath=True)
def jac_sinusoidal(p, x_data, y_data):
    a, b, c, d = p[0], p[1], p[2], p[3]
    n = x_data.size
    jac = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        angle = b * x_data[i] + c
        cos_angle = np.cos(angle)
        jac[i, 0] = -np.sin(angle)
        jac[i, 1] = -a * x_data[i] * cos_angle
        jac[i, 2] = -a * cos_angle
        jac[i, 3] = -1.0
    return jac

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        x_data = np.asarray(problem["x_data"], dtype=np.float64)
        y_data = np.asarray(problem["y_data"], dtype=np.float64)
        model_type = problem["model_type"]
        if model_type == "polynomial":
            deg = problem["degree"]
            V = np.vander(x_data, deg + 1)
            params = np.linalg.solve(V.T @ V, V.T @ y_data)
            return {"params": params.tolist()}
            return {"params": params[::-1].tolist()}

        elif model_type == "exponential":
            guess = np.array([1.0, 0.05, 0.0], dtype=np.float64)
            params_opt, _ = leastsq(r_exponential, guess, args=(x_data, y_data), Dfun=jac_exponential, maxfev=10000, ftol=1e-5, xtol=1e-5)

        elif model_type == "logarithmic":
            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
            params_opt, _ = leastsq(r_logarithmic, guess, args=(x_data, y_data), Dfun=jac_logarithmic, maxfev=10000, ftol=1e-5, xtol=1e-5)

        elif model_type == "sigmoid":
            guess = np.array([3.0, 0.5, np.median(x_data), 0.0], dtype=np.float64)
            params_opt, _ = leastsq(r_sigmoid, guess, args=(x_data, y_data), Dfun=jac_sigmoid, maxfev=10000, ftol=1e-5, xtol=1e-5)

        elif model_type == "sinusoidal":
            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float64)
            params_opt, _ = leastsq(r_sinusoidal, guess, args=(x_data, y_data), Dfun=jac_sinusoidal, maxfev=10000, ftol=1e-5, xtol=1e-5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return {"params": params_opt.tolist()}