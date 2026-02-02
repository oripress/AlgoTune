from typing import Any
import numpy as np
from scipy.integrate import ode
from numba import njit, cfunc
from numba.types import float64, CPointer, intc
import ctypes

# Pre-compile base functions
@njit(cache=True)
def _rober_core(y1, y2, y3, k1, k2, k3):
    f0 = -k1 * y1 + k3 * y2 * y3
    f1 = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
    f2 = k2 * y2**2
    return f0, f1, f2

@njit(cache=True)
def _jac_core(y1, y2, y3, k1, k2, k3):
    j00 = -k1
    j01 = k3 * y3
    j02 = k3 * y2
    j10 = k1
    j11 = -2 * k2 * y2 - k3 * y3
    j12 = -k3 * y2
    j20 = 0.0
    j21 = 2 * k2 * y2
    j22 = 0.0
    return j00, j01, j02, j10, j11, j12, j20, j21, j22

@njit(cache=True)
def rober_func(t, y, k1, k2, k3):
    f0, f1, f2 = _rober_core(y[0], y[1], y[2], k1, k2, k3)
    return np.array([f0, f1, f2])

@njit(cache=True)
def jac_func(t, y, k1, k2, k3):
    j00, j01, j02, j10, j11, j12, j20, j21, j22 = _jac_core(y[0], y[1], y[2], k1, k2, k3)
    return np.array([
        [j00, j01, j02],
        [j10, j11, j12],
        [j20, j21, j22]
    ])

# Warm up the functions
_dummy_y = np.array([1.0, 0.0, 0.0])
_dummy_t = 0.0
_ = rober_func(_dummy_t, _dummy_y, 0.04, 3e7, 1e4)
_ = jac_func(_dummy_t, _dummy_y, 0.04, 3e7, 1e4)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k = problem["k"]
        k1, k2, k3 = float(k[0]), float(k[1]), float(k[2])
        
        def rober_wrapper(t, y):
            return rober_func(t, y, k1, k2, k3)
        
        def jac_wrapper(t, y):
            return jac_func(t, y, k1, k2, k3)
        
        # Use ode interface with lsoda
        solver = ode(rober_wrapper, jac_wrapper)
        solver.set_integrator('lsoda', rtol=1e-11, atol=1e-9, nsteps=10000)
        solver.set_initial_value(y0, t0)
        
        result = solver.integrate(t1)
        
        if solver.successful():
            return result.tolist()
        else:
            raise RuntimeError("Solver failed")