import numpy as np
from scipy.integrate import solve_ivp

# Global rate constants (set at runtime)
_k1 = 0.0
_k2 = 0.0
_k3 = 0.0

# Preallocated buffers for RHS and Jacobian
_f_buf = np.zeros(3, dtype=float)
_J_buf = np.zeros((3, 3), dtype=float)

def _f(t, y):
    y1, y2, y3 = y
    _f_buf[0] = -_k1 * y1 + _k3 * y2 * y3
    _f_buf[1] =  _k1 * y1 - _k2 * y2 * y2 - _k3 * y2 * y3
    _f_buf[2] =  _k2 * y2 * y2
    return _f_buf

def _jac(t, y):
    y1, y2, y3 = y
    _J_buf[0, 0] = -_k1
    _J_buf[0, 1] =  _k3 * y3
    _J_buf[0, 2] =  _k3 * y2
    _J_buf[1, 0] =  _k1
    _J_buf[1, 1] = -2.0 * _k2 * y2 - _k3 * y3
    _J_buf[1, 2] = -_k3 * y2
    _J_buf[2, 0] =  0.0
    _J_buf[2, 1] =  2.0 * _k2 * y2
    _J_buf[2, 2] =  0.0
    return _J_buf

class Solver:
    def solve(self, problem, **kwargs):
        global _k1, _k2, _k3
        # Unpack and convert inputs
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        _k1, _k2, _k3 = problem["k"]

        # Solve stiff ODE with LSODA (ODEPACK) and analytic Jacobian
        sol = solve_ivp(
            _f,
            (t0, t1),
            y0,
            method="LSODA",
            jac=_jac,
            rtol=1e-11,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        # Return final state
        return sol.y[:, -1].tolist()
        y_end = integrator.integrate(t1)
        if not integrator.successful():
            raise RuntimeError("ODE integration failed")
        return y_end.tolist()