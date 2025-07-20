from typing import Any
import numpy as np
# Switch from the high-level 'solve_ivp' to the lower-level 'ode' interface
# to reduce Python overhead and gain more direct access to the Fortran solver.
from scipy.integrate import ode
import numba

# The Numba-JIT compiled functions are the core of the performance.
# These versions include micro-optimizations like pre-allocating output arrays
# and manual common subexpression elimination, which proved effective.

@numba.jit(nopython=True)
def seirs(t, y, beta, sigma, gamma, omega):
    """
    SEIRS model ODEs, micro-optimized for Numba.
    """
    S, E, I, R = y[0], y[1], y[2], y[3]
    
    out = np.empty(4)
    
    beta_S_I = beta * S * I
    sigma_E = sigma * E
    gamma_I = gamma * I
    omega_R = omega * R
    
    out[0] = -beta_S_I + omega_R
    out[1] =  beta_S_I - sigma_E
    out[2] =  sigma_E - gamma_I
    out[3] =  gamma_I - omega_R
    
    return out

@numba.jit(nopython=True)
def seirs_jac(t, y, beta, sigma, gamma, omega):
    """
    Jacobian of the SEIRS model, micro-optimized for Numba.
    """
    S, E, I, R = y[0], y[1], y[2], y[3]
    
    jac = np.empty((4, 4))
    
    beta_I = beta * I
    beta_S = beta * S
    
    jac[0, 0] = -beta_I; jac[0, 1] = 0.0;   jac[0, 2] = -beta_S; jac[0, 3] = omega
    jac[1, 0] = beta_I;  jac[1, 1] = -sigma; jac[1, 2] = beta_S;  jac[1, 3] = 0.0
    jac[2, 0] = 0.0;     jac[2, 1] = sigma;  jac[2, 2] = -gamma;  jac[2, 3] = 0.0
    jac[3, 0] = 0.0;     jac[3, 1] = 0.0;    jac[3, 2] = gamma;   jac[3, 3] = -omega
    
    return jac

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the SEIRS model using the low-overhead `scipy.integrate.ode`
        interface. Parameter handling is managed by closures to ensure
        robust compatibility with the Numba-JIT functions.
        """
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]

        # Define wrapper functions (closures) that have the correct (t, y)
        # signature for the `ode` object. These wrappers call the fast,
        # Numba-compiled core functions with the necessary model parameters.
        def rhs(t, y):
            return seirs(t, y, beta, sigma, gamma, omega)

        def jac(t, y):
            return seirs_jac(t, y, beta, sigma, gamma, omega)

        # 1. Instantiate the ODE solver object, passing the wrapper functions.
        solver = ode(rhs, jac)

        # 2. Set the integrator to 'lsoda' and configure its parameters.
        # The default nsteps=500 can be too low for stiff problems. Increasing
        # it gives the solver more steps to find a solution.
        solver.set_integrator('lsoda', rtol=1e-7, atol=1e-9, nsteps=5000)

        # 3. Set the initial conditions.
        solver.set_initial_value(y0, t0)

        # 4. Integrate the system directly to the final time t1.
        final_y = solver.integrate(t1)

        if not solver.successful():
            raise RuntimeError("ODE solver failed to integrate successfully.")
            
        return final_y.tolist()