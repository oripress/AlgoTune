import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

jax.config.update("jax_enable_x64", True)

class Solver:
    def __init__(self):
        # Pre-compile the entire solver for maximum performance
        self._compiled_solver = jax.jit(self._solve_ode)
    
    def solve(self, problem, **kwargs):
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = jnp.array(problem["y0"], dtype=jnp.float64)
        params = problem["params"]
        args = (
            jnp.array(params["beta"], dtype=jnp.float64),
            jnp.array(params["sigma"], dtype=jnp.float64),
            jnp.array(params["gamma"], dtype=jnp.float64),
            jnp.array(params["omega"], dtype=jnp.float64)
        )
        
        result = self._compiled_solver(t0, t1, y0, args)
        return result.tolist()
    
    def _solve_ode(self, t0, t1, y0, args):
        # Solve the ODE in one step without intermediate arrays
        solution = odeint(
            self._vector_field,
            y0,
            jnp.array([t0, t1]),
            *args,
            rtol=1e-10,
            atol=1e-10,
            mxstep=100000,
            hmax=5.0
        )
        return solution[-1]
    
    @staticmethod
    @jax.jit
    def _vector_field(y, t, beta, sigma, gamma, omega):
        S, E, I, R = y
        dSdt = -beta * S * I + omega * R
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I - omega * R
        return jnp.array([dSdt, dEdt, dIdt, dRdt])