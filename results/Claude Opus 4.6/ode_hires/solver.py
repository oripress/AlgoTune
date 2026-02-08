import numpy as np
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, SaveAt, PIDController
from functools import partial

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@jax.jit
def _solve_jax(y0, t0, t1, constants):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants[0], constants[1], constants[2], constants[3], constants[4], constants[5], constants[6], constants[7], constants[8], constants[9], constants[10], constants[11]
    
    def rhs(t, y, args):
        y1, y2, y3, y4, y5, y6, y7, y8 = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
        f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
        f2 = c1 * y1 - c5 * y2
        f3 = -c6 * y3 + c2 * y4 + c7 * y5
        f4 = c3 * y2 + c1 * y3 - c8 * y4
        f5 = -c9 * y5 + c2 * y6 + c2 * y7
        f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
        f7 = c10 * y6 * y8 - c12 * y7
        f8 = -c10 * y6 * y8 + c12 * y7
        return jnp.array([f1, f2, f3, f4, f5, f6, f7, f8])
    
    term = ODETerm(rhs)
    solver = Kvaerno5()
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)
    saveat = SaveAt(t1=True)
    
    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1000000,
    )
    return sol.ys[0]

# Warm up JIT
_dummy_y0 = jnp.zeros(8, dtype=jnp.float64)
_dummy_c = jnp.zeros(12, dtype=jnp.float64)
_result = _solve_jax(_dummy_y0, 0.0, 1.0, _dummy_c)
_result.block_until_ready()

class Solver:
    def solve(self, problem, **kwargs):
        y0 = jnp.array(problem["y0"], dtype=jnp.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        c = jnp.array(problem["constants"], dtype=jnp.float64)
        
        result = _solve_jax(y0, t0, t1, c)
        return np.array(result).tolist()