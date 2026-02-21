import numpy as np
from typing import Any
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, SaveAt, PIDController

jax.config.update("jax_enable_x64", True)

def rober(t, y, args):
    k1, k2, k3 = args
    y1, y2, y3 = y
    return jnp.array([
        -k1 * y1 + k3 * y2 * y3,
        k1 * y1 - k2 * y2**2 - k3 * y2 * y3,
        k2 * y2**2
    ])

@jax.jit
def solve_diffrax(y1, y2, y3, t0, t1, k1, k2, k3):
    y0 = jnp.array([y1, y2, y3], dtype=jnp.float64)
    k = jnp.array([k1, k2, k3], dtype=jnp.float64)
    term = ODETerm(rober)
    solver = Kvaerno5()
    stepsize_controller = PIDController(rtol=1e-11, atol=1e-9)
    
    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-4,
        y0=y0,
        args=k,
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(t1=True),
        max_steps=1000000
    )
    return sol.ys[0]

class Solver:
    def __init__(self):
        # Trigger JIT compilation
        solve_diffrax(1.0, 0.0, 0.0, 0.0, 1.0, 0.04, 3e7, 1e4)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = problem["y0"]
        k = problem["k"]
        
        ys = solve_diffrax(
            float(y0[0]), float(y0[1]), float(y0[2]),
            float(problem["t0"]), float(problem["t1"]),
            float(k[0]), float(k[1]), float(k[2])
        )
        return np.array(ys).tolist()
        ys = solve_diffrax(y0, t0, t1, k)
        return np.array(ys).tolist()