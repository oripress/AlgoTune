from typing import Any
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, SaveAt, PIDController

# Enable 64-bit precision for accuracy, which is crucial for this problem.
jax.config.update("jax_enable_x64", True)

# The ODE vector field and its Jacobian remain the same.
# Jitting them individually is still good practice.
@jax.jit
def hires_ode(t, y, args):
    c = args
    return jnp.array([
        -c[0] * y[0] + c[1] * y[1] + c[2] * y[2] + c[3],
        c[0] * y[0] - c[4] * y[1],
        -c[5] * y[2] + c[1] * y[3] + c[6] * y[4],
        c[2] * y[1] + c[0] * y[2] - c[7] * y[3],
        -c[8] * y[4] + c[1] * y[5] + c[1] * y[6],
        -c[9] * y[5] * y[7] + c[10] * y[3] + c[0] * y[4] - c[1] * y[5] + c[10] * y[6],
        c[9] * y[5] * y[7] - c[11] * y[6],
        -c[9] * y[5] * y[7] + c[11] * y[6]
    ])

@jax.jit
def hires_jac(t, y, args):
    c = args
    return jnp.array([
        [-c[0], c[1], c[2], 0., 0., 0., 0., 0.],
        [c[0], -c[4], 0., 0., 0., 0., 0., 0.],
        [0., 0., -c[5], c[1], c[6], 0., 0., 0.],
        [0., c[2], c[0], -c[7], 0., 0., 0., 0.],
        [0., 0., 0., 0., -c[8], c[1], c[1], 0.],
        [0., 0., 0., c[10], c[0], -c[9]*y[7] - c[1], c[10], -c[9]*y[5]],
        [0., 0., 0., 0., 0., c[9]*y[7], -c[11], c[9]*y[5]],
        [0., 0., 0., 0., 0., -c[9]*y[7], c[11], -c[9]*y[5]]
    ])

# --- JIT-friendly Refactoring ---

# 1. Define solver configuration as global, static objects.
#    The JIT compiler can treat these as compile-time constants.
term = ODETerm(hires_ode)
object.__setattr__(term, 'vf_j', hires_jac)
solver = Kvaerno5()
stepsize_controller = PIDController(rtol=1e-7, atol=1e-9)
# Let diffrax determine the optimal initial step size automatically.
dt0 = None
# Increase max_steps for a larger safety margin.
max_steps = 16**6

# 2. Create the JIT-compiled "pure" solver function.
@jax.jit
def _solve_problem(t0, t1, y0, constants):
    saveat = SaveAt(ts=jnp.array([t1]))
    
    sol = diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=constants,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps
    )
    return sol.ys[0]

# 3. The Solver class remains a thin wrapper around the JIT-compiled function.
#    This minimizes Python overhead for each call to solve().
class Solver:
    def __init__(self):
        pass

    def solve(self, problem, **kwargs) -> Any:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = jnp.array(problem["y0"])
        constants = jnp.array(problem["constants"])

        # Call the pre-compiled JAX function.
        y_final = _solve_problem(t0, t1, y0, constants)
        
        # Convert back to a standard Python list for the output.
        return y_final.tolist()