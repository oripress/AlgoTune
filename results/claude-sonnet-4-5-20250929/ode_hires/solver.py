import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, SaveAt, PIDController

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

def hires_jax(t, y, args):
    """JAX-compiled HIRES ODE system."""
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = args
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    
    # Compute intermediate products once
    c10_y6_y8 = c10 * y6 * y8
    
    f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
    f2 = c1 * y1 - c5 * y2
    f3 = -c6 * y3 + c2 * y4 + c7 * y5
    f4 = c3 * y2 + c1 * y3 - c8 * y4
    f5 = -c9 * y5 + c2 * (y6 + y7)
    f6 = -c10_y6_y8 + c11 * (y4 + y7) + c1 * y5 - c2 * y6
    f7 = c10_y6_y8 - c12 * y7
    f8 = -c10_y6_y8 + c12 * y7
    
    return jnp.array([f1, f2, f3, f4, f5, f6, f7, f8])

@jax.jit
@jax.jit
def solve_ode_jax(y0, t0, t1, constants):
    """JIT-compiled ODE solver."""
    term = ODETerm(hires_jax)
    # Balance speed and accuracy - validation uses rtol=1e-5
    stepsize_controller = PIDController(rtol=1e-7, atol=1e-8)
    
    solution = diffeqsolve(
        term,
        solver=Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=constants,
        saveat=SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=50000
    )
    
    return solution.ys[0]
    
    return solution.ys[0]
    
    return solution.ys[0]

class Solver:
    def __init__(self):
        """Pre-compile the JAX function."""
        # Warm up JAX compilation
        dummy_y = jnp.ones(8)
        dummy_c = jnp.ones(12)
        _ = solve_ode_jax(dummy_y, 0.0, 1.0, dummy_c)
    
    def solve(self, problem, **kwargs):
        """Solve the HIRES ODE system using diffrax."""
        y0 = jnp.array(problem["y0"], dtype=jnp.float64)
        t0 = jnp.float64(problem["t0"])
        t1 = jnp.float64(problem["t1"])
        constants = jnp.array(problem["constants"], dtype=jnp.float64)
        
        # Call JIT-compiled solver
        result = solve_ode_jax(y0, t0, t1, constants)
        
        # Return final state as list
        return result.tolist()