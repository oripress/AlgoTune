import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno5, PIDController
import numpy as np

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

class Solver:
    def __init__(self):
        # Define the vector field
        def vector_field(t, y, args):
            k1, k2, k3 = args
            y1, y2, y3 = y
            dy1 = -k1 * y1 + k3 * y2 * y3
            dy2 = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
            dy3 = k2 * y2**2
            return jnp.stack([dy1, dy2, dy3])

        self.term = ODETerm(vector_field)
        # Kvaerno5 is an L-stable method suitable for stiff problems
        self.solver = Kvaerno5()
        self.stepsize_controller = PIDController(rtol=1e-6, atol=1e-9)

        # JIT compile the solve function
        def solve_internal(t0, t1, y0, k):
            sol = diffeqsolve(
                self.term,
                self.solver,
                t0=t0,
                t1=t1,
                dt0=1e-6,
                y0=y0,
                args=k,
                stepsize_controller=self.stepsize_controller,
                max_steps=100000,
                throw=False
            )
            return sol.ys[-1]

        self._solve_jit = jax.jit(solve_internal)
        
        # Trigger compilation with dummy data
        dummy_y0 = jnp.array([1.0, 0.0, 0.0])
        dummy_k = jnp.array([0.04, 3e7, 1e4])
        self._solve_jit(0.0, 1.0, dummy_y0, dummy_k)

    def solve(self, problem: dict, **kwargs):
        y0 = jnp.array(problem["y0"])
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k = jnp.array(problem["k"])

        y_final = self._solve_jit(t0, t1, y0, k)
        
        return np.array(y_final).tolist()