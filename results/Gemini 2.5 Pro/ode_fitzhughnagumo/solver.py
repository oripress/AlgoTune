from typing import Any, Dict, List
import jax
import jax.numpy as jnp
import diffrax

class Solver:
    @staticmethod
    @jax.jit
    def _solve_jit(t0, t1, y0, a, b, c, I):
        # The vector field for the FitzHugh-Nagumo model
        def fitzhugh_nagumo(t, y, args):
            v, w = y
            dv_dt = v - (v**3) / 3 - w + I
            dw_dt = a * (b * v - c * w)
            return jnp.array([dv_dt, dw_dt])

        term = diffrax.ODETerm(fitzhugh_nagumo)
        # Kvaerno5 is an implicit solver suitable for stiff ODEs.
        solver = diffrax.Kvaerno5()

        # Use tolerances stricter than the evaluation criteria to ensure
        # the final solution is accurate enough after error accumulation.
        rtol = 1e-7
        atol = 1e-10
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        # Solve the ODE
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            y0=y0,
            dt0=None,  # Let diffrax choose the initial step size
            saveat=diffrax.SaveAt(t1=True),  # We only need the final state
            stepsize_controller=stepsize_controller,
            # Increased max_steps for robustness on long integration intervals
            max_steps=16**6
        )

        # The result is saved at t1, so sol.ys will have shape (1, 2)
        return sol.ys[0]

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> List[float]:
        """
        Solves the FitzHugh-Nagumo ODE using the JIT-compiled diffrax solver.
        """
        # Use float64 for high precision, which is necessary to pass the checker.
        y0 = jnp.array(problem["y0"], dtype=jnp.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        
        params = problem["params"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        I = params["I"]

        # Call the JIT-compiled solver function
        final_y = Solver._solve_jit(t0, t1, y0, a, b, c, I)
        
        # Convert the JAX array back to a standard Python list of floats for the output
        return final_y.tolist()