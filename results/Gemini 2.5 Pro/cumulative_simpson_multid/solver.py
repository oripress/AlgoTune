import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Any

@jax.jit
def jax_cumulative_simpson(y: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    JAX-based implementation of cumulative Simpson's rule, mimicking SciPy's algorithm.
    The algorithm first computes a cumulative trapezoidal integral, then corrects
    the values at points that are an even number of intervals from the start.
    """
    N = y.shape[-1]

    # For N < 2, the result is all zeros. jnp.zeros_like is a safe default.
    if N < 2:
        return jnp.zeros_like(y)

    # Step 1: Compute the N-point cumulative trapezoidal integral.
    # This provides the correct values for all odd-indexed points and a base for even ones.
    interval_areas = (y[..., :-1] + y[..., 1:]) * dx / 2.0
    # Use jnp.pad to prepend the initial zero for an N-point result.
    cumtrapz_res = jnp.pad(jnp.cumsum(interval_areas, axis=-1),
                           [(0, 0)] * (y.ndim - 1) + [(1, 0)])

    # Initialize the result with the trapezoidal values.
    res = cumtrapz_res

    # Step 2: Correct the values at even-indexed points (i = 2, 4, 6, ...).
    # The SciPy algorithm updates these points based on the values from the
    # original trapezoidal integral, not the intermediate results of the correction.
    def correction_loop_body(i, current_res):
        # This loop runs for i = 2, 3, 4, ..., N-1.
        # We only perform an update if i is even.
        def perform_update(r):
            # Formula: res[i] = res[i-2] + integral over [i-2, i] with Simpson's rule.
            # We use cumtrapz_res[i-2] as the base, which is the critical fix.
            simpson_term = (y[..., i-2] + 4*y[..., i-1] + y[..., i]) * dx / 3.0
            return r.at[..., i].set(cumtrapz_res[..., i-2] + simpson_term)
        
        # Use lax.cond to apply the update only for even i.
        new_res = jax.lax.cond(
            i % 2 == 0,
            perform_update,
            lambda r: r,  # Identity function for odd i
            current_res
        )
        return new_res

    # The loop starts at i=2. It will not run if N < 3, which is correct.
    res = jax.lax.fori_loop(2, N, loop_body, res)

    return res

class Solver:
    """
    A solver for the cumulative Simpson's rule problem, optimized with JAX.
    """
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Calculates the cumulative Simpson's rule integral on a 3D array.
        """
        y2 = jnp.asarray(problem["y2"])
        dx = problem["dx"]
        
        # The JIT'd function is robust to empty and small inputs.
        # Compute the standard N-point cumulative integral.
        full_result = jax_cumulative_simpson(y2, dx)
        
        # The reference solution (initial=None) has N-1 elements.
        # We achieve this by slicing our N-point result (which has res[0]=0).
        # JAX handles slicing of empty axes gracefully.
        final_result = full_result[..., 1:]

        return np.asarray(final_result)