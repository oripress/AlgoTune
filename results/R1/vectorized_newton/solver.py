import jax
import jax.numpy as jnp

class Solver:
    def __init__(self):
        pass

    def solve(self, problem, **kwargs):
        # Get fixed parameters
        try:
            a2 = kwargs['a2']
            a3 = kwargs['a3']
            a4 = kwargs['a4']
            a5 = kwargs['a5']
        except KeyError:
            try:
                n = len(problem['x0'])
            except:
                return {"roots": []}
            return {"roots": [float('nan')] * n}
        
        try:
            # Extract input arrays
            x0_arr = jnp.array(problem["x0"], dtype=jnp.float64)
            a0_arr = jnp.array(problem["a0"], dtype=jnp.float64)
            a1_arr = jnp.array(problem["a1"], dtype=jnp.float64)
            n = len(x0_arr)
            if len(a0_arr) != n or len(a1_arr) != n:
                return {"roots": [float("nan")] * n}
        except Exception:
            return {"roots": [float("nan")] * n}
        
        # Define the function and its derivative
        def func(x, a0, a1):
            u = (a0 + x * a3) / a5
            term1 = a2 * (jnp.exp(u) - 1)
            term2 = (a0 + x * a3) / a4
            return a1 - term1 - term2 - x

        def fprime(x, a0, a1):
            u = (a0 + x * a3) / a5
            return -a2 * jnp.exp(u) * (a3 / a5) - (a3 / a4) - 1
        
        # Vectorized Newton-Raphson implementation
        roots = x0_arr.copy()
        tol = 1e-8
        max_iter = 50
        
        for _ in range(max_iter):
            # Compute function value and derivative
            f_val = func(roots, a0_arr, a1_arr)
            df_val = fprime(roots, a0_arr, a1_arr)
            
            # Compute step, avoiding division by zero
            step = jnp.where(jnp.abs(df_val) > 1e-12, f_val / df_val, 0)
            roots_new = roots - step
            
            # Check convergence
            max_change = jnp.max(jnp.abs(step))
            if max_change < tol:
                roots = roots_new
                break
                
            roots = roots_new
        else:
            # Mark non-converged roots as NaN
            converged = jnp.abs(step) < tol
            roots = jnp.where(converged, roots, jnp.nan)
        
        return {"roots": roots.tolist()}