import jax
import jax.numpy as jnp
import diffrax
from diffrax import ODETerm, SaveAt, Tsit5, Kvaerno5, PIDController, diffeqsolve
from functools import partial

@partial(jax.jit, static_argnames=('mu',))
def solve_vdp(mu, y0, t0, t1):
    def vdp_rhs(t, y, args):
        x, v = y
        dx_dt = v
        dv_dt = mu * ((1 - x**2) * v - x)
        return jnp.array([dx_dt, dv_dt])
    
    # Choose solver based on stiffness
    solver = Kvaerno5() if mu > 100 else Tsit5()
    
    # Set step size parameters based on stiffness
    if mu > 10000:
        dt0 = 1e-6
        max_steps = int(1e6)
    elif mu > 1000:
        dt0 = 1e-5
        max_steps = int(1e5)
    elif mu > 100:
        dt0 = 1e-4
        max_steps = int(1e4)
    else:
        dt0 = 0.01
        max_steps = int(1e3)
    
    term = ODETerm(vdp_rhs)
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-9)
    
    solution = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=jnp.array(y0),
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(t1=True),
        max_steps=max_steps
    )
    return solution.ys[0]

class Solver:
    def __init__(self):
        # JAX will compile on first call automatically
        pass
    
    def solve(self, problem, **kwargs):
        mu = float(problem["mu"])
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        
        try:
            result = solve_vdp(mu, y0, t0, t1)
            return [float(result[0]), float(result[1])]
        except Exception as e:
            # Fallback to optimized SciPy implementation
            import numpy as np
            from scipy.integrate import solve_ivp
            
            sol = solve_ivp(
                lambda t, y: [y[1], mu * ((1 - y[0]**2) * y[1] - y[0])],
                (t0, t1),
                y0,
                method='Radau',
                rtol=1e-8,
                atol=1e-9,
                dense_output=False
            )
            return sol.y[:, -1].tolist()