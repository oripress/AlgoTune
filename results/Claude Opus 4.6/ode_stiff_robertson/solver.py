import numpy as np
from scipy.integrate import odeint

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k1, k2, k3 = float(problem["k"][0]), float(problem["k"][1]), float(problem["k"][2])
        
        def rhs(y, t):
            y1, y2, y3 = y[0], y[1], y[2]
            return [
                -k1 * y1 + k3 * y2 * y3,
                k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
                k2 * y2 * y2
            ]
        
        # col_deriv=True means we return transposed Jacobian (df_i/dy_j -> columns are equations)
        def jac_t(y, t):
            y2, y3 = y[1], y[2]
            return [
                [-k1, k1, 0.0],
                [k3 * y3, -2.0 * k2 * y2 - k3 * y3, 2.0 * k2 * y2],
                [k3 * y2, -k3 * y2, 0.0]
            ]
        
        result = odeint(
            rhs, y0, [t0, t1],
            Dfun=jac_t,
            col_deriv=True,
            rtol=1e-9,
            atol=1e-9,
            mxstep=100000,
            full_output=False
        )
        
        return result[-1].tolist()