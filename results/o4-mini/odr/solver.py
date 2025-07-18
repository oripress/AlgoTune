import numpy as np
import scipy.odr as odr

class Solver:
    def solve(self, problem: dict = None, **kwargs) -> dict:
        """
        Fit weighted orthogonal distance regression using scipy.odr.
        """
        # Determine input source
        if isinstance(problem, dict):
            x = problem.get("x")
            y = problem.get("y")
            sx = problem.get("sx")
            sy = problem.get("sy")
        else:
            x = kwargs.get("x")
            y = kwargs.get("y")
            sx = kwargs.get("sx")
            sy = kwargs.get("sy")

        # Convert to arrays
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        sx_arr = np.asarray(sx, dtype=float)
        sy_arr = np.asarray(sy, dtype=float)

        # Set up ODR problem
        data = odr.RealData(x_arr, y=y_arr, sx=sx_arr, sy=sy_arr)
        model = odr.Model(lambda B, x: B[0] * x + B[1])

        # Initial guess for slope and intercept
        odr_inst = odr.ODR(data, model, beta0=[0.0, 1.0])
        output = odr_inst.run()

        # Return slope and intercept
        return {"beta": output.beta.tolist()}