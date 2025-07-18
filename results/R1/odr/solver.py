import numpy as np
import scipy.odr as odr

def linear_model(B: np.ndarray, x: np.ndarray) -> np.ndarray:
    return B[0] * x + B[1]

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Convert inputs to numpy arrays with explicit typing
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sx = np.asarray(problem["sx"], dtype=np.float64)
        sy = np.asarray(problem["sy"], dtype=np.float64)
        
        # Setup ODR with reference parameters
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(linear_model)
        odr_instance = odr.ODR(data, model, beta0=[0.0, 1.0])
        
        # Run ODR
        output = odr_instance.run()
        return {"beta": output.beta.tolist()}