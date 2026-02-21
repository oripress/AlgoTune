import numpy as np
import sklearn.linear_model._cd_fast as cd_fast
from sklearn.utils import check_random_state

class Solver:
    def __init__(self):
        self.rng = check_random_state(0)

    def solve(self, problem: dict, **kwargs) -> list:
        X = np.array(problem["X"], dtype=np.float64, order='F')
        y = np.array(problem["y"], dtype=np.float64)
        n_samples, n_features = X.shape
        
        w = np.zeros(n_features, dtype=np.float64)
        alpha = 0.1 * n_samples
        
        cd_fast.enet_coordinate_descent(
            w, alpha, 0.0, X, y, 1000, 1e-4, self.rng, random=0, positive=0, do_screening=1
        )
        
        return w.tolist()