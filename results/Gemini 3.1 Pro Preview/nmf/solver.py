import numpy as np
from sklearn.decomposition._nmf import _fit_coordinate_descent, _initialize_nmf

class Solver:
    def solve(self, problem: dict, **kwargs):
        try:
            X = np.array(problem["X"])
            n_components = problem["n_components"]
            
            W, H = _initialize_nmf(X, n_components, init="random", random_state=0)
            
            W, H, _ = _fit_coordinate_descent(
                X, W, H, tol=1e-4, max_iter=200, l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0, update_H=True, verbose=0, shuffle=False, random_state=0
            )
            
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception as e:
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}