from typing import Any
import numpy as np
import sklearn

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        try:
            # use sklearn.decomposition.NMF to solve the task
            # Optimization: use float32 for faster computation and direct list conversion
            X = np.asarray(problem["X"], dtype=np.float32)
            model = sklearn.decomposition.NMF(
                n_components=problem["n_components"], 
                init="random", 
                random_state=0
            )
            W = model.fit_transform(X)
            H = model.components_
            # Direct conversion without intermediate variables
            return {"W": W.astype(np.float64).tolist(), "H": H.astype(np.float64).tolist()}
        except Exception as e:
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}  # return trivial answer