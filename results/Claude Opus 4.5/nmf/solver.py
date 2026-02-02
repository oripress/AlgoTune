from typing import Any
import numpy as np
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, list[list[float]]]:
        X = np.array(problem["X"], dtype=np.float64)
        n_components = problem["n_components"]
        
        try:
            # Use sklearn NMF with the same init as reference but faster solver
            model = NMF(
                n_components=n_components,
                init="random",   # Same as reference
                solver="cd",     # Coordinate descent is usually faster
                max_iter=200,
                tol=1e-4,
                random_state=0
            )
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception as e:
            n, d = X.shape
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}