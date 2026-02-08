import numpy as np
from numpy.random import RandomState
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem, **kwargs):
        X = np.asarray(problem["X"], dtype=np.float64)
        n_components = problem["n_components"]
        m, n = X.shape

        # Use sklearn's coordinate descent solver which is faster
        try:
            model = NMF(
                n_components=n_components,
                init="random",
                random_state=0,
                max_iter=200,
                solver="cd",
            )
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            W = np.zeros((m, n_components), dtype=float).tolist()
            H = np.zeros((n_components, n), dtype=float).tolist()
            return {"W": W, "H": H}