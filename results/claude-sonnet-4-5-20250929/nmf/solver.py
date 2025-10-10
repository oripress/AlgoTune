import numpy as np
from typing import Any
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Non-Negative Matrix Factorization using sklearn with float32.
        """
        X = np.array(problem["X"], dtype=np.float32)
        n_components = problem["n_components"]
        
        model = NMF(
            n_components=n_components,
            init='random',
            random_state=0
        )
        
        W = model.fit_transform(X)
        H = model.components_
        
        return {"W": W.tolist(), "H": H.tolist()}
        return {"W": W.tolist(), "H": H.tolist()}