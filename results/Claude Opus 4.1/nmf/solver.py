import numpy as np
from sklearn.decomposition import NMF
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[list[float]]]:
        """Solve NMF problem using sklearn."""
        try:
            # Extract problem parameters
            X = np.array(problem["X"])
            n_components = problem["n_components"]
            
            # Use sklearn NMF
            model = NMF(
                n_components=n_components,
                init='random',
                random_state=0,
                max_iter=200  # Reduced iterations for speed
            )
            
            W = model.fit_transform(X)
            H = model.components_
            
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception as e:
            # Return zero matrices if error
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}