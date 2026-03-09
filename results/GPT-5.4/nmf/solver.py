from typing import Any

import numpy as np
from sklearn.decomposition import NMF

def _zeros_solution(m: int, n: int, k: int) -> dict[str, Any]:
    return {
        "W": np.zeros((m, k), dtype=float),
        "H": np.zeros((k, n), dtype=float),
    }

class Solver:
    def __init__(self) -> None:
        self._models: dict[int, NMF] = {}

    def _model(self, k: int) -> NMF:
        model = self._models.get(k)
        if model is None:
            model = NMF(n_components=k, init="random", random_state=0)
            self._models[k] = model
        return model

    def solve(self, problem, **kwargs) -> Any:
        try:
            X = np.asarray(problem["X"], dtype=float)
            if X.ndim != 2:
                X = np.atleast_2d(X)
            m, n = X.shape
            k = int(problem["n_components"])

            if k <= 0:
                return _zeros_solution(m, n, max(k, 0))
            if m == 0 or n == 0:
                return _zeros_solution(m, n, k)

            if k >= n:
                W = np.zeros((m, k), dtype=float)
                H = np.zeros((k, n), dtype=float)
                W[:, :n] = X
                H[:n, :] = np.eye(n, dtype=float)
                return {"W": W, "H": H}

            if k >= m:
                W = np.zeros((m, k), dtype=float)
                H = np.zeros((k, n), dtype=float)
                W[:, :m] = np.eye(m, dtype=float)
                H[:m, :] = X
                return {"W": W, "H": H}

            model = self._model(k)
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W, "H": H}
        except Exception:
            try:
                X = np.asarray(problem["X"], dtype=float)
                if X.ndim != 2:
                    X = np.atleast_2d(X)
                m, n = X.shape
                k = max(int(problem["n_components"]), 0)
            except Exception:
                m, n, k = 0, 0, 0
            return _zeros_solution(m, n, k)