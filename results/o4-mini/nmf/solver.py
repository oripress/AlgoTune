import numpy as np
from sklearn.decomposition import non_negative_factorization

class Solver:
    def solve(self, problem, **kwargs):
        """
        Fast NMF: direct non_negative_factorization with CD solver,
        init random, tol 1e-4, max_iter 200, random_state 0.
        """
        X = np.array(problem.get("X", []), dtype=float)
        k = int(problem.get("n_components", 0))
        try:
            W, H, _ = non_negative_factorization(
                X,
                n_components=k,
                init='random',
                solver='cd',
                tol=1e-4,
                max_iter=200,
                random_state=0,
            )
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            # fallback to zeros
            m, n = X.shape
            W = np.zeros((m, k), dtype=float)
            H = np.zeros((k, n), dtype=float)
            return {"W": W.tolist(), "H": H.tolist()}