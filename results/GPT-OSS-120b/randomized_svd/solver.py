import numpy as np
from typing import Any, Dict, List
from sklearn.utils.extmath import randomized_svd

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute an approximate SVD using a fast randomized algorithm.
        Returns the top `n_components` singular vectors/values.
        """
        # Convert matrix to NumPy array (float32 for speed)
        A = np.asarray(problem["matrix"], dtype=np.float32)
        k = int(problem["n_components"])
        n, m = A.shape

        # Small/medium dense matrices: full SVD is faster
        if max(n, m) <= 200:
            U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
            U = U_full[:, :k]
            S = s_full[:k]
            V = Vt_full[:k, :].T
        else:
            # Large matrices: randomized SVD with zero power iterations (fastest)
            U, S, Vt = randomized_svd(
                A,
                n_components=k,
                n_iter=0,
                random_state=42,
            )
            V = Vt.T
        return {"U": U, "S": S, "V": V}