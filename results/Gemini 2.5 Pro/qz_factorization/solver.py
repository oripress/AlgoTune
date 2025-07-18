import numpy as np
from scipy.linalg import get_lapack_funcs
from typing import Any

# The LAPACK gges routine requires a 'select' function argument.
# Since we don't need to sort the eigenvalues, a dummy function is sufficient.
_select_dummy = lambda alphar, alphai, beta: True

class Solver:
    def __init__(self):
        """
        Initializes the solver by pre-loading the LAPACK function and a cache
        for optimal workspace sizes.
        """
        # Pre-load the double-precision 'dgges' function to avoid lookups in the solve loop.
        dummy_a = np.empty((0, 0), dtype=np.float64)
        self.gges = get_lapack_funcs('gges', (dummy_a,))
        # Cache to store the optimal lwork size for each matrix dimension encountered.
        self._lwork_cache = {}

    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the QZ factorization using a cached optimal workspace size to
        amortize the cost of the LAPACK workspace query.
        """
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        
        n = A.shape[0]

        # Get the optimal lwork size. Use a try/except block for fast cache access.
        try:
            lwork = self._lwork_cache[n]
        except KeyError:
            # Cache miss: query LAPACK for the optimal lwork size.
            # This call is expensive but only happens once per unique matrix size.
            # The query must not overwrite the input arrays.
            res = self.gges(_select_dummy, A, B, lwork=-1, overwrite_a=0, overwrite_b=0)
            lwork = int(res[-2][0].real)
            self._lwork_cache[n] = lwork

        # Call the pre-loaded LAPACK routine with the optimal workspace size.
        # Now we allow overwriting the input arrays for maximum performance.
        AA, BB, sdim, alphar, alphai, beta, Q, Z, work, info = self.gges(
            _select_dummy, A, B, lwork=lwork, overwrite_a=1, overwrite_b=1
        )
        
        # Convert the resulting numpy arrays back to lists for the required output format.
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }