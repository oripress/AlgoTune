from typing import Any
import numpy as np
# Import the function to get low-level LAPACK functions
from scipy.linalg import get_lapack_funcs

class Solver:
    def __init__(self):
        """
        Initializes the solver by obtaining a direct handle to the
        LAPACK 'dgesv' function. This is done once to avoid the overhead
        of looking up the function in every call to solve().
        """
        # We request the 'gesv' function. SciPy's f2py wrapper will
        # automatically select the double-precision version ('dgesv')
        # based on the types of the dummy arrays provided.
        A_dummy = np.empty((1, 1), dtype=np.float64)
        b_dummy = np.empty((1,), dtype=np.float64)
        
        # get_lapack_funcs returns a tuple, so we unpack it.
        # self.dgesv is now a direct, callable wrapper around the
        # compiled LAPACK routine.
        self.dgesv, = get_lapack_funcs(('gesv',), (A_dummy, b_dummy))

    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves Ax=b by calling the LAPACK dgesv routine directly.
        """
        # Create arrays from the input.
        # A must be in Fortran (column-major) order for optimal performance
        # with LAPACK.
        A = np.array(problem["A"], dtype=np.float64, order='F')
        b = np.array(problem["b"], dtype=np.float64)
        
        # Call the pre-loaded dgesv function.
        # This bypasses the Python-level overhead of scipy.linalg.solve (e.g.,
        # type checks, matrix property analysis, and other dispatch logic).
        # Using overwrite_a=True and overwrite_b=True is critical, as it
        # allows the LAPACK routine to work directly on the memory of A and b,
        # avoiding internal data copies.
        lu, piv, x, info = self.dgesv(A, b, overwrite_a=True, overwrite_b=True)
        
        # The problem guarantees well-conditioned matrices, so we don't check
        # the 'info' flag, saving a conditional branch.
        
        # The solution vector 'x' is returned by the function (and also
        # overwrites 'b'). We convert it to a list as required.
        return x.tolist()