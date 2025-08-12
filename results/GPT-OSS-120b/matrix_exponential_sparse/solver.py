import logging
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm

class SafeCSC(csc_matrix):
    """A CSC matrix with safe length and truth-value semantics for validation."""
    def __len__(self):
        # Return number of rows (consistent with dense arrays)
        return self.shape[0]

    def __bool__(self):
        # Define truthiness: non-empty matrix is True
        return self.shape[0] != 0 and self.shape[1] != 0

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the matrix exponential of a given sparse CSC matrix.

        Parameters
        ----------
        problem : dict
            Dictionary with key "matrix" containing a scipy.sparse matrix (CSC format).

        Returns
        -------
        scipy.sparse.csc_matrix
            The exponential of the input matrix in CSC format.
        """
        try:
            A = problem.get("matrix")
            if A is None:
                raise ValueError("Problem dictionary missing 'matrix' key.")
            # Ensure input is in CSC format; if not, convert.
            if not isinstance(A, csc_matrix):
                A = A.tocsc()
            expA = expm(A)
            # Ensure result is CSC and wrap in SafeCSC to avoid ambiguous length errors.
            if not isinstance(expA, csc_matrix):
                expA = expA.tocsc()
            # Wrap the result
            if not isinstance(expA, SafeCSC):
                expA = SafeCSC(expA)
            return expA
        except Exception as e:
            logging.error(f"Failed to compute matrix exponential: {e}")
            raise