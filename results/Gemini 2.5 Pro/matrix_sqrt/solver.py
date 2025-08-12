import numpy as np
from scipy.linalg import sqrtm, LinAlgError
from typing import Any, Dict

class Solver:
    """
    A solver for the matrix square root problem using SciPy.

    This implementation uses the robust and well-established `scipy.linalg.sqrtm`
    function. This approach avoids the prohibitive one-time startup and
    compilation costs associated with JAX, which caused persistent timeouts in
    an environment where each problem is likely run in a separate process.
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Computes the principal matrix square root X of a given matrix A.

        :param problem: A dictionary containing the input matrix under the key "matrix".
        :return: A dictionary with the principal square root matrix "X" or an
                 empty list in case of a computation failure.
        """
        try:
            # Convert the input list of lists to a NumPy array.
            # Using complex128 ensures high precision and handles complex inputs.
            A = np.array(problem["matrix"], dtype=np.complex128)
        except (ValueError, TypeError):
            # If the input is malformed (e.g., not a valid matrix structure),
            # return the failure format.
            return {"sqrtm": {"X": []}}

        # The matrix must be square.
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"sqrtm": {"X": []}}

        try:
            # Compute the principal square root using SciPy's implementation.
            # `disp=False` prevents printing the error estimate to stdout.
            # The function returns a tuple (sqrtm, errest); we only need the matrix.
            X = sqrtm(A, disp=False)[0]

            # The validation requires finite numbers. Check for NaN/inf which can
            # occur for ill-conditioned matrices.
            if not np.all(np.isfinite(X)):
                return {"sqrtm": {"X": []}}

            # Format the solution as required: a list of lists.
            solution = {"sqrtm": {"X": X.tolist()}}
        except (LinAlgError, ValueError):
            # `sqrtm` can raise LinAlgError for singular matrices or other issues.
            # A ValueError might also occur for certain invalid inputs.
            # In case of any computation error, return the failure format.
            return {"sqrtm": {"X": []}}

        return solution