import numpy as np
# Try to import the fast SciPy implementation once
try:
    from scipy.linalg import sqrtm as _scipy_sqrtm
except Exception:
    _scipy_sqrtm = None

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the principal matrix square root of a square (possibly complex) matrix.
        The principal square root X satisfies X @ X = A and has eigenvalues with
        non‑negative real parts (the standard principal branch of the complex square root).

        Parameters
        ----------
        problem : dict
            Dictionary with key "matrix" containing a list of lists representing the matrix.
            Elements may be strings like "1+2j" or numeric complex literals.

        Returns
        -------
        dict
            {"sqrtm": {"X": X.tolist()}} where X is the computed square‑root matrix.
        """
        raw_mat = problem.get("matrix")
        if raw_mat is None:
            return {"sqrtm": {"X": []}}

        # Fast conversion: NumPy can parse string representations of complex numbers directly
        try:
            A = np.array(raw_mat, dtype=complex)
        except Exception:
            return {"sqrtm": {"X": []}}

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"sqrtm": {"X": []}}

        # Compute principal square root using SciPy if available (fast and robust)
        if _scipy_sqrtm is not None:
            try:
                X = _scipy_sqrtm(A, disp=False)
                # scipy.linalg.sqrtm may return (X, info); keep only the matrix
                if isinstance(X, tuple):
                    X = X[0]
            except Exception:
                X = None
        else:
            X = None

        # Fallback to eigen‑decomposition for general matrices
        if X is None:
            try:
                w, V = np.linalg.eig(A)
                sqrt_w = np.sqrt(w)                     # principal branch
                V_D = V * sqrt_w                        # scale columns of V
                X = np.linalg.solve(V, V_D)             # X = V @ diag(sqrt_w) @ V^{-1}
            except Exception:
                return {"sqrtm": {"X": []}}
                return {"sqrtm": {"X": []}}
        return {"sqrtm": {"X": X.tolist()}}