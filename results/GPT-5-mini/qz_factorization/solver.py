import numpy as np
from scipy.linalg import qz
try:
    from scipy.linalg.lapack import get_lapack_funcs
    _HAS_GET_LAPACK_FUNCS = True
except Exception:
    _HAS_GET_LAPACK_FUNCS = False

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the QZ (generalized Schur) factorization for (A, B).

        Tries a low-level LAPACK 'gges' fast path (when available) for speed,
        and falls back to scipy.linalg.qz for robustness.

        Input:
            problem: {"A": [[...], ...], "B": [[...], ...]}

        Output:
            {"QZ": {"AA": ..., "BB": ..., "Q": ..., "Z": ...}}
        """
        if not isinstance(problem, dict):
            raise TypeError("problem must be a dict containing 'A' and 'B'")
        if "A" not in problem or "B" not in problem:
            raise KeyError("problem must contain keys 'A' and 'B'")

        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2-D arrays")
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("A and B must be square matrices of the same shape")

        n = A.shape[0]
        if n == 0:
            return {"QZ": {"AA": [], "BB": [], "Q": [], "Z": []}}

        # Determine dtype and output mode
        complex_input = np.iscomplexobj(A) or np.iscomplexobj(B)
        dtype = np.complex128 if complex_input else np.float64
        output_mode = "complex" if complex_input else "real"

        # Fortran-contiguous arrays (favored by LAPACK)
        A_f = np.array(A, dtype=dtype, order="F", copy=False)
        B_f = np.array(B, dtype=dtype, order="F", copy=False)

        # Quick return for 1x1 matrices
        if n == 1:
            AA = np.array([[A_f[0, 0]]], dtype=dtype)
            BB = np.array([[B_f[0, 0]]], dtype=dtype)
            Q = np.eye(1, dtype=dtype)
            Z = np.eye(1, dtype=dtype)
            return {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}

        # Fast path: try low-level LAPACK gges if available
        if _HAS_GET_LAPACK_FUNCS:
            try:
                gges = get_lapack_funcs("gges", (A_f, B_f))
                a = A_f.copy(order="F")
                b = B_f.copy(order="F")
                res = None

                # Try common calling conventions to be robust across SciPy versions
                try:
                    res = gges(a, b, jobvsl="V", jobvsr="V", sort="N", overwrite_a=1, overwrite_b=1)
                except TypeError:
                    try:
                        res = gges(a, b, "V", "V", "N")
                    except TypeError:
                        try:
                            res = gges(a, b)
                        except Exception:
                            res = None

                if res is not None:
                    # Normalize to tuple/list
                    if not isinstance(res, (list, tuple)):
                        res = (res,)

                    # Extract ndarray outputs of shape (n, n) - typically a, b, vsl, vsr
                    mats = [x for x in res if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape == (n, n)]
                    if len(mats) >= 4:
                        AA, BB, VSL, VSR = mats[0], mats[1], mats[2], mats[3]
                        return {
                            "QZ": {
                                "AA": AA.tolist(),
                                "BB": BB.tolist(),
                                "Q": VSL.tolist(),
                                "Z": VSR.tolist(),
                            }
                        }
            except Exception:
                # If anything goes wrong in the fast path, fall back to qz
                pass

        # Robust fallback using scipy.linalg.qz
        AA, BB, Q, Z = qz(
            A_f,
            B_f,
            output=output_mode,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
        )

        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }