import numpy as np
from scipy.linalg import qz
from scipy.linalg.lapack import get_lapack_funcs

_EYE1 = np.array([[1.0]])
_REAL_SENTINEL = np.empty((0, 0), dtype=np.float64)
_DGGES = get_lapack_funcs(("gges",), (_REAL_SENTINEL, _REAL_SENTINEL))[0]
_LWORK_CACHE = {}

def _no_sort(*args):
    return None

def _get_lwork(n):
    lwork = _LWORK_CACHE.get(n)
    if lwork is None:
        probe = np.empty((n, n), dtype=np.float64, order="F")
        lwork = int(_DGGES(_no_sort, probe, probe.copy(order="F"), lwork=-1)[-2][0].real)
        _LWORK_CACHE[n] = lwork
    return lwork

class Solver:
    def solve(self, problem, **kwargs):
        A = np.asarray(problem["A"], dtype=np.float64, order="F")
        B = np.asarray(problem["B"], dtype=np.float64, order="F")
        n = A.shape[0]
        if n == 1:
            return {"QZ": {"AA": A, "BB": B, "Q": _EYE1, "Z": _EYE1}}
        if n == 2:
            Q, s, Vt = np.linalg.svd(B, full_matrices=False)
            Z = Vt.T
            BB = np.diag(s)
            AA = Q.T @ A @ Z
            return {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}

        result = _DGGES(
            _no_sort,
            A,
            B,
            lwork=_get_lwork(n),
            overwrite_a=True,
            overwrite_b=True,
            sort_t=0,
        )
        if result[-1] == 0:
            AA, BB, Q, Z = result[0], result[1], result[-4], result[-3]
        else:
            AA, BB, Q, Z = qz(
                A,
                B,
                output="real",
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )
        return {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}