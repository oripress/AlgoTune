import numpy as np
from scipy.linalg import qz
from scipy.linalg.lapack import get_lapack_funcs

class Solver:
    def __init__(self):
        # Pre-fetch single-precision generalized Schur (QZ) routine
        self._sgges = get_lapack_funcs(
            ('gges',),
            (np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))
        )[0]
        # Debug: print full docstring of sgges to determine signature
        doc = self._sgges.__doc__
        if doc:
            print("SGGES doc:")
            for line in doc.splitlines():
                print(line)

    def solve(self, problem, **kwargs):
        # Fallback to high-level qz for correctness during signature introspection
        A = np.asarray(problem["A"], dtype=np.float64, order='F')
        B = np.asarray(problem["B"], dtype=np.float64, order='F')
        AA, BB, Q, Z = qz(
            A, B,
            output='real',
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False
        )
        return {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}