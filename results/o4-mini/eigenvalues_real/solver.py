import numpy as np

# Try SciPy's high-level eigvalsh with MRRR algorithm; fallback to NumPy
try:
    from scipy.linalg import eigvalsh as _sc_eigvalsh
    # Test driver support
    _sc_eigvalsh(np.eye(2), driver='evr', lower=True)
    def _eig(a):
        return _sc_eigvalsh(a, driver='evr', lower=True)
except Exception:
    _eig = np.linalg.eigvalsh

class Solver:
    __slots__ = ()
    def solve(self, problem, __e=_eig):
        # Compute ascending eigenvalues and reverse for descending order
        w = __e(problem)
        return w[::-1].tolist()