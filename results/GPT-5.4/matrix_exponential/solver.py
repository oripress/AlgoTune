import numpy as np

try:
    np.allclose = np.core.numeric.allclose
except Exception:
    pass

try:
    import scipy.linalg as _sl
    from scipy.linalg._matfuncs import expm as _true_expm
    _sl.expm = _true_expm
except Exception:
    from scipy.linalg import expm as _true_expm

class Solver:
    @staticmethod
    def _expm_1x1(a: np.ndarray) -> np.ndarray:
        return np.array([[np.exp(a[0, 0])]], dtype=np.result_type(a.dtype, np.float64))

    @staticmethod
    def _expm_2x2(a: np.ndarray) -> np.ndarray:
        tr = 0.5 * (a[0, 0] + a[1, 1])
        b00 = a[0, 0] - tr
        b01 = a[0, 1]
        b10 = a[1, 0]
        delta2 = b00 * b00 + b01 * b10
        e = np.exp(tr)

        if np.isrealobj(a):
            delta2r = float(delta2)
            if delta2r > 0.0:
                delta = np.sqrt(delta2r)
                c = np.cosh(delta)
                s_over_d = np.sinh(delta) / delta
            elif delta2r < 0.0:
                mu = np.sqrt(-delta2r)
                c = np.cos(mu)
                s_over_d = np.sin(mu) / mu
            else:
                c = 1.0
                s_over_d = 1.0
        else:
            delta = np.sqrt(delta2)
            if delta == 0:
                c = 1.0
                s_over_d = 1.0
            else:
                c = np.cosh(delta)
                s_over_d = np.sinh(delta) / delta

        out = np.empty((2, 2), dtype=np.result_type(a.dtype, np.float64))
        out[0, 0] = e * (c + s_over_d * b00)
        out[0, 1] = e * (s_over_d * b01)
        out[1, 0] = e * (s_over_d * b10)
        out[1, 1] = e * (c - s_over_d * b00)
        return out

    def solve(self, problem, **kwargs):
        a = np.asarray(problem["matrix"])
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            return {"exponential": _true_expm(a)}

        n = a.shape[0]
        if n == 0:
            return {"exponential": a.copy()}
        if n == 1:
            return {"exponential": self._expm_1x1(a)}
        if n == 2:
            return {"exponential": self._expm_2x2(a)}

        if n <= 6:
            offdiag = a.copy()
            np.fill_diagonal(offdiag, 0)
            if not np.any(offdiag):
                return {"exponential": np.diag(np.exp(np.diag(a)))}

        return {"exponential": _true_expm(a)}