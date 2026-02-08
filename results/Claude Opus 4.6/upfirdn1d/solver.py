import numpy as np
from numba import njit, prange  # noqa
from scipy.signal import upfirdn as _scipy_upfirdn

@njit(cache=True, parallel=True)
def _upfirdn1d_par(h, x, up, down):  # noqa
    len_h = h.shape[0]
    len_x = x.shape[0]
    out_len_full = (len_x - 1) * up + len_h
    out_len = (out_len_full - 1) // down + 1
    out = np.empty(out_len, dtype=np.float64)
    for i in prange(out_len):  # pylint: disable=not-an-iterable  # noqa
        n = i * down
        val = 0.0
        k_min = n - (len_x - 1) * up
        if k_min < 0:
            k_min = 0
        k_max = n
        if k_max > len_h - 1:
            k_max = len_h - 1
        r = n % up
        rem = k_min % up
        if rem <= r:
            k_start = k_min + (r - rem)
        else:
            k_start = k_min + (up - rem + r)
        k = k_start
        while k <= k_max:
            val += h[k] * x[(n - k) // up]
            k += up
        out[i] = val
    return out

@njit(cache=True)
def _upfirdn1d_seq(h, x, up, down):
    len_h = h.shape[0]
    len_x = x.shape[0]
    out_len_full = (len_x - 1) * up + len_h
    out_len = (out_len_full - 1) // down + 1
    out = np.empty(out_len, dtype=np.float64)
    for i in range(out_len):
        n = i * down
        val = 0.0
        k_min = n - (len_x - 1) * up
        if k_min < 0:
            k_min = 0
        k_max = n
        if k_max > len_h - 1:
            k_max = len_h - 1
        r = n % up
        rem = k_min % up
        if rem <= r:
            k_start = k_min + (r - rem)
        else:
            k_start = k_min + (up - rem + r)
        k = k_start
        while k <= k_max:
            val += h[k] * x[(n - k) // up]
            k += up
        out[i] = val
    return out

# Warm up both
_h = np.array([1.0, 2.0])
_x = np.array([1.0, 2.0, 3.0])
_upfirdn1d_par(_h, _x, 1, 1)
_upfirdn1d_par(_h, _x, 2, 3)
_upfirdn1d_seq(_h, _x, 1, 1)
_upfirdn1d_seq(_h, _x, 2, 3)

class Solver:
    def solve(self, problem, **kwargs):
        results = []
        for h, x, up, down in problem:
            h_arr = np.asarray(h, dtype=np.float64)
            x_arr = np.asarray(x, dtype=np.float64)
            lh = len(h_arr)
            lx = len(x_arr)
            out_full = (lx - 1) * up + lh
            out_len = (out_full - 1) // down + 1
            if out_len > 5000:
                res = _upfirdn1d_par(h_arr, x_arr, up, down)
            elif out_len > 50:
                res = _upfirdn1d_seq(h_arr, x_arr, up, down)
            else:
                res = _scipy_upfirdn(h_arr, x_arr, up=up, down=down)
            results.append(res)
        return results