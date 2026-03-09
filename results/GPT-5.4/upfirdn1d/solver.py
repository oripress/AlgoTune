from math import gcd
from typing import Any

import numpy as np
from numba import njit
from scipy import signal

try:
    from scipy.signal._upfirdn import _UpFIRDn
except Exception:  # pragma: no cover
    _UpFIRDn = None

@njit(cache=True)
def _upfirdn_real64(h: np.ndarray, x: np.ndarray, up: int, down: int) -> np.ndarray:
    n = x.shape[0]
    m = h.shape[0]
    if n == 0 or m == 0:
        return np.empty(0, dtype=np.float64)

    out_len = ((n - 1) * up + m + down - 1) // down
    out = np.zeros(out_len, dtype=np.float64)

    for i in range(out_len):
        pos = i * down

        start = 0
        if pos >= m:
            start = (pos - (m - 1) + up - 1) // up

        stop = pos // up
        if stop >= n:
            stop = n - 1

        acc = 0.0
        for t in range(start, stop + 1):
            acc += x[t] * h[pos - t * up]
        out[i] = acc

    return out

@njit(cache=True)
def _upfirdn_complex128(
    h: np.ndarray, x: np.ndarray, up: int, down: int
) -> np.ndarray:
    n = x.shape[0]
    m = h.shape[0]
    if n == 0 or m == 0:
        return np.empty(0, dtype=np.complex128)

    out_len = ((n - 1) * up + m + down - 1) // down
    out = np.zeros(out_len, dtype=np.complex128)

    for i in range(out_len):
        pos = i * down

        start = 0
        if pos >= m:
            start = (pos - (m - 1) + up - 1) // up

        stop = pos // up
        if stop >= n:
            stop = n - 1

        acc = 0.0j
        for t in range(start, stop + 1):
            acc += x[t] * h[pos - t * up]
        out[i] = acc

    return out

class Solver:
    def __init__(self) -> None:
        self._ufd_cache: dict[tuple[Any, ...], Any] = {}
        self._inv_cache: dict[tuple[int, int], int] = {}

        _upfirdn_real64(np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64), 1, 1)
        _upfirdn_complex128(
            np.ones(1, dtype=np.complex128),
            np.ones(1, dtype=np.complex128),
            1,
            1,
        )

    def _output_dtype(self, h_arr: np.ndarray, x_arr: np.ndarray) -> np.dtype:
        return np.result_type(h_arr.dtype, x_arr.dtype, np.float32)

    def _cached_upfirdn(
        self,
        h_obj: Any,
        h_arr: np.ndarray,
        x_arr: np.ndarray,
        up: int,
        down: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        return signal.upfirdn(
            np.asarray(h_arr, dtype=out_dtype),
            np.asarray(x_arr, dtype=out_dtype),
            up=up,
            down=down,
        )

    def _polyphase_numpy(
        self,
        h_arr: np.ndarray,
        x_arr: np.ndarray,
        up: int,
        down: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        n = x_arr.size
        m = h_arr.size
        if n == 0 or m == 0:
            return np.empty(0, dtype=out_dtype)

        h_cast = np.asarray(h_arr, dtype=out_dtype)
        x_cast = np.asarray(x_arr, dtype=out_dtype)

        y_len = (n - 1) * up + m
        out_len = (y_len + down - 1) // down

        g = gcd(up, down)
        u1 = up // g
        d1 = down // g

        if u1 == 1:
            conv = np.convolve(x_cast, h_cast)
            return conv[::d1][:out_len]

        inv_key = (d1, u1)
        inv = self._inv_cache.get(inv_key)
        if inv is None:
            inv = pow(d1, -1, u1)
            self._inv_cache[inv_key] = inv

        out = np.zeros(out_len, dtype=out_dtype)
        phase_stop = min(m, up)

        for p in range(0, phase_stop, g):
            hp = h_cast[p::up]
            if hp.size == 0:
                continue
            if hp.size == 1:
                conv = x_cast * hp[0]
            else:
                conv = np.convolve(x_cast, hp)

            r = p // g
            n0 = (r * inv) % u1
            if n0 >= out_len:
                continue

            q0 = (n0 * down - p) // up
            if q0 >= conv.size:
                continue

            out_count = (out_len - 1 - n0) // u1 + 1
            conv_count = (conv.size - 1 - q0) // d1 + 1
            count = out_count if out_count < conv_count else conv_count

            out[n0 : n0 + count * u1 : u1] = conv[q0 : q0 + count * d1 : d1]

        return out

    def _solve_one(self, h: Any, x: Any, up: int, down: int) -> np.ndarray:
        return signal.upfirdn(h, x, up=up, down=down)

    def solve(self, problem, **kwargs) -> Any:
        solve_one = self._solve_one
        return [solve_one(h, x, up, down) for h, x, up, down in problem]