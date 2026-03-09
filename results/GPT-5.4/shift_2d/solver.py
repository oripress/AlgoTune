from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import _nd_image, _ni_support, shift as _ref_shift, spline_filter

_ORDER = 3
_MODE = "constant"
_CVAL = 0.0
_MODE_CODE = _ni_support._extend_mode_to_code(_MODE)

class Solver:
    def __init__(self):
        self._shape = None
        self._filtered = None
        self._out = None
        self._shift_arr = np.empty(2, dtype=np.float64)
        self._use_fast = self._self_test()

    def _get_buffers(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._shape != shape:
            self._shape = shape
            self._filtered = np.empty(shape, dtype=np.float64)
            self._out = np.empty(shape, dtype=np.float64)
        return self._filtered, self._out

    def _fast_shift_2d(self, image: Any, shift: Any) -> np.ndarray:
        arr = np.asarray(image)
        sr = float(shift[0])
        sc = float(shift[1])

        ir = int(sr)
        ic = int(sc)
        if sr == ir and sc == ic:
            _, out = self._get_buffers(arr.shape)
            out.fill(0.0)

            n0, n1 = arr.shape
            src_r0 = max(0, -ir)
            src_r1 = min(n0, n0 - ir)
            src_c0 = max(0, -ic)
            src_c1 = min(n1, n1 - ic)
            dst_r0 = src_r0 + ir
            dst_r1 = src_r1 + ir
            dst_c0 = src_c0 + ic
            dst_c1 = src_c1 + ic
            if src_r0 < src_r1 and src_c0 < src_c1:
                out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
            return out

        filtered, out = self._get_buffers(arr.shape)
        spline_filter(arr, order=_ORDER, output=filtered, mode=_MODE)

        self._shift_arr[0] = -sr
        self._shift_arr[1] = -sc
        _nd_image.zoom_shift(
            filtered,
            None,
            self._shift_arr,
            out,
            _ORDER,
            _MODE_CODE,
            _CVAL,
            0,
            False,
        )
        return out

    def _self_test(self) -> bool:
        rng = np.random.default_rng(0)
        tests = [
            (np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64), (0.0, 0.0)),
            (
                np.array(
                    [[0.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                (0.5, -0.5),
            ),
            (np.arange(16.0, dtype=np.float64).reshape(4, 4), (1.25, 0.75)),
            (np.arange(16.0, dtype=np.float64).reshape(4, 4), (1.0, -2.0)),
        ]
        for n in (3, 5, 8):
            for _ in range(2):
                tests.append((rng.random((n, n)) * 255.0, tuple(rng.uniform(-2, 2, size=2))))
        for img, sh in tests:
            ref = _ref_shift(img, sh, order=_ORDER, mode=_MODE)
            got = self._fast_shift_2d(img, sh)
            if not np.allclose(got, ref, rtol=1e-11, atol=1e-11):
                return False
        return True

    def solve(self, problem, **kwargs) -> Any:
        if self._use_fast:
            return {"shifted_image": self._fast_shift_2d(problem["image"], problem["shift"])}
        return {
            "shifted_image": _ref_shift(
                problem["image"],
                problem["shift"],
                order=_ORDER,
                mode=_MODE,
            )
        }