from typing import Any

import numpy as np
from scipy.ndimage import _nd_image, _ni_support

_MODE_CONSTANT = _ni_support._extend_mode_to_code("constant")
_SPLINE_FILTER1D = _nd_image.spline_filter1d
_ZOOM_SHIFT = _nd_image.zoom_shift

class Solver:
    __slots__ = ("_filtered_buf", "_filtered_cap", "_out_buf", "_out_cap", "_zoom", "_sol")

    def __init__(self) -> None:
        self._filtered_buf = None
        self._filtered_cap = 0
        self._out_buf = None
        self._out_cap = 0
        self._zoom = np.empty(2, dtype=np.float64)
        self._sol = {"zoomed_image": None}

    def solve(self, problem, **kwargs) -> Any:
        image = problem["image"]
        z = problem["zoom_factor"]

        if z == 1:
            self._sol["zoomed_image"] = image
            return self._sol

        n0 = len(image)
        n1 = len(image[0])
        o0 = int(round(n0 * z))
        o1 = int(round(n1 * z))

        in_size = n0 * n1
        filtered_buf = self._filtered_buf
        if filtered_buf is None or self._filtered_cap < in_size:
            filtered_buf = np.empty(in_size, dtype=np.float64)
            self._filtered_buf = filtered_buf
            self._filtered_cap = in_size
        filtered = filtered_buf[:in_size].reshape(n0, n1)
        filtered[...] = image

        out_size = o0 * o1
        out_buf = self._out_buf
        if out_buf is None or self._out_cap < out_size:
            out_buf = np.empty(out_size, dtype=np.float64)
            self._out_buf = out_buf
            self._out_cap = out_size
        out = out_buf[:out_size].reshape(o0, o1)

        _SPLINE_FILTER1D(filtered, 3, 0, filtered, _MODE_CONSTANT)
        _SPLINE_FILTER1D(filtered, 3, 1, filtered, _MODE_CONSTANT)

        zoom = self._zoom
        zoom[0] = 1.0 if o0 == 1 else (n0 - 1.0) / (o0 - 1.0)
        zoom[1] = 1.0 if o1 == 1 else (n1 - 1.0) / (o1 - 1.0)
        _ZOOM_SHIFT(filtered, zoom, None, out, 3, _MODE_CONSTANT, 0.0, 0, False)

        self._sol["zoomed_image"] = out
        return self._sol

        self._sol["zoomed_image"] = out
        return self._sol

        self._sol["zoomed_image"] = out
        return self._sol