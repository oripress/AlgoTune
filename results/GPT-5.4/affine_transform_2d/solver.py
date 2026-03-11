from typing import Any
import math

import numpy as np
from numba import njit
from scipy.ndimage import affine_transform, spline_filter
from scipy.ndimage import _nd_image, _ni_support

@njit(cache=True)
def _integer_affine_2d(
    image: np.ndarray,
    a00: int,
    a01: int,
    a02: int,
    a10: int,
    a11: int,
    a12: int,
) -> np.ndarray:
    n0, n1 = image.shape
    out = np.zeros((n0, n1), dtype=image.dtype)
    for i in range(n0):
        y = a00 * i + a02
        x = a10 * i + a12
        for j in range(n1):
            if 0 <= y < n0 and 0 <= x < n1:
                out[i, j] = image[y, x]
            y += a01
            x += a11
    return out

@njit(cache=True)
def _cubic_affine_small(
    coeffs: np.ndarray,
    a00: float,
    a01: float,
    a02: float,
    a10: float,
    a11: float,
    a12: float,
) -> np.ndarray:
    n0, n1 = coeffs.shape
    out = np.empty((n0, n1), dtype=np.float64)
    for i in range(n0):
        y = a00 * i + a02
        x = a10 * i + a12
        for j in range(n1):
            fy = math.floor(y)
            fx = math.floor(x)
            ty = y - fy
            tx = x - fx

            wy0 = ((1.0 - ty) * (1.0 - ty) * (1.0 - ty)) / 6.0
            wy1 = (3.0 * ty * ty * ty - 6.0 * ty * ty + 4.0) / 6.0
            wy2 = (-3.0 * ty * ty * ty + 3.0 * ty * ty + 3.0 * ty + 1.0) / 6.0
            wy3 = (ty * ty * ty) / 6.0

            wx0 = ((1.0 - tx) * (1.0 - tx) * (1.0 - tx)) / 6.0
            wx1 = (3.0 * tx * tx * tx - 6.0 * tx * tx + 4.0) / 6.0
            wx2 = (-3.0 * tx * tx * tx + 3.0 * tx * tx + 3.0 * tx + 1.0) / 6.0
            wx3 = (tx * tx * tx) / 6.0

            iy0 = int(fy) - 1
            ix0 = int(fx) - 1

            acc = 0.0
            for dy in range(4):
                iy = iy0 + dy
                if 0 <= iy < n0:
                    if dy == 0:
                        wy = wy0
                    elif dy == 1:
                        wy = wy1
                    elif dy == 2:
                        wy = wy2
                    else:
                        wy = wy3
                    row = coeffs[iy]
                    for dx in range(4):
                        ix = ix0 + dx
                        if 0 <= ix < n1:
                            if dx == 0:
                                wx = wx0
                            elif dx == 1:
                                wx = wx1
                            elif dx == 2:
                                wx = wx2
                            else:
                                wx = wx3
                            acc += row[ix] * wy * wx

            out[i, j] = acc
            y += a01
            x += a11
    return out

class Solver:
    __slots__ = ("_mode_code", "_buf_shape", "_buf_filtered", "_buf_out")

    def __init__(self) -> None:
        self._mode_code = _ni_support._extend_mode_to_code("constant")
        self._buf_shape = None
        self._buf_filtered = None
        self._buf_out = None
        dummy = np.zeros((2, 2), dtype=np.float64)
        _integer_affine_2d(dummy, 1, 0, 0, 0, 1, 0)
        _cubic_affine_small(dummy, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def solve(self, problem, **kwargs) -> Any:
        image = problem["image"]
        matrix = problem["matrix"]

        try:
            if image.ndim == 2 and matrix.shape == (2, 3):
                a00 = matrix[0, 0]
                a01 = matrix[0, 1]
                a02 = matrix[0, 2]
                a10 = matrix[1, 0]
                a11 = matrix[1, 1]
                a12 = matrix[1, 2]

                if a00 == 1.0 and a01 == 0.0 and a10 == 0.0 and a11 == 1.0:
                    if a02 == 0.0 and a12 == 0.0:
                        return {"transformed_image": image.copy()}

                    i02 = int(a02)
                    i12 = int(a12)
                    if a02 == i02 and a12 == i12:
                        n0, n1 = image.shape
                        out = np.zeros_like(image)
                        d0s = 0 if i02 >= 0 else -i02
                        d0e = n0 if i02 <= 0 else n0 - i02
                        d1s = 0 if i12 >= 0 else -i12
                        d1e = n1 if i12 <= 0 else n1 - i12
                        if d0s < d0e and d1s < d1e:
                            s0s = i02 if i02 >= 0 else 0
                            s1s = i12 if i12 >= 0 else 0
                            out[d0s:d0e, d1s:d1e] = image[
                                s0s : s0s + (d0e - d0s), s1s : s1s + (d1e - d1s)
                            ]
                        return {"transformed_image": out}

                ia00 = int(a00)
                ia01 = int(a01)
                ia02 = int(a02)
                ia10 = int(a10)
                ia11 = int(a11)
                ia12 = int(a12)
                if (
                    a00 == ia00
                    and a01 == ia01
                    and a02 == ia02
                    and a10 == ia10
                    and a11 == ia11
                    and a12 == ia12
                ):
                    return {
                        "transformed_image": _integer_affine_2d(
                            image, ia00, ia01, ia02, ia10, ia11, ia12
                        )
                    }

                shape = image.shape
                if shape != self._buf_shape:
                    self._buf_shape = shape
                    self._buf_filtered = np.empty(shape, dtype=np.float64)
                    self._buf_out = np.empty(shape, dtype=np.float64)

                spline_filter(image, 3, output=self._buf_filtered)
                if shape[0] * shape[1] <= 4096:
                    return {
                        "transformed_image": _cubic_affine_small(
                            self._buf_filtered,
                            float(a00),
                            float(a01),
                            float(a02),
                            float(a10),
                            float(a11),
                            float(a12),
                        )
                    }

                _nd_image.geometric_transform(
                    self._buf_filtered,
                    None,
                    None,
                    np.ascontiguousarray(matrix[:, :2], dtype=np.float64),
                    np.ascontiguousarray(matrix[:, 2], dtype=np.float64),
                    self._buf_out,
                    3,
                    self._mode_code,
                    0.0,
                    0,
                    None,
                    None,
                )
                return {"transformed_image": self._buf_out}
        except Exception:
            pass

        try:
            return {
                "transformed_image": affine_transform(
                    image, matrix, 0.0, None, None, 3, "constant"
                )
            }
        except Exception:
            return {"transformed_image": []}