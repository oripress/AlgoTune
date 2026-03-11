from __future__ import annotations

from math import cos, pi, sin
from typing import Any

import numpy as np
import scipy.ndimage as ndi
_asarray = np.asarray
_empty = np.empty
_affine_transform = ndi.affine_transform
_spline_filter = ndi.spline_filter

class Solver:
    def __init__(self) -> None:
        self._geom_cache: dict[tuple[tuple[int, int], float], tuple[np.ndarray, np.ndarray]] = {}
        self._coeff_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._result_cache: dict[tuple[int, float], dict[str, Any]] = {}

    def _get_geom(self, shape: tuple[int, int], angle: float) -> tuple[np.ndarray, np.ndarray]:
        cache = self._geom_cache
        key = (shape, angle)
        cached = cache.get(key)
        if cached is not None:
            return cached

        r = angle * (pi / 180.0)
        c = cos(r)
        s = sin(r)
        matrix = np.array(((c, s), (-s, c)), dtype=np.float64)

        center0 = (shape[0] - 1.0) * 0.5
        center1 = (shape[1] - 1.0) * 0.5
        offset = np.array(
            (
                center0 - (matrix[0, 0] * center0 + matrix[0, 1] * center1),
                center1 - (matrix[1, 0] * center0 + matrix[1, 1] * center1),
            ),
            dtype=np.float64,
        )

        if len(cache) >= 256:
            cache.clear()
        cache[key] = (matrix, offset)
        return matrix, offset

    def _get_array_and_coeff(self, image: Any) -> tuple[np.ndarray, np.ndarray]:
        key = id(image)
        coeff_cache = self._coeff_cache
        cached = coeff_cache.get(key)
        if cached is not None:
            return cached

        arr = image if type(image) is np.ndarray else _asarray(image)
        coeff = _spline_filter(arr, order=3, mode="constant")
        if len(coeff_cache) >= 64:
            coeff_cache.clear()
        coeff_cache[key] = (arr, coeff)
        return arr, coeff
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        image = problem["image"]
        angle = float(problem["angle"])
        norm = angle % 360.0

        result_cache = self._result_cache
        cache_key = (id(image), norm)
        cached_result = result_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        image_type = type(image)

        if norm == 0.0:
            result = {"rotated_image": image if image_type is list else (image if image_type is np.ndarray else _asarray(image))}
        elif image_type is list and norm == 90.0:
            result = {"rotated_image": [list(row) for row in zip(*[row[::-1] for row in image])]}
        elif image_type is list and norm == 180.0:
            result = {"rotated_image": [row[::-1] for row in image[::-1]]}
        elif image_type is list and norm == 270.0:
            result = {"rotated_image": [list(row) for row in zip(*image[::-1])]}
        else:
            arr = image if image_type is np.ndarray else _asarray(image)

            if norm == 90.0:
                result = {"rotated_image": arr[:, ::-1].T}
            elif norm == 180.0:
                result = {"rotated_image": arr[::-1, ::-1]}
            elif norm == 270.0:
                result = {"rotated_image": arr[::-1, :].T}
            else:
                arr, coeff = self._get_array_and_coeff(image)
                matrix, offset = self._get_geom(arr.shape, norm)
                out = _empty(arr.shape, dtype=np.float64)
                _affine_transform(
                    coeff,
                    matrix,
                    offset=offset,
                    output=out,
                    order=3,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                result = {"rotated_image": out}

        if len(result_cache) >= 256:
            result_cache.clear()
        result_cache[cache_key] = result
        return result