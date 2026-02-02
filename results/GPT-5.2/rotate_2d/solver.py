from __future__ import annotations

from typing import Any, Dict, Tuple
import math

import numpy as np
import scipy.ndimage

class Solver:
    """
    Fast 2D rotation matching:
      scipy.ndimage.rotate(image, angle, reshape=False, order=3, mode="constant", cval=0.0)

    Optimizations:
      - Exact 0/90/180/270 deg via NumPy.
      - Use affine_transform directly (less overhead than rotate).
      - Cache (matrix, offset) by (h, w, angle).
      - Reuse an output buffer for repeated shapes/dtypes.
    """

    __slots__ = (
        "reshape",
        "order",
        "mode",
        "_eps",
        "_cache",
        "_affine",
        "_out",
        "_out_key",
        "_deg2rad",
    )

    def __init__(self) -> None:
        self.reshape = False
        self.order = 3
        self.mode = "constant"
        self._eps = 1e-12
        self._deg2rad = math.pi / 180.0

        self._affine = scipy.ndimage.affine_transform

        self._cache: Dict[
            Tuple[int, int, float],
            Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[float, float]],
        ] = {}

        self._out: np.ndarray | None = None
        self._out_key: Tuple[int, int, np.dtype] | None = None

    def _ensure_out(self, h: int, w: int, dtype: np.dtype) -> np.ndarray:
        key = (h, w, dtype)
        out = self._out
        if out is None or self._out_key != key:
            out = np.empty((h, w), dtype=dtype)
            self._out = out
            self._out_key = key
        return out

    def _get_matrix_offset(
        self, h: int, w: int, angle: float
    ) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[float, float]]:
        key = (h, w, angle)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        theta = angle * self._deg2rad
        c = math.cos(theta)
        s = math.sin(theta)

        # Matches scipy.ndimage.rotate convention for 2D default axes=(1,0) in (row, col):
        # input = [[cos, sin], [-sin, cos]] @ output + offset
        m00 = c
        m01 = s
        m10 = -s
        m11 = c

        c0 = (h - 1) * 0.5
        c1 = (w - 1) * 0.5
        off0 = c0 - (m00 * c0 + m01 * c1)
        off1 = c1 - (m10 * c0 + m11 * c1)

        matrix = ((m00, m01), (m10, m11))
        offset = (off0, off1)
        self._cache[key] = (matrix, offset)
        return matrix, offset

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        img = problem["image"]
        angle = float(problem["angle"])

        # Benchmark inputs are expected to already be numpy arrays with .shape
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim != 2:
            return {"rotated_image": []}

        # Keep dtype when already float; otherwise cast to float64 (reference tolerates).
        if not np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float64, copy=False)

        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)

        a = angle % 360.0

        # 0-degree shortcut
        if abs(a) < self._eps:
            return {"rotated_image": img}

        # Exact right angles
        k = int(round(a / 90.0))
        if abs(a - 90.0 * k) < self._eps:
            return {"rotated_image": np.rot90(img, k=k)}

        h, w = img.shape
        matrix, offset = self._get_matrix_offset(h, w, angle)
        out = self._ensure_out(h, w, img.dtype)

        self._affine(
            img,
            matrix,
            offset=offset,
            output_shape=(h, w),
            output=out,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        return {"rotated_image": out}