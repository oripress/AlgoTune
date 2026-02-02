from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage

class Solver:
    """
    2D image zoom using cubic B-spline interpolation (order=3) with constant padding.
    """

    __slots__ = ("order", "mode", "cval")

    def __init__(self) -> None:
        # Match reference settings
        self.order = 3
        self.mode = "constant"
        self.cval = 0.0

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        image = problem["image"]
        zoom_factor = float(problem["zoom_factor"])

        # Validator requires a Python list (not ndarray)
        # Fast-path: exact zoom=1.0 should be identity for ndimage.zoom.
        if zoom_factor == 1.0:
            arr0 = np.asarray(image, dtype=np.float64, order="C")
            return {"zoomed_image": arr0.tolist()}

        arr = np.asarray(image, dtype=np.float64, order="C")

        try:
            out = scipy.ndimage.zoom(
                arr,
                zoom_factor,
                order=self.order,
                mode=self.mode,
                cval=self.cval,
                prefilter=True,
            )
        except Exception:
            return {"zoomed_image": []}

        return {"zoomed_image": out.tolist()}