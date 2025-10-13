from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.ndimage as ndi

class Solver:
    def __init__(self) -> None:
        # Settings must match the validator/reference
        self.order = 3
        self.mode = "constant"
        self.cval = 0.0

    @staticmethod
    def _as_float_array(x: Any) -> np.ndarray:
        arr = np.asarray(x)
        # Ensure 2D float64 array
        if arr.dtype.kind != "f":
            arr = arr.astype(np.float64, copy=False)
        else:
            # Promote to float64 if lower precision for consistent numerics
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64, copy=False)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _is_integer_shift(shift: Sequence[float], tol: float = 1e-12) -> tuple[bool, tuple[int, int]]:
        if len(shift) != 2:
            return False, (0, 0)
        sr, sc = float(shift[0]), float(shift[1])
        ir = int(round(sr))
        ic = int(round(sc))
        if abs(sr - ir) <= tol and abs(sc - ic) <= tol:
            return True, (ir, ic)
        return False, (ir, ic)

    @staticmethod
    def _integer_shift2d(img: np.ndarray, ir: int, ic: int) -> np.ndarray:
        # Constant padding with zeros outside
        out = np.zeros_like(img, dtype=np.float64)
        nrows, ncols = img.shape

        # Destination slice ranges
        dr_start = max(0, ir)
        dr_end = min(nrows, ir + nrows)
        dc_start = max(0, ic)
        dc_end = min(ncols, ic + ncols)

        if dr_start >= dr_end or dc_start >= dc_end:
            return out  # Entirely shifted out

        # Corresponding source slice ranges
        sr_start = max(0, -ir)
        sc_start = max(0, -ic)

        out[dr_start:dr_end, dc_start:dc_end] = img[
            sr_start : sr_start + (dr_end - dr_start),
            sc_start : sc_start + (dc_end - dc_start),
        ]
        return out

    def solve(self, problem, **kwargs) -> Any:
        """
        Shift a 2D image using cubic spline interpolation (order=3)
        and constant boundary mode (padding with 0).
        Returns a dict with key 'shifted_image' as a nested list.
        """
        try:
            image = problem["image"]
            shift_vec = problem["shift"]
        except Exception:
            return {"shifted_image": []}

        # Convert to float64 contiguous array
        img = self._as_float_array(image)

        # Validate shape (expect 2D)
        if img.ndim != 2:
            # Attempt to coerce to 2D; if impossible, fail gracefully
            try:
                img = np.atleast_2d(img).astype(np.float64, copy=False)
            except Exception:
                return {"shifted_image": []}

        # Fast path: exact integer shift
        is_int, (ir, ic) = self._is_integer_shift(shift_vec)
        if is_int:
            out = self._integer_shift2d(img, ir, ic)
            return {"shifted_image": out.tolist()}

        # General path: use scipy.ndimage.shift with cubic spline interpolation
        try:
            # Use output preallocation to reduce temporary allocations
            out = np.empty_like(img, dtype=np.float64)
            ndi.shift(
                img,
                shift=tuple(float(s) for s in shift_vec),
                order=self.order,
                mode=self.mode,
                cval=self.cval,
                output=out,
                prefilter=True,
            )
        except Exception:
            return {"shifted_image": []}

        return {"shifted_image": out.tolist()}