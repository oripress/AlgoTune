from __future__ import annotations

from typing import Any

import numpy as np

try:
    # SciPy is available in this benchmark environment (used by reference/validator).
    from scipy.ndimage import shift as _ndi_shift
    from scipy.ndimage import spline_filter as _ndi_spline_filter
except Exception:  # pragma: no cover
    _ndi_shift = None
    _ndi_spline_filter = None

class Solver:
    __slots__ = (
        "order",
        "mode",
        "_shift_func",
        "_spline_filter_func",
        "_cache_shape",
        "_out",
        "_out_rows",
        "_coeff",
    )

    def __init__(self, order: int = 3, mode: str = "constant"):
        self.order = int(order)
        self.mode = str(mode)
        self._shift_func = _ndi_shift
        self._spline_filter_func = _ndi_spline_filter
        self._cache_shape: tuple[int, int] | None = None
        self._out: np.ndarray | None = None
        self._out_rows: list[np.ndarray] | None = None
        self._coeff: np.ndarray | None = None

    @staticmethod
    def _is_int_shift(x: float) -> bool:
        r = round(x)
        return abs(x - r) <= 1e-12

    def _ensure_buffers(self, shape: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        if (
            self._cache_shape != shape
            or self._out is None
            or self._out_rows is None
            or self._coeff is None
        ):
            out = np.empty(shape, dtype=np.float64)
            coeff = np.empty(shape, dtype=np.float64)
            self._out = out
            self._coeff = coeff
            self._out_rows = [out[i] for i in range(shape[0])]
            self._cache_shape = shape
        return self._out, self._out_rows, self._coeff  # type: ignore[return-value]

    @staticmethod
    def _int_shift_inplace(out: np.ndarray, img: np.ndarray, sr: int, sc: int) -> None:
        out.fill(0.0)
        n0, n1 = img.shape

        if sr >= 0:
            dst_r0 = sr
            src_r0 = 0
            lr = n0 - sr
        else:
            dst_r0 = 0
            src_r0 = -sr
            lr = n0 + sr
        if lr <= 0:
            return

        if sc >= 0:
            dst_c0 = sc
            src_c0 = 0
            lc = n1 - sc
        else:
            dst_c0 = 0
            src_c0 = -sc
            lc = n1 + sc
        if lc <= 0:
            return

        out[dst_r0 : dst_r0 + lr, dst_c0 : dst_c0 + lc] = img[
            src_r0 : src_r0 + lr, src_c0 : src_c0 + lc
        ]

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        image = problem["image"]
        shift_vector = problem["shift"]

        try:
            sr = float(shift_vector[0])
            sc = float(shift_vector[1])
        except Exception:
            return {"shifted_image": []}

        if not (np.isfinite(sr) and np.isfinite(sc)):
            return {"shifted_image": []}

        img = np.asarray(image)
        if img.ndim != 2:
            return {"shifted_image": []}
        if not (img.flags.c_contiguous and img.dtype == np.float64):
            img = np.asarray(img, dtype=np.float64, order="C")

        # Exact no-op shift: avoid any heavy work.
        if sr == 0.0 and sc == 0.0:
            return {"shifted_image": list(img)}

        out, out_rows, coeff = self._ensure_buffers(img.shape)

        # Integer shift fast path (exact).
        if self._is_int_shift(sr) and self._is_int_shift(sc):
            self._int_shift_inplace(out, img, int(round(sr)), int(round(sc)))
            return {"shifted_image": out_rows}

        shift_func = self._shift_func
        spline_filter_func = self._spline_filter_func
        if shift_func is None or spline_filter_func is None:
            return {"shifted_image": []}

        # Key optimization: compute spline coefficients into a reusable buffer,
        # then do the shift with prefilter disabled (so shift won't allocate coefficients).
        spline_filter_func(
            img,
            order=self.order,
            output=coeff,
            mode=self.mode,
        )
        shift_func(
            coeff,
            (sr, sc),
            output=out,
            order=self.order,
            mode=self.mode,
            cval=0.0,
            prefilter=False,
        )
        return {"shifted_image": out_rows}