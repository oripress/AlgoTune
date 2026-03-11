from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self) -> None:
        self.order = 3
        self.mode = "constant"
        self._shift = scipy.ndimage.shift
        self._out_cache: dict[tuple[tuple[int, ...], str], np.ndarray] = {}

    def _get_out(self, shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
        key = (shape, np.dtype(dtype).str)
        out = self._out_cache.get(key)
        if out is None:
            out = np.empty(shape, dtype=dtype)
            self._out_cache[key] = out
        return out

    def _integer_shift(self, image: np.ndarray, sr: int, sc: int) -> np.ndarray:
        out = self._get_out(image.shape, image.dtype)
        out.fill(0)

        nrows, ncols = image.shape
        src_r0 = max(0, -sr)
        src_r1 = min(nrows, nrows - sr) if sr >= 0 else nrows
        dst_r0 = max(0, sr)
        dst_r1 = dst_r0 + (src_r1 - src_r0)

        src_c0 = max(0, -sc)
        src_c1 = min(ncols, ncols - sc) if sc >= 0 else ncols
        dst_c0 = max(0, sc)
        dst_c1 = dst_c0 + (src_c1 - src_c0)

        if src_r1 > src_r0 and src_c1 > src_c0:
            out[dst_r0:dst_r1, dst_c0:dst_c1] = image[src_r0:src_r1, src_c0:src_c1]
        return out

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        raw_image = problem["image"]
        image = raw_image if isinstance(raw_image, np.ndarray) else np.asarray(raw_image)

        sr = float(problem["shift"][0])
        sc = float(problem["shift"][1])

        if sr == 0.0 and sc == 0.0:
            return {"shifted_image": image}

        if image.ndim != 2:
            try:
                shifted = self._shift(
                    image, (sr, sc), order=self.order, mode=self.mode
                )
            except Exception:
                return {"shifted_image": []}
            return {"shifted_image": shifted}

        nrows, ncols = image.shape

        if abs(sr) >= nrows or abs(sc) >= ncols:
            out = self._get_out(image.shape, image.dtype)
            out.fill(0)
            return {"shifted_image": out}

        if sr.is_integer() and sc.is_integer():
            return {"shifted_image": self._integer_shift(image, int(sr), int(sc))}

        try:
            out = self._get_out(image.shape, image.dtype)
            shifted = self._shift(
                image, (sr, sc), order=self.order, mode=self.mode, output=out
            )
        except Exception:
            return {"shifted_image": []}

        return {"shifted_image": shifted}