from typing import Any, Dict, List

import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self, order: int = 3, mode: str = "constant") -> None:
        self.order = int(order)
        self.mode = str(mode)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        image = problem.get("image")
        matrix = problem.get("matrix")
        try:
            if image is None or matrix is None:
                return {"transformed_image": []}

            # Convert inputs to efficient NumPy arrays
            img = np.asarray(image, dtype=np.float64, order="C")
            m_arr = np.asarray(matrix, dtype=np.float64)

            # Fast path: exact identity (no translation)
            if m_arr.ndim == 2:
                if m_arr.shape == (2, 2):
                    if np.array_equal(m_arr, np.eye(2, dtype=m_arr.dtype)):
                        return {"transformed_image": list(img)}
                elif m_arr.shape == (2, 3):
                    # Identity linear part
                    if np.array_equal(m_arr[:, :2], np.eye(2, dtype=m_arr.dtype)):
                        t = m_arr[:, 2]
                        # If zero translation -> identity
                        if np.array_equal(t, np.zeros(2, dtype=m_arr.dtype)):
                            return {"transformed_image": list(img)}
                        # Fast integer translation with zero padding
                        t_rounded = np.rint(t)
                        if np.allclose(t, t_rounded, rtol=0.0, atol=0.0):
                            oy = int(t_rounded[0])
                            ox = int(t_rounded[1])
                            h, w = img.shape
                            out = np.zeros_like(img)
                            y_start = max(0, -oy)
                            y_end = min(h, h - oy)
                            x_start = max(0, -ox)
                            x_end = min(w, w - ox)
                            if y_start < y_end and x_start < x_end:
                                out[y_start:y_end, x_start:x_end] = img[
                                    y_start + oy : y_end + oy, x_start + ox : x_end + ox
                                ]
                            return {"transformed_image": list(out)}
            elif m_arr.ndim == 1 and m_arr.shape[0] == 2:
                # 1D vector (per-axis factors) equal to [1, 1]
                if np.array_equal(m_arr, np.ones(2, dtype=m_arr.dtype)):
                    return {"transformed_image": list(img)}

            # General case: use SciPy with preallocated output
            out = np.empty_like(img)
            scipy.ndimage.affine_transform(
                img,
                m_arr,
                order=self.order,
                mode=self.mode,
                cval=0.0,
                prefilter=True,
                output=out,
            )
            return {"transformed_image": list(out)}
        except Exception:
            return {"transformed_image": []}