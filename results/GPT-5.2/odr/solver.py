from __future__ import annotations

from typing import Any

import numpy as np
import scipy.odr as odr

def _lin(B: np.ndarray, x: np.ndarray) -> np.ndarray:
    return B[0] * x + B[1]

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sx = np.asarray(problem["sx"], dtype=np.float64)
        sy = np.asarray(problem["sy"], dtype=np.float64)

        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(_lin)
        out = odr.ODR(data, model, beta0=[0.0, 1.0]).run()
        return {"beta": out.beta.tolist()}