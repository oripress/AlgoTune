from __future__ import annotations

from typing import Any, Dict, List

from scipy import signal

_fftconvolve = signal.fftconvolve

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, List[float]]:
        x = problem["signal_x"]
        y = problem["signal_y"]
        if len(x) == 0 or len(y) == 0:
            return {"convolution": []}
        mode = problem.get("mode", "full")
        return {"convolution": _fftconvolve(x, y, mode=mode).tolist()}