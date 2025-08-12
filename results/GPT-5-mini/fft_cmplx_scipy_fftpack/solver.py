from typing import Any
from scipy.fft import fftn as _fftn

class Solver:
    def solve(self, problem: Any, **kwargs) -> Any:
        """Minimal, low-overhead call into scipy.fft.fftn (pocketfft)."""
        return _fftn(problem, **kwargs)